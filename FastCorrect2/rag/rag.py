import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

# from multiprocessing import context # 'context' is unused
from sentence_transformers import SentenceTransformer
import faiss
from pycorrector.gpt.gpt_corrector import GptCorrector
from load import load_terms_from_db
import logging  # Add logging for better feedback
import json  # Add json import for saving results

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 加载企业词表
logging.info("Loading terms from database...")
terms = load_terms_from_db()
if not terms:
    logging.error("Failed to load terms. Exiting.")
    exit()
logging.info(f"Loaded {len(terms)} terms.")

logging.info("Loading sentence transformer model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
logging.info("Encoding term descriptions...")
# Ensure descriptions are strings
descriptions = [str(term.get("description", "")) for term in terms]
embeddings = model.encode(descriptions)

# 构建FAISS索引
logging.info("Building FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
logging.info("FAISS index built successfully.")


# 监听企业数据源变化（如CRM系统） - Placeholder function
def on_term_updated(term):
    """Handles incremental updates to the term index."""
    try:
        description = str(term.get("description", ""))
        if description:
            embedding = model.encode([description])  # Encode expects a list
            index.add(embedding)
            logging.info(f"Term '{term.get('term')}' added/updated in index.")
        else:
            logging.warning(
                f"Term '{term.get('term')}' has no description, skipping index update."
            )
    except Exception as e:
        logging.error(f"Error updating index for term '{term.get('term')}': {e}")


def retrieve_context(query_text, top_k=3):
    """Retrieves relevant terms based on query text similarity to descriptions."""
    try:
        # Encode expects a list or string, ensure query_text is a string
        if not isinstance(query_text, str):
            logging.warning(
                f"retrieve_context expects a string, received {type(query_text)}. Converting."
            )
            query_text = str(query_text)

        query_embedding = model.encode([query_text])  # Pass as a list
        distances, indices = index.search(query_embedding, top_k)
        # Filter out results with low similarity (optional, requires IndexFlatL2 for distances)
        # Or check distances if using IndexFlatIP (dot product > threshold)
        retrieved_terms = [terms[i] for i in indices[0]]
        logging.debug(
            f"Query: '{query_text}', Retrieved indices: {indices[0]}, Terms: {[t['term'] for t in retrieved_terms]}"
        )
        return retrieved_terms
    except Exception as e:
        logging.error(f"Error during context retrieval for query '{query_text}': {e}")
        return []


# 构建Prompt模板
def build_prompt(query_text, contexts):
    """Builds the prompt for the correction model."""
    if not contexts:
        logging.warning(
            f"No context retrieved for query: '{query_text}'. Using base prompt."
        )
        context_str = "无相关企业术语信息。"
    else:
        context_str = "\n".join(
            [
                f"- {ctx.get('term', 'N/A')}（别名：{', '.join(ctx.get('aliases', []))}）: {ctx.get('description', 'N/A')}"
                for ctx in contexts
            ]
        )
    # Add description to context for better understanding by LLM
    return f"已知相关企业术语信息：\n{context_str}\n\n请基于以上信息，修正以下文本中的潜在错误，特别是企业术语相关的错误：\n“{query_text}”"


sys_prompt = """你是一个专业的中文文本纠错助手。请仔细阅读提供的企业术语信息，并利用这些信息来识别和修正输入文本中的错误，尤其是与这些术语相关的错误。请直接输出修正后的文本。"""

logging.info("Initializing GptCorrector...")
# Consider adding error handling for model loading
try:
    m = GptCorrector("shibing624/chinese-text-correction-1.5b")
    logging.info("GptCorrector initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize GptCorrector: {e}")
    exit()


def correct_text_list(error_sentences):
    """Corrects a list of sentences using RAG."""
    corrected_results = []
    for sentence in error_sentences:
        logging.info(f"Processing sentence: '{sentence}'")
        contexts = retrieve_context(sentence)
        prompt_user_part = build_prompt(sentence, contexts)
        full_prompt = f"{sys_prompt}\n\n{prompt_user_part}"  # Combine system and user prompt parts

        logging.debug(
            f"Full prompt for '{sentence}':\n{full_prompt}"
        )  # Log the full prompt for debugging

        corrected_sentence = "Correction failed"  # Default value
        detail = None  # Default value

        try:
            # Call the correction model and handle its return value flexibly
            correction_output = m.correct(prompt_user_part, system_prompt=sys_prompt)

            # Determine how to extract corrected_sentence and detail based on output type/structure
            if isinstance(correction_output, tuple) and len(correction_output) == 2:
                corrected_sentence, detail = correction_output
            elif isinstance(
                correction_output, str
            ):  # If only the corrected string is returned
                corrected_sentence = correction_output
                detail = None  # No details available
            else:
                # Handle other potential return types or log a warning
                logging.warning(
                    f"Unexpected return format from m.correct for sentence '{sentence}'. Output: {correction_output}"
                )
                # Attempt to use the output as the corrected sentence, assuming it might be the primary result
                corrected_sentence = str(correction_output)
                detail = None

            logging.info(f"Source: '{sentence}' -> Target: '{corrected_sentence}'")

        except AttributeError:
            # ... (AttributeError handling remains the same) ...
            logging.error(
                "The 'correct' method might not be available in GptCorrector or works differently. Trying correct_batch with single item."
            )
            # Fallback or alternative: Use correct_batch for a single item
            try:
                batch_res = m.correct_batch(
                    [prompt_user_part], system_prompt=sys_prompt
                )
                if batch_res and len(batch_res) > 0:
                    # Assuming batch_res is a list of dictionaries like [{'source': '...', 'target': '...'}]
                    corrected_info = batch_res[0]
                    corrected_sentence = corrected_info.get(
                        "target", "Correction failed"
                    )
                    detail = corrected_info  # Store the whole dict as detail
                    # logging.info(
                    #     f"Source: '{sentence}' -> Target: '{corrected_sentence}' (using correct_batch)"
                    # )

                else:
                    logging.error(
                        f"correct_batch did not return expected results for: '{sentence}'"
                    )
                    # corrected_sentence remains "Correction failed"

            except Exception as e_batch:
                logging.error(
                    f"Error during text correction for sentence '{sentence}' using correct_batch: {e_batch}"
                )
                detail = str(e_batch)  # Store error as detail
                # corrected_sentence remains "Correction failed"

        except Exception as e:
            # Log the specific error, including the problematic sentence
            logging.error(
                f"Error during text correction for sentence '{sentence}': {e}",
                exc_info=True,  # Add traceback info
            )
            detail = str(e)  # Store error as detail
            # corrected_sentence remains "Correction failed"

        # Append result in the desired format
        corrected_results.append(
            {
                "source": sentence,
                "target": corrected_sentence,
                # "details": detail # Keep details internally if needed, but don't include in final list per request
            }
        )
        # Print in the simplified format
        print(f"Source: {sentence}")
        print(f"Target: {corrected_sentence}")
        print("-" * 20)

    return corrected_results


if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.json"  # Define an output file name for JSON results
    error_sentences = []
    logging.info(f"Reading sentences from {input_file}...")
    try:
        # read input.txt
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:  # Avoid processing empty lines
                    error_sentences.append(stripped_line)
        logging.info(f"Read {len(error_sentences)} sentences.")

        if error_sentences:
            logging.info("Starting text correction process...")
            results = correct_text_list(error_sentences)
            logging.info("Text correction process finished.")

            # Save results to a JSON file with the desired format
            logging.info(f"Saving results to {output_file}...")
            try:
                with open(output_file, "w", encoding="utf-8") as outfile:
                    json.dump(results, outfile, ensure_ascii=False, indent=4)
                logging.info(f"Results successfully saved to {output_file}.")
            except Exception as e_save:
                logging.error(f"Failed to save results to {output_file}: {e_save}")

        else:
            logging.info("Input file is empty or contains no valid sentences.")

    except FileNotFoundError:
        logging.error(f"Input file '{input_file}' not found.")
    except Exception as e:
        logging.error(f"An error occurred in the main execution block: {e}")
