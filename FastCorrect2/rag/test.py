from typing import Any, Dict

retrieval_prompt = "为这个句子生成表示以用于检索相关文章:"
PROMPT = """根据上下文回答问题，如果知识片段不包含答案请回复"未知"，不要回复知识片段以外的知识，回复内容请保持精简，不要提供过多的分析
## 知识片段
----
{}
----
## 问题:
----
{}
----"""


def preprocess_dataset(
    examples,
    retriever_tokenizer,
    generator_tokenizer,
    query_column_name="query",
    passage_column_name="passage",
    answer_column_name="answer",
    query_max_len=128,
    passage_max_len=512,
    generator_max_len=512,
) -> Dict[str, Any]:
    # 1.prepare
    querie_list = examples[query_column_name]
    passage_list = examples[passage_column_name]
    answers = examples[answer_column_name]
    # 2. concat
    queries = [retrieval_prompt + query for query in querie_list]
    passages = [retrieval_prompt + passage for passage in passage_list]
    # 3 embedding
    retriever_query_tokens = retriever_tokenizer(
        queries, truncation=True, padding="max_length", max_length=query_max_len
    )
    retriever_passage_tokens = retriever_tokenizer(
        passages, truncation=True, padding="max_length", max_length=passage_max_len
    )
    # 4 input
    casual_input_text = []
    for passage, query, answer in zip(passage_list, querie_list, answers, strict=True):
        casual_input_text.append(
            """<|im_start|>user
            {}<|im_end|>
            <|im_start|>assistant
            {}""".format(
                PROMPT.format(passage, query)
            ),
            answer,
        )

    casual_input_tokens = generator_tokenizer(
        casual_input_text,
        truncation=True,
        padding="max_length",
        max_length=generator_max_len,
    )

    query_passage_text = []
    for query, passage in zip(querie_list, passage_list):
        query_passage_text.append(
            """<|im_start|>user
            {}<|im_end|>
            <|im_start|>assistant
            {}""".format(
                PROMPT.format(passage, query)
            )
        )

    query_passage_lengths = []
    query_passage_tokens = generator_tokenizer(
        query_passage_text,
        padding=False,
    )

    for single_query_passage in query_passage_tokens["input_ids"]:
        query_passage_lengths.append(len(single_query_passage))

    pre_batch = {}

    for k, v in retriever_query_tokens.items():
        pre_batch[f"retriever_query_{k}"] = v
    for k, v in retriever_passage_tokens.items():
        pre_batch[f"retriever_passage_{k}"] = v
    for k, v in casual_input_tokens.items():
        pre_batch[f"generator_input_{k}"] = v

    pre_batch["query_passage_input_len"] = query_passage_lengths
    return pre_batch
