import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

from multiprocessing import context
from sentence_transformers import SentenceTransformer
import faiss
from pycorrector.gpt.gpt_corrector import GptCorrector
from load import load_terms_from_db


# 加载企业词表
terms = load_terms_from_db()
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode([term["description"] for term in terms])

# 构建FAISS索引
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)


# 监听企业数据源变化（如CRM系统）
def on_term_updated(term):
    embedding = model.encode(term["description"])
    index.add(embedding)  # 增量更新索引


def retrieve_context(query_text, top_k=4):
    query_embedding = model.encode(query_text)
    _, indices = index.search(query_embedding, top_k)
    return [terms[i] for i in indices[0]]


# 构建Prompt模板
def build_prompt(query_text, contexts):
    context_str = "\n".join(
        [f"- {ctx['term']}（别名：{', '.join(ctx['aliases'])}）" for ctx in contexts]
    )
    return f"已知企业术语：\n{context_str}\n请修正以下文本：{query_text}"


sys_prompt = """你是一个专业的中文文本纠错助手，基于已知的企业术语生成纠正后的文本"""

m = GptCorrector("shibing624/chinese-text-correction-1.5b")


def correct_text(query_text):
    contexts = retrieve_context(query_text)
    # prompt = build_prompt(query_text, contexts)
    prompt = f"{sys_prompt}\n\n{build_prompt(query_text, contexts)}"
    print(prompt)
    print()

    batch_res = m.correct_batch(
        error_sentences,
        system_prompt=prompt,
    )
    for i in batch_res:
        print(i)
        print()


if __name__ == "__main__":
    error_sentences = []
    # read input.txt
    with open("input.txt", "r", encoding="utf-8") as f:
        for line in f:
            error_sentences.append(line.strip())
    correct_text(error_sentences)
