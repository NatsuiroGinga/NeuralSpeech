import numpy as np
from sentence_transformers import SentenceTransformer


def retrieve_context(query_text, top_k=3):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 确保 query_embedding 是二维的
    query_embedding = model.encode([query_text])  # 输入为列表，自动生成二维数组
    query_embedding = np.array(
        query_embedding, dtype=np.float32
    )  # 显式转换为 numpy 数组

    # 执行搜索并验证返回值
    distances, indices = index.search(query_embedding, top_k)

    # 处理单条查询结果（FAISS 返回的 indices 是二维数组）
    return [terms[i] for i in indices[0]]
