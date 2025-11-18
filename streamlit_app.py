# =====================================================
# ✅ 新版 Pinecone 初始化与语义搜索函数
# =====================================================

from pinecone import Pinecone
import numpy as np
import os

# 初始化 Pinecone 客户端
def get_pinecone_client() -> Pinecone.Index:
    pc = Pinecone(api_key="pcsk_JPQMS_zQZ9MfrD4aSEe8b69PoxsjcsvoSPEHpzgYGt4GPm8bv7ED95Wjy4u7vPmxSnjj")
    # 根据 Streamlit 代码配置的 Index 名称与 Host
    index = pc.Index(
        name="developer-quickstart-py",
        host="https://developer-quickstart-py-9d1pu2j.svc.aped-4627-b74a.pinecone.io"
    )
    return index


# 语义检索函数
def semantic_search(user_query: str, openai_client, top_k: int = 5):
    """
    给定自然语言查询 -> Azure OpenAI 生成向量 -> Pinecone 搜索最相似的文档
    """
    # 生成 query embedding
    emb = openai_client.embeddings.create(
        input=user_query,
        model="text-embedding-ada-002"
    )
    query_vector = emb.data[0].embedding

    # 检索 Pinecone
    index = get_pinecone_client()
    search_resp = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # 输出调试信息
    print(f"\nQuery: {user_query}\n")
    print("-" * 60)
    for i, match in enumerate(search_resp.matches, 1):
        text = match.metadata.get("text", "[no text]")
        print(f"[{i}] Score: {match.score:.4f} | {text[:120]}{'...' if len(text)>120 else ''}")

    return query_vector, search_resp
