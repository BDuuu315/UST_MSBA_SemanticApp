# =============================================================
# RAG System with Azure OpenAI + Pinecone (integrated version)
# =============================================================

from openai import AzureOpenAI
from pinecone import Pinecone
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any

# =============================================================
# 1️⃣ Azure OpenAI 初始化
# =============================================================
openai_client = AzureOpenAI(
    api_key="aed78ad4701e4823ad0e7e233c877b8c",   # ⚠️ 请替换为你自己的 API KEY
    api_version="2023-05-15",
    azure_endpoint="https://hkust.azure-api.net"
)

# =============================================================
# 2️⃣ 替换后的 Pinecone 初始化与语义检索逻辑
# =============================================================

def get_pinecone_client():
    """
    初始化 Pinecone 客户端（使用 Streamlit 版本配置）
    """
    pc = Pinecone(api_key="pcsk_JPQMS_zQZ9MfrD4aSEe8b69PoxsjcsvoSPEHpzgYGt4GPm8bv7ED95Wjy4u7vPmxSnjj")
    index = pc.Index(
        name="developer-quickstart-py",
        host="https://developer-quickstart-py-9d1pu2j.svc.aped-4627-b74a.pinecone.io"
    )
    return index


def semantic_search(user_query: str, openai_client, top_k: int = 5):
    """
    语义检索：使用 Azure 生成 embedding -> Pinecone 搜索相似文档
    """
    # === Step 1. 生成 query 向量 ===
    emb = openai_client.embeddings.create(
        input=user_query,
        model="text-embedding-ada-002"
    )
    query_vector = emb.data[0].embedding

    # === Step 2. 检索 Pinecone ===
    index = get_pinecone_client()
    search_resp = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # === Step 3. 输出检索日志 ===
    print(f"\nQuery: {user_query}\n")
    print("-" * 60)
    for i, match in enumerate(search_resp.matches, 1):
        text = match.metadata.get("text", "[no text]")
        print(f"[{i}] Score: {match.score:.4f} | {text[:120]}{'...' if len(text)>120 else ''}")

    return query_vector, search_resp


# =============================================================
# 3️⃣ 构建增强 Prompt（RAG）
# =============================================================

def build_augmented_prompt(user_query: str, search_results) -> str:
    """
    结合用户问题与 Pinecone 检索到的文档，构建 RAG Prompt
    """
    context_chunks = []

    for i, match in enumerate(search_results.matches, 1):
        doc_text = (
            match.metadata.get("text")
            or match.metadata.get("chunk_text", "")
        )
        context_chunks.append(f"[Document {i}]\n{doc_text}")

    context_block = "\n\n".join(context_chunks)

    augmented_prompt = f"""
You are an intelligent assistant. Please answer the user's question
strictly based on the context provided below.

Guidelines:
1. Only use the information from the **Context** section.
2. Do NOT fabricate or guess.
3. If the answer is not present in the context, reply with:
   "The provided context does not contain the answer."

User Query:
{user_query}

Context:
{context_block}
""".strip()

    return augmented_prompt


# =============================================================
# 4️⃣ RAG 主流程（使用 Azure OpenAI 回答）
# =============================================================

def rag_answer_with_azure(
    user_question: str,
    openai_client,
    top_k: int = 5,
    model: str = "gpt-35-turbo",
    temperature: float = 0.2,
    max_tokens: int = 1536
) -> Dict[str, Any]:
    """
    综合 RAG 检索 + Azure 回答
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] User question: {user_question}")

    # 检索 Pinecone
    query_vec, search_results = semantic_search(user_question, openai_client, top_k=top_k)

    # 构建 RAG 提示词
    aug_prompt = build_augmented_prompt(user_question, search_results)

    print("\n" + "=" * 80)
    print("Final RAG Prompt sent to LLM (preview):")
    print("=" * 80)
    print(aug_prompt[:1000] + "\n...")

    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Calling Azure {model}...")
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": aug_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.8,
            presence_penalty=0.2,
            frequency_penalty=0.2
        )

        answer = response.choices[0].message.content.strip()
        usage = response.usage

        print(f"Token usage → prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, total: {usage.total_tokens}")

        return {
            "query": user_question,
            "answer": answer,
            "model": model,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            },
            "timestamp": datetime.now().isoformat(),
            "results": [m.metadata for m in search_results.matches]
        }

    except Exception as e:
        error_msg = f"[ChatGPT Calling Failed] {str(e)}"
        print(error_msg)
        return {
            "answer": "An error occurred while generating the response.",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# =============================================================
# 5️⃣ 主程序入口（调试用示例）
# =============================================================
if __name__ == "__main__":
    query = "What is disease prevention?"
    rag_result = rag_answer_with_azure(query, openai_client, top_k=5)
    
    print("\n" + "=" * 80)
    print("💡 Final Answer:")
    print("=" * 80)
    print(rag_result["answer"])
