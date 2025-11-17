# semantic_search_app.py
# ========================================================
# HKUST ISOM 6670G - Semantic Search Chat (RAG Demo)
# ========================================================

import os
import random
import streamlit as st
from datetime import datetime
from typing import Dict, Any

from openai import AzureOpenAI
from pinecone import Pinecone

# ========================================================
# ✅ 基础配置
# ========================================================

st.set_page_config(page_title="Stop watching GuHao", layout="centered")

st.title("🔍 Semantic Search + AI Chat (RAG)")
st.caption("Prototype for HKUST Business Analytics (ISOM6670G)")

# ========================================================
# ✅ 初始化状态
# ========================================================

if "conversations" not in st.session_state:
    st.session_state["conversations"] = []
if "conversation_titles" not in st.session_state:
    st.session_state["conversation_titles"] = []
if "active_chat_index" not in st.session_state:
    st.session_state["active_chat_index"] = None

# ========================================================
# ✅ Sidebar
# ========================================================

st.sidebar.header("App Settings")

# API Keys (从环境变量/Secrets中读取)
openai_key = st.sidebar.text_input("Enter your HKUST Azure OpenAI Key:", type="password")
pinecone_key = st.sidebar.text_input("Enter your Pinecone API Key:", type="password")

# 保存 session 内部状态
if openai_key:
    st.session_state["OPENAI_API_KEY"] = openai_key
if pinecone_key:
    st.session_state["PINECONE_API_KEY"] = pinecone_key

st.sidebar.divider()

# 新建会话按钮
if st.sidebar.button("🆕 New Chat", use_container_width=True):
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = len(st.session_state["conversations"]) - 1

# 清除所有历史
if st.sidebar.button("🗑️ Clear All Chats", use_container_width=True):
    st.session_state["conversations"].clear()
    st.session_state["conversation_titles"].clear()
    st.session_state["active_chat_index"] = None
    st.rerun()

# --- 历史区块 + 删除单条按钮 ---
st.sidebar.subheader("🧾 Chat History")

if not st.session_state["conversations"]:
    st.sidebar.info("No history yet. Create a new chat to start.")
else:
    for i, title in enumerate(st.session_state["conversation_titles"]):
        cols = st.sidebar.columns([5, 1])
        with cols[0]:
            label = f"🟢 {title}" if i == st.session_state["active_chat_index"] else title
            disabled = i == st.session_state["active_chat_index"]
            if st.button(label, key=f"chat_{i}", disabled=disabled, use_container_width=True):
                st.session_state["active_chat_index"] = i
        with cols[1]:
            if st.button("🗑️", key=f"del_{i}", help="Delete this chat"):
                del st.session_state["conversations"][i]
                del st.session_state["conversation_titles"][i]
                if st.session_state["active_chat_index"] == i:
                    st.session_state["active_chat_index"] = None
                elif st.session_state["active_chat_index"] and st.session_state["active_chat_index"] > i:
                    st.session_state["active_chat_index"] -= 1
                st.rerun()

# ========================================================
# ✅ 辅助函数：Client 初始化
# ========================================================

@st.cache_resource
def get_openai_client(api_key: str):
    """初始化 Azure OpenAI 客户端"""
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )


@st.cache_resource
def get_pinecone_client(api_key: str):
    """初始化 Pinecone 客户端"""
    return Pinecone(api_key=api_key)


# ========================================================
# ✅ 主区逻辑
# ========================================================

if st.session_state["active_chat_index"] is None:
    st.info("Click **New Chat** in the sidebar to start.")
    st.stop()

chat_index = st.session_state["active_chat_index"]
current_chat = st.session_state["conversations"][chat_index]

# 展示历史消息
for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入区
user_query = st.chat_input("Type your question here...")

if user_query:
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("🔐 Please enter your Azure OpenAI key in the sidebar.")
        st.stop()

    if not st.session_state.get("PINECONE_API_KEY"):
        st.error("🔐 Please enter your Pinecone API key in the sidebar.")
        st.stop()

    # --- Step 1: 显示并存储用户输入 ---
    st.chat_message("user").write(user_query)
    current_chat.append({"role": "user", "content": user_query})
    if len(current_chat) == 1:
        st.session_state["conversation_titles"][chat_index] = user_query[:40]

    # --- Step 2: 生成 embedding，搜索 Pinecone ---
    with st.spinner("Generating embedding & searching Pinecone..."):
        try:
            openai_client = get_openai_client(st.session_state["OPENAI_API_KEY"])
            pc = get_pinecone_client(st.session_state["PINECONE_API_KEY"])

            # 1️⃣ 获取 embedding
            emb = openai_client.embeddings.create(
                input=user_query,
                model="text-embedding-ada-002"
            )
            query_vector = emb.data[0].embedding

            # 2️⃣ 查询 Pinecone Index
            index_name = "developer-quickstart-py"
            idx = pc.Index(index_name)
            search_results = idx.query(vector=query_vector, top_k=3, include_metadata=True)

            # 拼接上下文
            context_str = ""
            for match in search_results.matches:
                src = match.metadata.get("text", "")
                context_str += f"[score={match.score:.2f}] {src}\n"

        except Exception as e:
            st.error(f"Error connecting to Pinecone/OpenAI: {e}")
            st.stop()

    # --- Step 3: 调用 GPT (RAG) 输出 ---
    with st.spinner("Generating AI answer..."):
        try:
            enhanced_prompt = f"""
You are a knowledgeable BA tutor at HKUST.
Use only the provided context to answer the user's question.
Be concise, factual, and cite sources if available.

Context:
{context_str}

Question:
{user_query}
"""

            response = openai_client.chat.completions.create(
                model="gpt-4o",  # 或 "gpt-4o-mini" 视部署配置而定
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0.2,
                max_tokens=500,
            )

            answer = response.choices[0].message.content.strip()
            full_answer = f"**Answer:**\n{answer}\n\n---\n**Context Preview:**\n{context_str[:400]}"
        except Exception as e:
            full_answer = f"⚠️ LLM generation error: {str(e)}"

    # --- Step 4: 显示并存储回答 ---
    st.chat_message("assistant").write(full_answer)
    current_chat.append({"role": "assistant", "content": full_answer})

# ========================================================
# ✅ 底部说明
# ========================================================
with st.expander("ℹ️ About this App"):
    st.markdown("""
    **Semantic Search Workflow:**
    1. Convert user query → embedding vector
    2. Retrieve top-k most similar document chunks from Pinecone
    3. Inject retrieved context into GPT prompt (RAG)
    4. Generate answer grounded in retrieved information
    """)
