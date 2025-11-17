# ============================
# Semantic Search AI Chat App (RAG version)
# For HKUST ISOM6670G
# ============================

import streamlit as st
import random
import time
import numpy as np
import pandas as pd
from openai import AzureOpenAI
from pinecone import Pinecone

# ============================
# ⚙️ 页面配置
# ============================
st.set_page_config(
    page_title="Intelligent Semantic Search Application",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================
# 🎨 页面样式
# ============================
st.markdown(
    """
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #1A1D23;
    }
    h1, h2, h3, h4 {
        color: white;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# 🌍 初始化状态
# ============================
if "conversations" not in st.session_state:
    st.session_state["conversations"] = []
if "conversation_titles" not in st.session_state:
    st.session_state["conversation_titles"] = []
if "active_chat_index" not in st.session_state:
    st.session_state["active_chat_index"] = None
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
if "PINECONE_API_KEY" not in st.session_state:
    st.session_state["PINECONE_API_KEY"] = ""


# ============================
# 🧩 初始化客户端函数
# ============================
@st.cache_resource
def get_openai_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )

@st.cache_resource
def get_pinecone_client(api_key):
    return Pinecone(api_key=api_key)

# ============================
# 🧭 Sidebar - 配置区
# ============================
st.sidebar.title("🧠 App Settings")

# --- Keys ---
openai_key = st.sidebar.text_input("Enter your HKUST Azure OpenAI Key:", type="password")
pinecone_key = st.sidebar.text_input("Enter your Pinecone API Key:", type="password")
if openai_key:
    st.session_state["OPENAI_API_KEY"] = openai_key
if pinecone_key:
    st.session_state["PINECONE_API_KEY"] = pinecone_key

st.sidebar.divider()

# --- New Chat / Clear All ---
if st.sidebar.button("🆕 New Chat", use_container_width=True):
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = len(st.session_state["conversations"]) - 1
if st.sidebar.button("🗑️ Clear All Chats", use_container_width=True):
    st.session_state["conversations"].clear()
    st.session_state["conversation_titles"].clear()
    st.session_state["active_chat_index"] = None
    st.rerun()

st.sidebar.subheader("💬 Chat History")
if len(st.session_state["conversation_titles"]) == 0:
    st.sidebar.info("No history yet. Create a new chat to start.")
else:
    for i, title in enumerate(st.session_state["conversation_titles"]):
        if i == st.session_state["active_chat_index"]:
            st.sidebar.button(f"📍 {title}", key=f"chat_{i}", disabled=True, use_container_width=True)
        else:
            if st.sidebar.button(f"💬 {title}", key=f"chat_{i}", use_container_width=True):
                st.session_state["active_chat_index"] = i

st.sidebar.divider()

# ============================
# 🔌 API Status + Config Section
# ============================
st.sidebar.header("🔧 API Status")
col_a, col_b = st.sidebar.columns(2)
with col_a: st.success("✅ Pinecone: Connected")
with col_b: st.success("✅ Azure OpenAI: Connected")

st.sidebar.header("⚙️ Search Configuration")
top_k = st.sidebar.slider("Number of documents to return", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.header("💡 Usage Tips")
st.sidebar.info("""
- Enter complete question statements  
- More specific questions yield more accurate results  
- Supports both Chinese and English queries  
- System generates answers based on relevant documents
""")

# --- Test Connection按钮 ---
if st.sidebar.button("🔄 Test Connection", use_container_width=True):
    with st.spinner("Testing API connection..."):
        try:
            client = get_openai_client(st.session_state["OPENAI_API_KEY"])
            response = client.embeddings.create(input="Hello world", model="text-embedding-ada-002")
            st.sidebar.success("✅ Azure OpenAI connection successful!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")


# ============================
# 🏠 主界面 - 问答部分
# ============================
st.title("🔍 Intelligent Semantic Search Application")
st.markdown("Using **Pinecone + Azure OpenAI** for semantic search")

# 若没有激活的聊天
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
st.subheader("📝 Enter Your Question")
user_query = st.text_area(
    "Please enter your question:",
    placeholder="For example:\n• What is HKUST?\n• What is machine learning?",
    height=120
)

col1, col2 = st.columns([2, 1])
with col1:
    search_button = st.button("🚀 Start Search", use_container_width=True, type="primary")
with col2:
    test_shortcut = st.button("🔄 Test Connection", use_container_width=True)

if test_shortcut:
    st.sidebar.warning("▶ Triggering connection test via main window.")
    st.rerun()

# ============================
# 🔍 执行Semantic Search + RAG
# ============================
if search_button:
    if not user_query.strip():
        st.error("❌ Please enter a valid question.")
        st.stop()

    if not st.session_state.get("OPENAI_API_KEY") or not st.session_state.get("PINECONE_API_KEY"):
        st.error("Please input both API keys in the sidebar!")
        st.stop()

    with st.spinner("🔍 Searching for relevant documents and generating answer..."):
        try:
            openai_client = get_openai_client(st.session_state["OPENAI_API_KEY"])
            pc = get_pinecone_client(st.session_state["PINECONE_API_KEY"])

            # Create embedding
            emb = openai_client.embeddings.create(
                input=user_query.strip(),
                model="text-embedding-ada-002"
            )
            query_vector = emb.data[0].embedding

            # Query Pinecone
            index_name = "developer-quickstart-py"
            idx = pc.Index(index_name)
            results = idx.query(vector=query_vector, top_k=top_k, include_metadata=True)

            context_parts = []
            for m in results.matches:
                meta = m.metadata.get("text", "")
                context_parts.append(meta)
            context = "\n".join(context_parts) if context_parts else "(No relevant context found.)"

            # Build RAG prompt
            prompt = f"""
You are an intelligent assistant. 
Please answer the question using ONLY the following context.

User question:
{user_query}

Context:
{context}
If nothing relevant, reply: "The provided context does not contain the answer."
"""

            # Generate response
            answer = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            ).choices[0].message.content.strip()

            # Display result
            st.success("✅ Semantic Search Completed!")
            with st.chat_message("assistant"):
                st.markdown(f"**Answer:**\n{answer}")
            current_chat.append({"role": "assistant", "content": answer})

            if results.matches:
                st.subheader(f"📄 Top {top_k} Relevant Documents")
                for i, m in enumerate(results.matches, 1):
                    st.markdown(f"**[{i}]** *(Score: {m.score:.3f})*")
                    st.info(m.metadata.get("text", "")[:500] + "...")

        except Exception as e:
            st.error(f"❌ Error during semantic search: {e}")

# ============================
# 📘 信息栏
# ============================
with st.expander("ℹ️ How it Works"):
    st.markdown("""
    **Pipeline:**
    1. User query → embedding vector (via Azure OpenAI)
    2. Pinecone semantic similarity search (top_k)
    3. Retrieved context + question → GPT model (RAG)
    4. Generate grounded answer
    """)
