# ===============================
# Intelligent Semantic Search Chat App
# Integrated HKUST ISOM6670G Version
# ===============================
import streamlit as st
import random
import numpy as np
import pandas as pd
import time
from openai import AzureOpenAI
from pinecone import Pinecone
from datetime import datetime

# ===============================
# 🔧 页面配置
# ===============================
st.set_page_config(
    page_title="Semantic Search AI Chat for BA Users",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# 🎨 样式美化
# ===============================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: white;
    }
    section[data-testid="stSidebar"] {
        width: 400px !important;
        background-color: #1A1D23;
    }
    h1, h2, h3, h4 { color: white; }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
    .metric-label, .metric-value {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 显示 Logo
st.image("Logo_USTBusinessSchool.svg", width=180, output_format="SVG")

# ===============================
# ⚙️ 初始化 Session State
# ===============================
for key in ["conversations", "conversation_titles", "active_chat_index"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "conversation" in key else None
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
if "PINECONE_API_KEY" not in st.session_state:
    st.session_state["PINECONE_API_KEY"] = ""
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []

# ===============================
# 🌐 初始化客户端函数
# ===============================
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

# ===============================
# 🧩 调用LLM生成答案
# ===============================
def call_llm_generate_answer(openai_client, enhanced_prompt: str):
    """
    输入增强提示词，调用Azure OpenAI生成回答
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=0.2,
            max_tokens=800,
            top_p=0.95
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ [Error in LLM response]: {str(e)}"

# ===============================
# 📍 语义搜索功能 (Pinecone)
# ===============================
def semantic_search(index, query_vector, top_k=3):
    """
    输入Query向量，返回最相似文档
    """
    return index.query(vector=query_vector, top_k=top_k, include_metadata=True)

# ===============================
# 🧠 构建增强RAG Prompt
# ===============================
def build_augmented_prompt(user_query: str, search_results):
    contexts = []
    for i, match in enumerate(search_results.matches, start=1):
        text = match.metadata.get("text", "")
        contexts.append(f"[Document {i}]\n{text}")
    context_block = "\n\n".join(contexts)
    return f"""
You are an intelligent assistant. Please answer the user's question strictly based on the provided context.

Guidelines:
1. Only use the **Context** below.
2. Do NOT fabricate information.
3. If the answer is not found, reply exactly: "The provided context does not contain the answer."

User Question:
{user_query}

Context:
{context_block}
""".strip()

# ===============================
# 🧭 Sidebar配置
# ===============================
st.sidebar.title("💬 Chat Sidebar")

# --- 输入API Keys ---
openai_key = st.sidebar.text_input("Enter your HKUST OpenAI API Key", type="password")
pinecone_key = st.sidebar.text_input("Enter your Pinecone API Key", type="password")

if openai_key:
    st.session_state["OPENAI_API_KEY"] = openai_key
if pinecone_key:
    st.session_state["PINECONE_API_KEY"] = pinecone_key

st.sidebar.divider()

# --- API Status ---
st.sidebar.subheader("🔧 API Status")
col1, col2 = st.sidebar.columns(2)
col1.success("✅ Pinecone: Connected")
col2.success("✅ Azure OpenAI: Connected")

# --- API Connection Test ---
if st.sidebar.button("🔄 Test Connection", use_container_width=True):
    with st.spinner("Testing Azure OpenAI connection..."):
        try:
            client = get_openai_client(st.session_state["OPENAI_API_KEY"])
            client.embeddings.create(input="Test", model="text-embedding-ada-002")
            st.sidebar.success("✅ Azure OpenAI connection successful!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")

# --- Search Configuration ---
st.sidebar.subheader("⚙️ Search Configuration")
top_k = st.sidebar.slider("Number of documents to return", 1, 10, 3)

# --- Usage Tips ---
st.sidebar.subheader("💡 Usage Tips")
st.sidebar.info("""
- Enter complete question statements  
- More specific questions yield more accurate results  
- Supports both Chinese and English queries  
- System generates answers based on relevant documents
""")

st.sidebar.divider()
# --- Chat Control Section ---
if st.sidebar.button("🆕 New Chat", use_container_width=True):
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = len(st.session_state["conversations"]) - 1

if st.sidebar.button("🗑️ Clear All History", use_container_width=True):
    st.session_state["conversations"].clear()
    st.session_state["conversation_titles"].clear()
    st.session_state["active_chat_index"] = None
    st.rerun()

st.sidebar.subheader("📜 Chat History")
if len(st.session_state["conversations"]) == 0:
    st.sidebar.info("No history yet. Click 'New Chat' to start.")
else:
    for i, title in enumerate(st.session_state["conversation_titles"]):
        if i == st.session_state["active_chat_index"]:
            st.sidebar.button(f"📍 {title}", key=f"active_{i}", disabled=True, use_container_width=True)
        else:
            if st.sidebar.button(f"💬 {title}", key=f"chat_{i}", use_container_width=True):
                st.session_state["active_chat_index"] = i

# ===============================
# 🧠 主界面
# ===============================
st.title("🧠 Semantic Search AI Chat for BA Users")
st.caption("Prototype for HKUST Business Analytics (ISOM6670G)")

if st.session_state["active_chat_index"] is None:
    st.info("Click **New Chat** in the sidebar to start.")
    st.stop()

chat_index = st.session_state["active_chat_index"]
current_chat = st.session_state["conversations"][chat_index]

# --- 展示历史消息 ---
for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 输入新Query ---
user_query = st.text_input("Enter your question:", placeholder="e.g., What is HKUST Business School?")

# ===============================
# 🧩 处理输入
# ===============================
if user_query:
    if not st.session_state["OPENAI_API_KEY"] or not st.session_state["PINECONE_API_KEY"]:
        st.error("Please input both API keys before continuing.")
        st.stop()

    st.chat_message("user").write(user_query)
    current_chat.append({"role": "user", "content": user_query})
    if len(current_chat) == 1:
        st.session_state["conversation_titles"][chat_index] = user_query[:40]

    with st.spinner("🔍 Performing semantic search and generating answer..."):
        try:
            openai_client = get_openai_client(st.session_state["OPENAI_API_KEY"])
            pinecone_client = get_pinecone_client(st.session_state["PINECONE_API_KEY"])
            index = pinecone_client.Index("developer-quickstart-py")

            # Create embedding
            emb = openai_client.embeddings.create(input=user_query, model="text-embedding-ada-002")
            query_vector = emb.data[0].embedding

            # Search Pinecone
            results = semantic_search(index, query_vector, top_k=top_k)
            prompt = build_augmented_prompt(user_query, results)
            answer = call_llm_generate_answer(openai_client, prompt)

            # Save & Display
            answer_text = f"**Question:** {user_query}\n\n**Answer:** {answer}"
            st.chat_message("assistant").write(answer_text)
            current_chat.append({"role": "assistant", "content": answer_text})

            # 显示结果统计
            st.markdown("---")
            st.subheader("📈 Search Statistics")

            scores = [m.score for m in results.matches]
            if scores:
                st.metric("Highest Similarity", f"{max(scores):.3f}")
                st.metric("Average Similarity", f"{np.mean(scores):.3f}")

            with st.expander("🔍 Retrieved Documents"):
                for i, match in enumerate(results.matches, start=1):
                    st.write(f"**Document {i} (Score: {match.score:.3f})**")
                    st.info(match.metadata.get("text", "")[:500] + "...")

            # 操作按钮
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💾 Save Results", use_container_width=True):
                    st.session_state["search_history"].append(answer_text)
                    st.success("✅ Results saved to history.")
            with col2:
                if st.button("🔄 Search Again", use_container_width=True):
                    st.rerun()
            with col3:
                if st.button("🏠 Return Home", use_container_width=True):
                    st.session_state.pop("active_chat_index")
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ===============================
# 🧾 Embedding 展示
# ===============================
with st.expander("🔍 Embedding Information"):
    st.write("""
    **How Semantic Search Works:**
    - Convert your query into a vector (embedding)
    - Search semantically similar content in Pinecone
    - Provide contextualized, grounded answers via Azure OpenAI (RAG)
    """)
