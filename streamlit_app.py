import streamlit as st
import random
import numpy as np
import pandas as pd
from pinecone import Pinecone
from openai import AzureOpenAI

# ========== 配置区 ==========
PINECONE_API_KEY = "pcsk_6oHDXL_QyzEgtuEzHkTacffEiBW4gmGjPVfb4MAuz2Wy3M47yA5WR7XePPodEW1p6d6XyW"
PINECONE_INDEX_NAME = "geo-semantic"
PINECONE_HOST = "https://geo-semantic-u90uigv.svc.aped-4627-b74a.pinecone.io"
PINECONE_NAMESPACE = "__default__"
# =============================

# --- 页面设置 ---
st.set_page_config(page_title="Semantic Search AI Chat", layout="centered", initial_sidebar_state="expanded")

# --- 页面样式 ---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {position: relative;}
.logo {position: fixed; top: 10px; left: 15px; z-index: 100;}
section[data-testid="stSidebar"] {width: 380px !important; min-width: 380px !important; height: 100vh; overflow: auto;}
section[data-testid="stSidebar"] > div {width: 380px !important; padding-top: 2rem; height: 100%;}
.stSidebar .stButton>button {width: 100%;}
.main .block-container {padding-left: 400px; padding-right: 2rem;}
</style>
""", unsafe_allow_html=True)

st.image("Logo_USTBusinessSchool.svg", width=120, output_format="SVG")

# ========== 状态初始化 ==========
for key, default in {
    "conversations": [], "conversation_titles": [], "active_chat_index": None,
    "OPENAI_API_KEY": None, "documents": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state["documents"]:
    st.session_state["documents"] = [
        {"id": 1, "content": "HKUST Business School offers MBA programs with focus on analytics.", "embedding": None},
        {"id": 2, "content": "The ISOM department provides courses in information systems.", "embedding": None},
    ]

# ========== 初始化 Azure OpenAI 和 Pinecone ==========
@st.cache_resource
def get_azure_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )

@st.cache_resource
def get_pinecone_client():
    pc = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_HOST)
    return pc.Index(PINECONE_INDEX_NAME)

# ========== 侧边栏 ==========
st.sidebar.title("Chat Sidebar")

api_key = st.sidebar.text_input(
    "Enter your HKUST OpenAI API Key",
    type="password",
    help="You can check ISOM 6670G syllabus to get set-up instructions."
)
if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

st.sidebar.markdown("---")

# --- API 测试 ---
if st.sidebar.button("🔄 Test Connection", use_container_width=True):
    try:
        client = get_azure_client(st.session_state["OPENAI_API_KEY"])
        resp = client.embeddings.create(input="Hello world", model="text-embedding-ada-002")
        st.sidebar.success("✅ Azure OpenAI connection successful!")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {e}")

# --- 搜索配置 ---
st.sidebar.header("Search Configuration")
top_k = st.sidebar.slider("Number of documents to return", 1, 10, 3)

# --- 新建/清除聊天 ---
if st.sidebar.button("New Chat", use_container_width=True):
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = len(st.session_state["conversations"]) - 1

if st.sidebar.button("Clear All History", use_container_width=True):
    st.session_state["conversations"].clear()
    st.session_state["conversation_titles"].clear()
    st.session_state["active_chat_index"] = None
    st.rerun()

# --- 历史记录 ---
st.sidebar.subheader("History")
if not st.session_state["conversations"]:
    st.sidebar.info("No history yet. Click 'New Chat' to start.")
else:
    for i, title in enumerate(st.session_state["conversation_titles"]):
        display_title = title if len(title) <= 20 else title[:20]+"..."
        if i == st.session_state["active_chat_index"]:
            st.sidebar.button(f"📍 {display_title}", key=f"chat_active_{i}", disabled=True, use_container_width=True)
        else:
            if st.sidebar.button(f"💬 {display_title}", key=f"chat_{i}", use_container_width=True):
                st.session_state["active_chat_index"] = i

# ========== 主区域 ==========
st.title("Semantic Search AI Chat for BA Users")
st.caption("A Semantic Search App prototype integrated with Pinecone semantic retrieval.")

if not st.session_state["conversations"]:
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = 0
    st.rerun()

chat_index = st.session_state["active_chat_index"]
current_chat = st.session_state["conversations"][chat_index]

# --- 聊天历史展示 ---
for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 用户输入 ---
user_query = st.text_input(
    label="Enter your question:",
    placeholder="e.g., Tell me about science fiction movies from 1990s",
)

if st.session_state["active_chat_index"] is None:
    st.info("Click *'New Chat'* in the sidebar to start a conversation.")
    st.stop()

if user_query:
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please input your HKUST OpenAI API key in the sidebar first.")
        st.stop()

    st.chat_message("user").write(user_query)
    current_chat.append({"role": "user", "content": user_query})
    if len(current_chat) == 1:
        st.session_state["conversation_titles"][chat_index] = user_query[:40]

    with st.spinner("Performing semantic search in Pinecone..."):
        try:
            # 初始化 Client
            openai_client = get_azure_client(st.session_state["OPENAI_API_KEY"])
            pinecone_index = get_pinecone_client()

            # 生成查询 embedding
            response = openai_client.embeddings.create(
                input=user_query,
                model="text-embedding-ada-002"
            )
            query_vector = response.data[0].embedding
            vector_dim = len(query_vector)

            # 🔹 调用 Pinecone 查询
            results = pinecone_index.query(
                namespace=PINECONE_NAMESPACE,
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )

            # 🔹 构造结果显示内容
            answer_lines = [f"🔍 **Top {top_k} semantic matches (cosine similarity):**\n"]
            if results.matches:
                for i, m in enumerate(results.matches, start=1):
                    text = m.metadata.get("text", "")[:200]
                    score = m.score
                    answer_lines.append(f"{i}. *(score={score:.3f})* → {text}...")
            else:
                answer_lines.append("No results found in Pinecone index.")

            answer_text = "\n".join(answer_lines)
            confidence = round(random.uniform(0.80, 0.98), 2)

        except Exception as e:
            answer_text = f"❌ Error during semantic search:\n{str(e)}"
            confidence = 0.0

    st.chat_message("assistant").markdown(answer_text)
    current_chat.append({"role": "assistant", "content": answer_text})

# ========== 展开框：Embedding & 机制说明 ==========
with st.expander("🔍 Embedding Information"):
    st.markdown("""
    **How Semantic Search Works**
    1. Convert user query into an embedding vector.
    2. Pinecone efficiently searches nearest neighbor vectors using cosine similarity.
    3. The most semantically similar documents are retrieved and ranked.
    """)
    if 'query_vector' in locals():
        st.metric("Embedding Dimension", len(query_vector))
        st.code(query_vector[:10])
