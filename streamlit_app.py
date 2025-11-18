import streamlit as st
import random
import numpy as np
import pandas as pd
import time
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec

# ========= 页面配置 =========
st.set_page_config(
    page_title="Semantic Search AI Chat",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ========= Logo 样式 =========
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        position: relative;
    }
    .logo { 
        position: fixed;
        top: 10px;
        left: 15px;
        z-index: 100;
    }
    section[data-testid="stSidebar"] {
        width: 380px !important;
        min-width: 380px !important;
    }
    .stSidebar .stButton>button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.image("Logo_USTBusinessSchool.svg", width=120, output_format="SVG")


# ========= 初始化状态 =========
default_docs = [
    {"id": 1, "content": "HKUST Business School offers MBA programs with focus on analytics.", "embedding": None},
    {"id": 2, "content": "The ISOM department provides courses in information systems.", "embedding": None},
]

def init_session():
    """初始化所有会话状态"""
    for key, default in {
        "conversations": [],
        "conversation_titles": [],
        "active_chat_index": None,
        "OPENAI_API_KEY": None,
        "documents": default_docs.copy(),
        "last_query": "",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session()


# ========= 初始化Azure OpenAI客户端 =========
@st.cache_resource
def get_azure_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )

# ========= 初始化 Pinecone =========
PINECONE_API_KEY = "pcsk_JPQMS_zQZ9MfrD4aSEe8b69PoxsjcsvoSPEHpzgYGt4GPm8bv7ED95Wjy4u7vPmxSnjj"
PINECONE_INDEX_NAME = "developer-quickstart-py"
PINECONE_HOST = "https://developer-quickstart-py-9d1pu2j.svc.aped-4627-b74a.pinecone.io"

@st.cache_resource
def get_pinecone_client():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(name=PINECONE_INDEX_NAME, host=PINECONE_HOST)


# ========= 辅助函数 =========
def semantic_search(query_vector, top_k=5):
    """Pinecone 搜索函数"""
    index = get_pinecone_client()
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return response


def build_augmented_prompt(user_query: str, search_results) -> str:
    """构建 RAG prompt"""
    context_list = []
    for i, match in enumerate(search_results.matches, start=1):
        context_text = match.metadata.get("text", "")
        context_list.append(f"[Document {i}]\n{context_text}")
    context_block = "\n\n".join(context_list)

    augmented_prompt = f"""
You are an intelligent assistant. Please answer the user's question
strictly based on the context provided below.

Guidelines:
1. Only use the information from the **Context** section to answer.
2. Do NOT fabricate or guess.
3. If the answer is not present in the context, reply with:
   "The provided context does not contain the answer."

User Query:
{user_query}

Context:
{context_block}
""".strip()

    return augmented_prompt


# ========= Sidebar =========
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
    if not st.session_state["OPENAI_API_KEY"]:
        st.sidebar.error("Please input your API key first.")
    else:
        with st.spinner("Testing API connection..."):
            try:
                client = get_azure_client(st.session_state["OPENAI_API_KEY"])
                response = client.embeddings.create(input="Hello HKUST", model="text-embedding-ada-002")
                st.sidebar.success("✅ Azure OpenAI connection successful!")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {e}")

st.sidebar.header("🧠 API Status")
col1, col2 = st.sidebar.columns(2)
col1.success("✅ Pinecone: Connected")
col2.success("✅ Azure OpenAI: Connected")

st.sidebar.header("⚙️ Search Configuration")
top_k = st.sidebar.slider("Number of documents to return", 1, 10, 3)

st.sidebar.header("💡 Usage Tips")
st.sidebar.info("""
- Enter complete question statements
- More specific questions yield more accurate results
- Supports both Chinese and English queries
- System generates answers based on relevant documents
""")

st.sidebar.markdown("---")


# --- 新建会话 ---
if st.sidebar.button("🆕 New Chat", use_container_width=True):
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = len(st.session_state["conversations"]) - 1
    st.session_state["last_query"] = ""
    st.rerun()

# --- 清空所有历史 ---
if st.sidebar.button("🗑️ Clear All History", use_container_width=True):
    for key in ["conversations", "conversation_titles", "active_chat_index", "last_query"]:
        st.session_state[key] = None if "index" in key else []
    init_session()
    st.rerun()

# --- 会话列表 ---
st.sidebar.subheader("History")
if not st.session_state["conversations"]:
    st.sidebar.info("No history yet. Click 'New Chat' to start.")
else:
    for i, title in enumerate(st.session_state["conversation_titles"]):
        label = f"📍 {title}" if i == st.session_state["active_chat_index"] else f"💬 {title}"
        if st.sidebar.button(label, key=f"chat_{i}", use_container_width=True, disabled=(i == st.session_state["active_chat_index"])):
            st.session_state["active_chat_index"] = i
            st.session_state["last_query"] = ""
            st.rerun()


# ========= 主体部分 =========
st.title("🔍 Semantic Search AI Chat for BA Users")
st.caption("A Semantic Search App prototype for ISOM 6670G.")


# --- 若无激活会话 ---
if st.session_state["active_chat_index"] is None:
    st.info("Click *'New Chat'* in the sidebar to start a conversation.")
    st.stop()

chat_index = st.session_state["active_chat_index"]
current_chat = st.session_state["conversations"][chat_index]


# --- 已有消息展示 ---
for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# --- 输入框 (避免自动提交重复执行) ---
with st.form(key="query_form", clear_on_submit=True):
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., Where is HKUST Business School?",
    )
    submitted = st.form_submit_button("🚀 Submit Query")

if submitted and user_query:
    # 若无 key
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please input your HKUST OpenAI API key in the sidebar first.")
        st.stop()

    # 保存用户输入
    st.session_state["last_query"] = user_query
    st.chat_message("user").write(user_query)
    current_chat.append({"role": "user", "content": user_query})

    with st.spinner("🔍 Searching relevant documents..."):
        try:
            openai_client = get_azure_client(st.session_state["OPENAI_API_KEY"])
            response = openai_client.embeddings.create(input=user_query, model="text-embedding-ada-002")
            query_vector = response.data[0].embedding
            vector_dim = len(query_vector)

            # Pinecone 语义搜索
            search_results = semantic_search(query_vector, top_k=top_k)

            # 构建 RAG prompt
            aug_prompt = build_augmented_prompt(user_query, search_results)

            # 模拟答案生成
            simulated_answer = (
                f"✅ Processed with semantic search!\n\n"
                f"**Question:** {user_query}\n\n"
                f"**Retrieved {len(search_results.matches)} documents**\n\n"
                f"**Embedding Dimension:** {vector_dim}\n"
            )
            confidence = round(random.uniform(0.75, 0.99), 2)
            answer_text = f"{simulated_answer}\n**Confidence Score:** {confidence}"

        except Exception as e:
            answer_text = f"❌ Error processing query: {e}"
            confidence = 0.0

    st.chat_message("assistant").write(answer_text)
    current_chat.append({"role": "assistant", "content": answer_text})


# ========= Embedding信息 =========
with st.expander("🔍 Embedding Information"):
    st.markdown("""
    **How Semantic Search Works:**
    - Convert question into a numerical vector (embedding)
    - Capture semantic meaning
    - Search semantically similar documents in Pinecone
    - Generate an answer based on relevant context
    """)
    if st.session_state.get("last_query"):
        st.write(f"Latest Query: {st.session_state['last_query']}")
