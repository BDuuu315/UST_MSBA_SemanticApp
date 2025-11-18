import streamlit as st
import random
import numpy as np
from openai import AzureOpenAI
from pinecone import Pinecone

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="Intelligent Semantic Search", layout="wide")

# -------------------- 样式 --------------------
st.markdown("""
<style>
body, [data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #F5F5F5;
}
h1, h2, h3, h4, h5 { color: #FFFFFF; }
.stTextInput>div>div>input,
textarea {
    background-color: #1E222A !important;
    color: white !important;
}
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
}
.stButton>button[kind=primary] {
    background-color: #E74C3C;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.image("Logo_USTBusinessSchool.svg", width=120)

# -------------------- 初始化状态 --------------------
def init_session():
    defaults = {
        "page": "home",
        "conversations": [],
        "conversation_titles": [],
        "active_chat_index": None,
        "openai_api_key": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# -------------------- 初始化 Azure --------------------
@st.cache_resource
def get_azure_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )

# -------------------- 初始化 Pinecone --------------------
@st.cache_resource
def get_pinecone_client():
    pc = Pinecone(api_key="pcsk_JPQMS_zQZ9MfrD4aSEe8b69PoxsjcsvoSPEHpzgYGt4GPm8bv7ED95Wjy4u7vPmxSnjj")
    index = pc.Index(
        name="developer-quickstart-py",
        host="https://developer-quickstart-py-9d1pu2j.svc.aped-4627-b74a.pinecone.io"
    )
    return index

# -------------------- 语义搜索函数 --------------------
def semantic_search(vector, top_k=5):
    index = get_pinecone_client()
    return index.query(vector=vector, top_k=top_k, include_metadata=True)

# -------------------- Sidebar --------------------
st.sidebar.title("Chat History")

api_key = st.sidebar.text_input("Enter your HKUST API key", type="password")
if api_key:
    st.session_state.openai_api_key = api_key

st.sidebar.markdown("---")

if st.sidebar.button("Clear All History", use_container_width=True):
    st.session_state.conversations.clear()
    st.session_state.conversation_titles.clear()
    st.session_state.active_chat_index = None
    st.session_state.page = "home"
    st.rerun()

if len(st.session_state.conversations) == 0:
    st.sidebar.info("No saved results yet.")
else:
    for i, title in enumerate(st.session_state.conversation_titles):
        if st.sidebar.button(f"{title}", key=f"hist_{i}", use_container_width=True):
            st.session_state.active_chat_index = i
            st.session_state.current_result = st.session_state.conversations[i]
            st.session_state.page = "result"
            st.rerun()

# ===============================================================
# 页面一：首页
# ===============================================================
if st.session_state.page == "home":
    st.markdown("<h1>🔍 Intelligent Semantic Search Application</h1>", unsafe_allow_html=True)
    st.caption("Using Pinecone + Azure OpenAI for semantic search")

    st.markdown("<h3>Enter Your Question</h3>", unsafe_allow_html=True)
    user_query = st.text_area(
        "e.g., Where is HKUST Business School? What is machine learning?",
        placeholder="What is your question?",
        height=120,
    )

    col1, col2 = st.columns([1, 0.5])
    with col1:
        start_btn = st.button("Start Search", use_container_width=True)
    with col2:
        test_btn = st.button("Test Connection", use_container_width=True)

    if test_btn:
        if not st.session_state.openai_api_key:
            st.error("Please input your Azure API key first.")
        else:
            with st.spinner("Testing Azure OpenAI..."):
                try:
                    client = get_azure_client(st.session_state.openai_api_key)
                    client.embeddings.create(input="Hello world", model="text-embedding-ada-002")
                    st.success("✅ Connection successful.")
                except Exception as e:
                    st.error(f"❌ Failed: {e}")

    if start_btn:
        if not user_query.strip():
            st.warning("Please enter your question.")
            st.stop()
        if not st.session_state.openai_api_key:
            st.error("Please enter your HKUST API key in sidebar first.")
            st.stop()

        # 生成 embedding
        with st.spinner("Generating embeddings..."):
            client = get_azure_client(st.session_state.openai_api_key)
            emb = client.embeddings.create(input=user_query, model="text-embedding-ada-002")
            query_vector = emb.data[0].embedding
            dim = len(query_vector)

        # Pinecone 语义搜索
        with st.spinner("Running semantic search..."):
            try:
                results = semantic_search(query_vector, top_k=5)
            except Exception as e:
                st.error(f"Error querying Pinecone: {e}")
                st.stop()

        # 模拟答案
        answer = "This is an intelligent answer generated by semantic search based on relevant documents."
        st.session_state.current_result = {
            "query": user_query,
            "answer": answer,
            "vector_dim": dim,
            "vector_sample": query_vector[:10],
            "results": results.matches
        }
        st.session_state.page = "result"
        st.rerun()

# ===============================================================
# 页面二：结果展示
# ===============================================================
if st.session_state.page == "result":
    r = st.session_state.current_result

    st.markdown("### 💡 Intelligent Answer Based on Semantic Search")
    st.info(r["answer"])

    st.markdown("---")
    st.markdown(f"### 📄 Relevant Documents ({len(r['results'])} found)")
    if len(r["results"]) == 0:
        st.write("No relevant documents found.")
    else:
        for i, m in enumerate(r["results"], 1):
            preview = m.metadata.get("text", "")[:150]
            st.markdown(f"**{i}.** (score: {m.score:.3f}) — {preview}")

    st.markdown("---")
    st.markdown("### Search Statistics")

    st.write("**Search Information:**")
    st.write(f"- 📝 Query Content: {current_search['query']}")
    st.write(f"- 📊 Documents Returned: {len(current_search['results'])}")
    st.write(f"- 🎯 Highest Similarity: {df['score'].max():.3f}")
    st.write(f"- 📍 Average Similarity: {df['score'].mean():.3f}")
    st.write(f"- ⏰ Search Time: {current_search['timestamp']}")
    st.markdown("""
    **How Semantic Search Works:**
    - Convert question into a numerical vector (embedding)
    - Capture semantic meaning
    - Calculate similarity between question and document embeddings
    - Most relevant documents are returned based on semantic similarity
    """)
    st.metric("Embedding Dimension", r["vector_dim"])
    st.write("First 10 embedding values:")
    st.code(str(r["vector_sample"]))

    st.markdown("---")

    # ====== Save History 和 Search Again ======
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save History", use_container_width=True):
            title = r["query"][:40]
            if title not in st.session_state.conversation_titles:
                st.session_state.conversation_titles.append(title)
                st.session_state.conversations.append(r)
            st.success("Result saved to sidebar history.")

    with col2:
        if st.button("Search Again", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
