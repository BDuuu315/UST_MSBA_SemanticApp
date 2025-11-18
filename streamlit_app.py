import streamlit as st
import random
from pinecone import Pinecone
from openai import AzureOpenAI

# ===== 基本配置 =====
PINECONE_API_KEY = "pcsk_6oHDXL_QyzEgtuEzHkTacffEiBW4gmGjPVfb4MAuz2Wy3M47yA5WR7XePPodEW1p6d6XyW"
PINECONE_INDEX_NAME = "geo-semantic"
PINECONE_ENV_HOST = "https://geo-semantic-u90uigv.svc.aped-4627-b74a.pinecone.io"
PINECONE_NAMESPACE = "__default__"

# ===== 页面样式与配置 =====
st.set_page_config(page_title="🎬 Movie Semantic Search", layout="centered", initial_sidebar_state="expanded")
st.title("🎬 Movie Semantic Search AI Chat")
st.caption("A semantic search demo integrated with Pinecone movie database.")
st.markdown("""
<style>
.result-card {
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
}
.result-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}
.result-id {
    font-weight: 600;
    background-color: #f4f4f4;
    padding: 6px 12px;
    border-radius: 6px;
    color: #333;
}
.result-score {
    font-size: 14px;
    color: #666;
}
.result-content span {
    display: block;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ===== 状态初始化 =====
for key, default in {
    "conversations": [], "conversation_titles": [], "active_chat_index": None, "OPENAI_API_KEY": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ===== 初始化 Azure OpenAI + Pinecone =====
@st.cache_resource
def get_azure_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )

@st.cache_resource
def get_pinecone_client():
    pc = Pinecone(api_key=PINECONE_API_KEY, host=PINECONE_ENV_HOST)
    return pc.Index(PINECONE_INDEX_NAME)

# ===== 侧边栏配置 =====
st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Enter your HKUST Azure OpenAI API Key", type="password")
if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

top_k = st.sidebar.slider("Number of results to return (Top K)", 1, 10, 5)

if st.sidebar.button("Test Connection"):
    with st.spinner("Testing Azure OpenAI connection..."):
        try:
            client = get_azure_client(st.session_state["OPENAI_API_KEY"])
            client.embeddings.create(input="Hello", model="text-embedding-ada-002")
            st.sidebar.success("✅ Azure OpenAI connection successful!")
        except Exception as e:
            st.sidebar.error(f"❌ Azure OpenAI connection failed: {str(e)}")

# ===== 聊天框 & Query 输入 =====
if len(st.session_state["conversations"]) == 0:
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = 0

chat_index = st.session_state["active_chat_index"]
current_chat = st.session_state["conversations"][chat_index]

for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.text_input("Ask about a movie:", placeholder="e.g., A movie about alien worlds and adventure")

if user_query:
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please input your HKUST OpenAI API key in the sidebar first.")
        st.stop()

    st.chat_message("user").write(user_query)
    current_chat.append({"role": "user", "content": user_query})

    with st.spinner("Embedding & querying Pinecone..."):
        try:
            # 1️⃣ 创建 Embedding
            openai_client = get_azure_client(st.session_state["OPENAI_API_KEY"])
            embedding_response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=user_query
            )
            query_vector = embedding_response.data[0].embedding

            # 2️⃣ 查询 Pinecone
            index = get_pinecone_client()
            
            # 🚀 强制将输入 query 和返回结果都用 UTF‑8 处理，避免 latin-1 报错
            try:
                # 在 query 前先检查是否为 bytes 或 str，并强制 utf-8 编码
                if isinstance(user_query, bytes):
                    user_query = user_query.decode("utf-8", errors="ignore")
                else:
                    user_query = user_query.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            
                query_results = index.query(
                    namespace=PINECONE_NAMESPACE,
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
            except UnicodeEncodeError as ue:
                st.error(f"Encoding error: {ue}. Trying fallback encoding...")
                query_results = index.query(
                    namespace=PINECONE_NAMESPACE,
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
            except Exception as e:
                raise e

            # 3️⃣ 展示结果
            if not query_results.matches:
                st.chat_message("assistant").warning("No matching movies found.")
                current_chat.append({"role": "assistant", "content": "No matching results found."})
            else:
                cards_html = ""
                for rank, match in enumerate(query_results.matches, start=1):
                    m = match.metadata
                    cards_html += f"""
                    <div class='result-card'>
                        <div class='result-header'>
                            <div class='result-id'>#{rank} — ID: {match.id}</div>
                            <div class='result-score'>Score: {match.score:.4f}</div>
                        </div>
                        <div class='result-content'>
                            <span><b>box-office:</b> {m.get("box-office", "N/A"):,}</span>
                            <span><b>genre:</b> {m.get("genre", "N/A")}</span>
                            <span><b>summary:</b> {m.get("summary", "N/A")[:250]}...</span>
                            <span><b>title:</b> {m.get("title", "N/A")}</span>
                            <span><b>year:</b> {m.get("year", "N/A")}</span>
                        </div>
                    </div>
                    """
                html_output = f"<div>{cards_html}</div>"
                st.chat_message("assistant").markdown("### 🔍 Top Matching Movies")
                st.markdown(html_output, unsafe_allow_html=True)
                current_chat.append({"role": "assistant", "content": "Top matching movie results displayed."})

        except Exception as e:
            err_msg = f"❌ Error querying Pinecone: {str(e)}"
            st.chat_message("assistant").error(err_msg)
            current_chat.append({"role": "assistant", "content": err_msg})

# ===== Embedding 信息展示 =====
with st.expander("🔍 Embedding Information"):
    st.markdown("""
    **How it works:**
    1. Convert query into the OpenAI embedding vector.
    2. Perform cosine similarity search in Pinecone vector index.
    3. Retrieve and display most similar movie records (with metadata).
    """)
    if 'query_vector' in locals():
        st.metric("Embedding Dimension", len(query_vector))
        st.code(query_vector[:10])
