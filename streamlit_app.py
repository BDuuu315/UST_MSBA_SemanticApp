import streamlit as st
import numpy as np
from openai import AzureOpenAI
from pinecone import Pinecone
from datetime import datetime

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

# -------------------- Azure --------------------
@st.cache_resource
def get_azure_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )

# -------------------- Pinecone --------------------
PINECONE_API_KEY = "pcsk_JPQMS_zQZ9MfrD4aSEe8b69PoxsjcsvoSPEHpzgYGt4GPm8bv7ED95Wjy4u7vPmxSnjj"
PINECONE_INDEX_NAME = "msba-lab-1537"
PINECONE_NAMESPACE = "default"

@st.cache_resource
def get_pinecone_client():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    return index

# -------------------- 相似度计算 --------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# -------------------- 语义检索 --------------------
def semantic_search(user_query: str, openai_client, top_k: int = 10):
    index = get_pinecone_client()

    emb_resp = openai_client.embeddings.create(
        input=user_query,
        model="text-embedding-ada-002"
    )
    query_vector = np.array(emb_resp.data[0].embedding)

    search_resp = index.query(
        namespace=PINECONE_NAMESPACE,
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )

    # 过滤相似度高于阈值 0.75 的结果
    filtered_matches = [m for m in search_resp.matches if m.score >= 0.75]

    return query_vector, filtered_matches

# -------------------- 构建增强提示 (RAG prompt) --------------------
def build_augmented_prompt(user_query: str, search_results) -> str:
    context_chunks = []
    for i, match in enumerate(search_results, 1):
        text = (
            match.metadata.get("text")
            or match.metadata.get("chunk_text")
            or match.metadata.get("content")
            or ""
        )
        context_chunks.append(f"[Document {i}]\n{text.strip()}")
    context_block = "\n\n".join(context_chunks)

    augmented_prompt = f"""
You are an intelligent assistant. Please answer the user's question strictly based on the context provided below.

Guidelines:
1. Only use the information from the **Context** section.
2. Do NOT fabricate or guess.
3. If the answer is not present in the context, reply with:
   "The provided context does not contain the answer."

User Question:
{user_query}

Context:
{context_block}
""".strip()

    return augmented_prompt

# -------------------- 用 Azure 生成答案 --------------------
def rag_answer_with_azure(prompt: str, client, model="gpt", temperature=0.2, max_tokens=1536):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Azure RAG generation failed: {e}")
        return "An error occurred while generating the response."

# -------------------- Sidebar --------------------
st.sidebar.title("💬 Chat History")

api_key = st.sidebar.text_input("Enter your HKUST API key", type="password")
if api_key:
    st.session_state.openai_api_key = api_key

st.sidebar.markdown("---")

if st.sidebar.button("🧹 Clear All History", use_container_width=True):
    st.session_state.conversations.clear()
    st.session_state.conversation_titles.clear()
    st.session_state.active_chat_index = None
    st.session_state.page = "home"
    st.rerun()

if len(st.session_state.conversations) == 0:
    st.sidebar.info("No saved results yet.")
else:
    for i, title in enumerate(st.session_state.conversation_titles):
        if st.sidebar.button(f"💬 {title}", key=f"hist_{i}", use_container_width=True):
            st.session_state.active_chat_index = i
            st.session_state.current_result = st.session_state.conversations[i]
            st.session_state.page = "result"
            st.rerun()

# ===============================================================
# 页面一：首页
# ===============================================================
if st.session_state.page == "home":
    st.markdown("<h1>🔍 Intelligent Semantic Search Application (Enhanced)</h1>", unsafe_allow_html=True)
    st.caption("Using Pinecone + Azure OpenAI (RAG integrated)")

    user_query = st.text_area(
        "📝 Enter your question",
        placeholder="e.g., What is HKUST? What is machine learning?",
        height=120,
    )

    col1, col2 = st.columns([1, 0.5])
    with col1:
        start_btn = st.button("🚀 Start Search", use_container_width=True)
    with col2:
        test_btn = st.button("🔄 Test Connection", use_container_width=True)

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
            st.error("Please enter your API key first.")
            st.stop()

        with st.spinner("Generating embeddings & performing semantic search..."):
            client = get_azure_client(st.session_state.openai_api_key)
            q_vec, matches = semantic_search(user_query, client, top_k=10)

        if len(matches) == 0:
            st.error("No documents found with cosine similarity > 0.75 😔")
            st.stop()

        with st.spinner("Building augmented prompt & generating intelligent answer..."):
            aug_prompt = build_augmented_prompt(user_query, matches)
            answer = rag_answer_with_azure(aug_prompt, client)

        st.session_state.current_result = {
            "query": user_query,
            "answer": answer,
            "vector_dim": len(q_vec),
            "vector_sample": q_vec[:10],
            "results": matches
        }
        st.session_state.page = "result"
        st.rerun()

# ===============================================================
# 页面二：结果展示
# ===============================================================
if st.session_state.page == "result":
    r = st.session_state.current_result
    st.markdown("### 💡 Intelligent Answer (RAG Based)")
    st.info(r["answer"])

    st.markdown("---")
    st.markdown(f"### 📄 Relevant Documents (Top {len(r['results'])} / score ≥ 0.75)")
    for i, m in enumerate(r["results"], 1):
        preview = (m.metadata.get("text") or m.metadata.get("chunk_text") or m.metadata.get("content") or "")[:150]
        st.markdown(f"**{i}.** (score: {m.score:.3f}) — {preview}...")

    st.markdown("---")
    st.markdown("### 📊 Search Statistics")
    st.metric("Embedding Dimension", r["vector_dim"])
    st.write("First 10 embedding values:")
    st.code(str(r["vector_sample"]))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save History", use_container_width=True):
            title = r["query"][:40]
            if title not in st.session_state.conversation_titles:
                st.session_state.conversation_titles.append(title)
                st.session_state.conversations.append(r)
            st.success("✅ Saved to sidebar history.")

    with col2:
        if st.button("🔁 Search Again", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
