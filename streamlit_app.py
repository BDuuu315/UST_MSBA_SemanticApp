import streamlit as st
import numpy as np
from openai import AzureOpenAI
from pinecone import Pinecone
from datetime import datetime

# ==============================================================
# 🧭 PAGE CONFIG
# ==============================================================
st.set_page_config(page_title="Intelligent Semantic Search", layout="wide")

# ==============================================================
# 🎨 STYLES
# ==============================================================
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

# ==============================================================
# 💾 SESSION INIT
# ==============================================================
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

# ==============================================================
# 🤖 AZURE OPENAI CLIENT
# ==============================================================
@st.cache_resource
def get_azure_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )

# ==============================================================
# 🧱 PINECONE CLIENT + SEMANTIC SEARCH (替换后的版本)
# ==============================================================
@st.cache_resource
def get_pinecone_client():
    pc = Pinecone(api_key="pcsk_JPQMS_zQZ9MfrD4aSEe8b69PoxsjcsvoSPEHpzgYGt4GPm8bv7ED95Wjy4u7vPmxSnjj")
    index = pc.Index(
        name="developer-quickstart-py",
        host="https://developer-quickstart-py-9d1pu2j.svc.aped-4627-b74a.pinecone.io"
    )
    return index


def semantic_search(user_query: str, openai_client, top_k=5):
    """
    使用 Azure 生成嵌入向量，并在 Pinecone 上实现语义检索
    """
    # === 生成 query 向量 ===
    emb = openai_client.embeddings.create(
        input=user_query,
        model="text-embedding-ada-002"
    )
    query_vector = emb.data[0].embedding

    # === 检索 Pinecone ===
    index = get_pinecone_client()
    search_resp = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # === 打印结果调试 ===
    st.write(f"**🔎 Query:** {user_query}")
    st.write(f"**📊 Top-{top_k} results retrieved from Pinecone**")
    for i, m in enumerate(search_resp.matches, 1):
        text_preview = m.metadata.get("text", "[no text]")[:120]
        st.write(f"{i}. ({m.score:.3f}) {text_preview}")

    return query_vector, search_resp

# ==============================================================
# 🧩 BUILD RAG PROMPT
# ==============================================================
def build_augmented_prompt(user_query: str, search_results) -> str:
    context_chunks = []
    for i, match in enumerate(search_results.matches, 1):
        doc_text = (
            match.metadata.get("text")
            or match.metadata.get("chunk_text", "")
        )
        context_chunks.append(f"[Document {i}]\n{doc_text.strip()}")

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

# ==============================================================
# 🧠 RAG ANSWER VIA AZURE OPENAI
# ==============================================================
def generate_rag_answer(user_query, openai_client, search_results):
    prompt = build_augmented_prompt(user_query, search_results)
    response = openai_client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=768
    )
    return response.choices[0].message.content.strip()

# ==============================================================
# 📜 SIDEBAR
# ==============================================================
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

# ==============================================================
# 🏠 PAGE 1: HOME
# ==============================================================
if st.session_state.page == "home":
    st.markdown("<h1>🔍 Intelligent Semantic Search Application</h1>", unsafe_allow_html=True)
    st.caption("Using Pinecone + Azure OpenAI for semantic search")

    st.markdown("<h3>📝 Enter Your Question</h3>", unsafe_allow_html=True)
    user_query = st.text_area(
        "For example:\n• What is HKUST?\n• What is machine learning?",
        placeholder="Type your natural language question here...",
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
            st.error("Please enter your API key in sidebar first.")
            st.stop()

        with st.spinner("Connecting to Azure & Pinecone..."):
            client = get_azure_client(st.session_state.openai_api_key)
            query_vec, results = semantic_search(user_query, client, top_k=5)

        with st.spinner("Generating intelligent answer..."):
            try:
                answer = generate_rag_answer(user_query, client, results)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.stop()

        st.session_state.current_result = {
            "query": user_query,
            "answer": answer,
            "vector_dim": len(query_vec),
            "vector_sample": query_vec[:10],
            "results": results.matches
        }
        st.session_state.page = "result"
        st.rerun()

# ==============================================================
# 📄 PAGE 2: RESULTS
# ==============================================================
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
    st.markdown("### 📊 Search Statistics")
    st.metric("Embedding Dimension", r["vector_dim"])
    st.write("First 10 embedding values:")
    st.code(str(r["vector_sample"]))

    st.markdown("---")
    if st.button("💾 Save to History", use_container_width=True):
        title = r["query"][:40]
        st.session_state.conversation_titles.append(title)
        st.session_state.conversations.append(r)
        st.success("✅ Result saved to sidebar history.")
