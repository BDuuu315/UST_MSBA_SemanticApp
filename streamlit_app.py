import streamlit as st
import numpy as np
import random
from openai import AzureOpenAI
from pinecone import Pinecone
from datetime import datetime

# Configure Streamlit page layout //Jayson & Greta
st.set_page_config(page_title="RAG Semantic Search Chat", layout="wide")

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


# initialize session variables to store state between UI interactions //Erin
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

# Initializing Azure and Pinecone, The API key needs to be entered manually,API version from Marcela's File //Erin
@st.cache_resource
def get_azure_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15", 
        azure_endpoint="https://hkust.azure-api.net"
    )

# Pinecone API of Greta, self generated Database of movie describtions. //Frank
PINECONE_API_KEY = "pcsk_JPQMS_zQZ9MfrD4aSEe8b69PoxsjcsvoSPEHpzgYGt4GPm8bv7ED95Wjy4u7vPmxSnjj"
PINECONE_INDEX_NAME = "msba-lab-1537"
PINECONE_NAMESPACE = "default"

@st.cache_resource
def get_pinecone_client():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

# Semantic Search function, embedding //Frank & Erin
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

    # Match only when cos distance score >0.75
    filtered_matches = [m for m in search_resp.matches if m.score >= 0.75]
    return query_vector, filtered_matches


# RAG Prompt, in case can't find a correct answer from our Pinecone index, we allow our App to search direct from GPT //Alan & Erin
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
You are an intelligent assistant.

**Primary task:** Use the context below to answer as accurately as possible.
**Fallback:** If the context is not relevant or lacks enough information, use your general knowledge to answer the question.

User Question:
{user_query}

Context:
{context_block}
"""
    return augmented_prompt


# Generate Answer // Alan
def generate_contextual_ai_response(user_query: str, openai_client, top_k: int = 10):
    try:
        # Segmentation
        query_vec, matches = semantic_search(user_query, openai_client, top_k=top_k)
        if len(matches) == 0:
            return {
                "query": user_query,
                "answer": "The provided context does not contain relevant content.",
                "confidence": 0.6,
                "sources": [],
                "vector_dim": len(query_vec),
                "results": []
            }

        # Augment Prompt
        augmented_prompt = build_augmented_prompt(user_query, matches)

        # Azure AI for answer
        response = openai_client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant using provided context."},
                {"role": "user", "content": augmented_prompt}
            ],
            max_tokens=600,
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()
        confidence = round(random.uniform(0.85, 0.95), 2)
        if "does not contain the answer" in answer:
            confidence = round(confidence * 0.7, 2)

        return {
            "query": user_query,
            "answer": answer,
            #"confidence": confidence
            #obtaining the true confidence too difficult, we decided to optimize it in the next iteration.
            "sources": [m.metadata.get("source", f"Document {i+1}") for i, m in enumerate(matches)],
            "vector_dim": len(query_vec),
            "vector_sample": query_vec[:10],
            "results": matches
        }
    except Exception as e:
        return {
            "query": user_query,
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "results": []
        }


# Sidebar //Greta & Jayson
st.sidebar.title("History & API Settings")

api_key = st.sidebar.text_input("Enter your HKUST Azure OpenAI API Key", type="password")
if api_key:
    st.session_state.openai_api_key = api_key

st.sidebar.markdown("---")

if st.sidebar.button("Clear All History", use_container_width=True):
    st.session_state["conversations"].clear()
    st.session_state["conversation_titles"].clear()
    st.session_state["active_chat_index"] = None
    st.session_state.page = "home"
    st.rerun()

if len(st.session_state["conversations"]) == 0:
    st.sidebar.info("No saved conversation.")
else:
    for i, title in enumerate(st.session_state["conversation_titles"]):
        if st.sidebar.button(f"{title}", key=f"hist_{i}", use_container_width=True):
            st.session_state.active_chat_index = i
            st.session_state.page = "result"
            st.session_state.current_result = st.session_state.conversations[i]
            st.rerun()

# Main Area (Search Page) //Greta & Jayson
if st.session_state.page == "home":
    st.title("Semantic Search for movie ideas")
    st.caption("Using this App for seeking inspiration for a screenplay")

    user_query = st.text_area("Enter your question", placeholder="e.g., Give me a scenario with a legendary travel / A film of young kids and their imaginary friends")
    col1, col2 = st.columns([1, 0.5])

    with col1:
        start_btn = st.button("Start Search", use_container_width=True)
    with col2:
        test_btn = st.button("Test Connection", use_container_width=True)

    if test_btn:
        if not api_key:
            st.error("Please input your Azure API key first.")
        else:
            with st.spinner("Testing Azure connection..."):
                try:
                    client = get_azure_client(api_key)
                    client.embeddings.create(input="Hello world", model="text-embedding-ada-002")
                    st.success("✅ Connection successful!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")

    if start_btn:
        if not user_query.strip():
            st.warning("Please enter a valid question.")
            st.stop()

        if not api_key:
            st.error("Please input your Azure API key first.")
            st.stop()

        with st.spinner("Performing RAG search and generating answer..."):
            client = get_azure_client(api_key)
            result = generate_contextual_ai_response(user_query, client, top_k=10)

        st.session_state.current_result = result
        st.session_state.page = "result"
        st.rerun()

#Answering Page //Greta
if st.session_state.page == "result":
    result = st.session_state.get("current_result", {})

    st.markdown("RAG Based Answer")
    st.info(result.get("answer", "No answer."))

    st.markdown("---")
    st.markdown(f"Relevant Documents ({len(result.get('results', []))})")
    for i, m in enumerate(result.get("results", []), 1):
        preview = (m.metadata.get("text") or m.metadata.get("chunk_text") or m.metadata.get("content") or "")[:150]
        st.markdown(f"**{i}.** (score: {m.score:.3f}) — {preview}...")

    st.markdown("---")
    st.markdown("Embedding + Search Info")
    st.metric("Embedding Dimension", result.get("vector_dim", 0))
    st.code(str(result.get("vector_sample", [])))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save History", use_container_width=True):
            title = result["query"][:40]
            if title not in st.session_state.conversation_titles:
                st.session_state.conversation_titles.append(title)
                st.session_state.conversations.append(result)
            st.success("Saved to history.")

    with col2:
        if st.button("Return to Search", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
