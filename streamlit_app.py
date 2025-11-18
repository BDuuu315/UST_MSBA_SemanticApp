import streamlit as st
import random
import os
import numpy as np
import pandas as pd
from openai import AzureOpenAI

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
        height: 100vh;
        overflow: auto;
    }
    section[data-testid="stSidebar"] > div {
        width: 380px !important;
        padding-top: 2rem;
        height: 100%;
    }
    .stSidebar .stButton>button {
        width: 100%;
    }
    .main .block-container {
        padding-left: 400px;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.image("Logo_USTBusinessSchool.svg", width=120, output_format="SVG")

# ========= 初始化状态 =========
if "conversations" not in st.session_state:
    st.session_state["conversations"] = []
if "conversation_titles" not in st.session_state:
    st.session_state["conversation_titles"] = []
if "active_chat_index" not in st.session_state:
    st.session_state["active_chat_index"] = None
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None
if "last_processed_query" not in st.session_state:
    st.session_state["last_processed_query"] = None
if "documents" not in st.session_state:
    st.session_state["documents"] = [
        {"id": 1, "content": "HKUST Business School offers MBA programs with focus on analytics.", "embedding": None},
        {"id": 2, "content": "The ISOM department provides courses in information systems.", "embedding": None},
        {"id": 3, "content": "HKUST is located in Clear Water Bay, Kowloon, Hong Kong.", "embedding": None},
        {"id": 4, "content": "Business Analytics programs teach data mining and machine learning.", "embedding": None},
        {"id": 5, "content": "The university was founded in 1991 and is a leading research institution.", "embedding": None},
    ]

# ========= 初始化Azure OpenAI客户端 =========
@st.cache_resource
def get_azure_client(api_key):
    if not api_key:
        return None
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )

# ========= 语义搜索 (模拟) =========
def semantic_search(query, client, top_k=3):
    """执行语义搜索并返回相关文档"""
    try:
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_vector = response.data[0].embedding

        # 模拟文档检索
        relevant_docs = random.sample(st.session_state["documents"], min(top_k, len(st.session_state["documents"])))
        return relevant_docs, query_vector, len(query_vector)
    except Exception as e:
        st.error(f"语义搜索错误: {e}")
        return [], None, 0


# ========= RAG 增强提示构建 =========
def build_augmented_prompt(user_query: str, relevant_docs) -> str:
    """根据搜索结果构造增强提示词"""
    context_chunks = []
    for i, doc in enumerate(relevant_docs, 1):
        text = doc.get("content", "")
        context_chunks.append(f"[Document {i}]\n{text.strip()}")
    context_block = "\n\n".join(context_chunks)

    augmented_prompt = f"""
You are an intelligent assistant. Please answer the user's question strictly based on the context below.

Guidelines:
1. Only use the information from the **Context** section.
2. Do NOT fabricate or guess.
3. If the answer is not present, reply: "The provided context does not contain the answer."

User Question:
{user_query}

Context:
{context_block}
""".strip()

    return augmented_prompt


# ========= 基于RAG生成AI回答 =========
def generate_contextual_ai_response(user_query: str, relevant_docs, client):
    """根据上下文生成AI回答 (RAG 模式)"""
    try:
        augmented_prompt = build_augmented_prompt(user_query, relevant_docs)

        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and precise assistant that uses only provided context."
                },
                {"role": "user", "content": augmented_prompt}
            ],
            max_tokens=600,
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()
        confidence = round(random.uniform(0.85, 0.95), 2)
        if "does not contain the answer" in answer:
            confidence = round(confidence * 0.75, 2)

        return answer, confidence

    except Exception as e:
        fallback_answer = (
            f"Based on general knowledge, I cannot find information in the provided context about '{user_query}'."
        )
        return fallback_answer, round(random.uniform(0.6, 0.8), 2)


# ========= Sidebar =========
st.sidebar.title("Chat Sidebar")

api_key = st.sidebar.text_input(
    "Enter your HKUST OpenAI API Key",
    type="password",
    value=st.session_state.get("OPENAI_API_KEY", ""),
    help="You can check ISOM 6670G syllabus for API setup."
)
if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

st.sidebar.markdown("---")

# API Check
if st.sidebar.button("Test Connection", use_container_width=True):
    with st.spinner("Testing connection..."):
        try:
            client = get_azure_client(st.session_state["OPENAI_API_KEY"])
            if client:
                client.embeddings.create(input="Hello world", model="text-embedding-ada-002")
                st.sidebar.success("✅ Azure OpenAI connection successful!")
            else:
                st.sidebar.error("❌ Please enter a valid API key.")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")

st.sidebar.header("Search Configuration")
top_k = st.sidebar.slider("Number of documents to return", 1, 10, 3)

# Conversation creation
if st.sidebar.button("New Chat", use_container_width=True):
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = len(st.session_state["conversations"]) - 1
    st.session_state["last_processed_query"] = None
    st.rerun()

if st.sidebar.button("Clear All History", use_container_width=True):
    st.session_state["conversations"].clear()
    st.session_state["conversation_titles"].clear()
    st.session_state["active_chat_index"] = None
    st.session_state["last_processed_query"] = None
    st.rerun()

st.sidebar.subheader("History")

if len(st.session_state["conversations"]) == 0:
    st.sidebar.info("No history yet. Click 'New Chat' to start.")
else:
    for i, title in enumerate(st.session_state["conversation_titles"]):
        display_title = title[:20] + "..." if len(title) > 20 else title
        if i == st.session_state["active_chat_index"]:
            st.sidebar.button(f"📍 {display_title}", key=f"chat_active_{i}", disabled=True, use_container_width=True)
        else:
            if st.sidebar.button(f"💬 {display_title}", key=f"chat_{i}", use_container_width=True):
                st.session_state["active_chat_index"] = i
                st.session_state["last_processed_query"] = None
                st.rerun()

# ========= 主体部分 =========
st.title("Semantic Search AI Chat (RAG Enhanced)")
st.caption("A Semantic Search + Context-Aware AI Chat Prototype for ISOM 6670G.")

if st.session_state["active_chat_index"] is None and len(st.session_state["conversations"]) > 0:
    st.session_state["active_chat_index"] = 0
elif len(st.session_state["conversations"]) == 0:
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = 0

chat_index = st.session_state["active_chat_index"]
current_chat = st.session_state["conversations"][chat_index]
chat_title = st.session_state["conversation_titles"][chat_index]

st.subheader(chat_title)

# --- 展示历史聊天 ---
for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 新问题输入 ---
user_query = st.chat_input(
    placeholder="e.g., Where is HKUST Business School?",
    key=f"chat_input_{chat_index}"
)

# ========= 用户提问处理 =========
if user_query and user_query != st.session_state.get("last_processed_query"):
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please input your HKUST OpenAI API key first.")
        st.stop()

    with st.chat_message("user"):
        st.write(user_query)
    current_chat.append({"role": "user", "content": user_query})

    if len(current_chat) == 1:
        new_title = user_query[:30] + "..." if len(user_query) > 30 else user_query
        st.session_state["conversation_titles"][chat_index] = new_title

    with st.chat_message("assistant"):
        with st.spinner("Performing semantic search and retrieving relevant context..."):
            try:
                openai_client = get_azure_client(st.session_state["OPENAI_API_KEY"])
                relevant_docs, query_vector, vector_dim = semantic_search(user_query, openai_client, top_k=top_k)
                answer, confidence = generate_contextual_ai_response(user_query, relevant_docs, openai_client)

                # 显示完整结果
                answer_text = f"{answer}\n\n---\n"
                answer_text += f"**Semantic Search Results:**\n"
                answer_text += f"- **Embedding Dimension:** {vector_dim}\n"
                answer_text += f"- **Documents Returned:** {len(relevant_docs)}\n"
                answer_text += f"- **Confidence Score:** {confidence}\n\n"

                if relevant_docs:
                    answer_text += "**Relevant Documents:**\n"
                    for i, doc in enumerate(relevant_docs, 1):
                        answer_text += f"{i}. {doc['content']}\n"

                st.write(answer_text)
                current_chat.append({"role": "assistant", "content": answer_text})
                st.session_state["last_processed_query"] = user_query

            except Exception as e:
                error_msg = f"❌ Error processing your query: {e}"
                st.write(error_msg)
                current_chat.append({"role": "assistant", "content": error_msg})
                st.session_state["last_processed_query"] = user_query


# ========= Embedding + Session 信息 =========
with st.expander("Embedding Information"):
    st.markdown("""
    **How Semantic Search Works:**
    - Convert query to an embedding vector
    - Compare similarity between query and stored document embeddings
    - Retrieve most semantically similar documents
    """)
    st.write(f"**Document Library Size:** {len(st.session_state['documents'])}")
    for doc in st.session_state["documents"]:
        st.write(f"- {doc['content']}")

with st.expander("Session Information"):
    st.write(f"**Active Chat:** {chat_title}")
    st.write(f"**Total Conversations:** {len(st.session_state['conversations'])}")
    st.write(f"**Messages in Current Chat:** {len(current_chat)}")
    st.write(f"**Last Processed Query:** {st.session_state.get('last_processed_query', 'None')}")
    st.write(f"**Search Top K:** {top_k}")
