
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
    </style>
    """,
    unsafe_allow_html=True
)
st.image("Logo_USTBusinessSchool.svg", width=120, output_format="SVG")

# ========= 初始化状态 =========
if "conversations" not in st.session_state:
    st.session_state["conversations"] = []  # 每个元素为 list[dict(role, content)]
if "conversation_titles" not in st.session_state:
    st.session_state["conversation_titles"] = []  # 保存会话标题
if "active_chat_index" not in st.session_state:
    st.session_state["active_chat_index"] = None
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None

# ========= 初始化Azure OpenAI客户端 =========
@st.cache_resource
def get_azure_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )


# ========= Sidebar =========
st.sidebar.title("Chat Sidebar")


# --- 输入 API Key ---
api_key = st.sidebar.text_input(
    "Enter your HKUST OpenAI API Key",
    type="password",
    help="You can check ISOM 6670G syllabus to get set-up instructions."
)
if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

st.sidebar.markdown("---")

# --- 新建会话按钮 ---
if st.sidebar.button("🆕 New Chat", use_container_width=True):
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = len(st.session_state["conversations"]) - 1

# --- 清除所有历史按钮  ---
if st.sidebar.button("🗑️ Clear All History", use_container_width=True):
    st.session_state["conversations"].clear()
    st.session_state["conversation_titles"].clear()
    st.session_state["active_chat_index"] = None
    st.rerun()

# --- 历史列表 ---
st.sidebar.subheader("History")

if len(st.session_state["conversations"]) == 0:
    st.sidebar.info("No history yet. Click 'New Chat' to start.")
else:
    for i, title in enumerate(st.session_state["conversation_titles"]):
        max_length = 20
        if len(title) > max_length:
            display_title = title[:max_length] + "..."
        else:
            display_title = title

        if i == st.session_state["active_chat_index"]:
            st.sidebar.button(f"📍 {display_title}", key=f"chat_active_{i}", disabled=True, use_container_width=True)
        else:
            if st.sidebar.button(f"💬 {display_title}", key=f"chat_{i}", use_container_width=True):
                st.session_state["active_chat_index"] = i

# ========= 主体部分 =========
st.title("Semantic Search AI Chat for BA Users")
st.caption("A Semantic Search App prototype for ISOM 6670G.")

# --- 输入新消息 ---
user_query = st.text_input(
    label="Enter your question:",
    placeholder="e.g., Where is HKUST Business School?",
    help="Type your natural language question here."
)

# ========= 初始化状态 =========
if "conversations" not in st.session_state:
    st.session_state["conversations"] = []
if "conversation_titles" not in st.session_state:
    st.session_state["conversation_titles"] = []
if "active_chat_index" not in st.session_state:
    st.session_state["active_chat_index"] = None
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None
if "documents" not in st.session_state:
    st.session_state["documents"] = [
        {"id": 1, "content": "HKUST Business School offers MBA programs with focus on analytics.", "embedding": None},
        {"id": 2, "content": "The ISOM department provides courses in information systems.", "embedding": None},
    ]
# --- 没有激活的聊天时提示 ---
if st.session_state["active_chat_index"] is None:
    st.info("Click *'New Chat'* in the sidebar to start a conversation.")
    st.stop()

# --- 已选定的会话 ---
chat_index = st.session_state["active_chat_index"]
current_chat = st.session_state["conversations"][chat_index]
chat_title = st.session_state["conversation_titles"][chat_index]

# --- 展示已有消息 ---
for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if user_query:
    # 若没有 API key，不允许继续
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please input your HKUST OpenAI API key in the sidebar first.")
        st.stop()

    # 1️⃣ 立即显示并保存用户输入
    st.chat_message("user").write(user_query)
    current_chat.append({"role": "user", "content": user_query})

    # 若这是该会话第一条消息，则用它更新标题
    if len(current_chat) == 1:
        st.session_state["conversation_titles"][chat_index] = user_query[:40]

    # 2️⃣ 生成embedding并获取AI回答
    with st.spinner("Processing your query with semantic search..."):
        try:
            # 初始化Azure OpenAI客户端
            openai_client = get_azure_client(st.session_state["OPENAI_API_KEY"])
            
            # 为查询生成embedding
            response = openai_client.embeddings.create(
                input=user_query,
                model="text-embedding-ada-002"
            )
            
            # 获取embedding向量
            query_vector = response.data[0].embedding
            vector_dim = len(query_vector)
            
            # 模拟语义搜索结果（这里可以替换为你的实际搜索逻辑）
            # 基于embedding进行相似度搜索等操作
            
            # 生成回答
            simulated_answer = (
                f"Question has been processed with semantic search!\n\n"
                f"**Question:** {user_query}\n\n"
                f"**Embedding Dimension:** {vector_dim}\n"
            )
            confidence = round(random.uniform(0.75, 0.99), 2)
            answer_text = f"{simulated_answer}\n\n**Confidence Score:** {confidence}"
            
        except Exception as e:
            answer_text = f"Error processing your query: {str(e)}\n\nPlease check your API key and try again."
            confidence = 0.0

    # 3️⃣ 显示 AI 回复并保存
    st.chat_message("assistant").write(answer_text)
    current_chat.append({"role": "assistant", "content": answer_text})

# ========= 显示embedding信息 =========
with st.expander("🔍 Embedding Information"):
    st.markdown("""
    **How Semantic Search Works:**
    - Convert question into a numerical vector (embedding)
    - Capture semantic meaning
    - Calculate similarity between question and document embeddings
    - Most relevant documents are returned based on semantic similarity
    """)
    
    if 'query_vector' in locals():
        st.metric("Embedding Dimension", vector_dim)
        st.write("First 10 embedding values:")
        st.code(str(query_vector[:10]))
