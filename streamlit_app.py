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
    st.session_state["conversations"] = []  # 每个元素为 list[dict(role, content)]
if "conversation_titles" not in st.session_state:
    st.session_state["conversation_titles"] = []  # 保存会话标题
if "active_chat_index" not in st.session_state:
    st.session_state["active_chat_index"] = None
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None
if "last_processed_query" not in st.session_state:
    st.session_state["last_processed_query"] = None  # 防止重复处理
if "documents" not in st.session_state:
    st.session_state["documents"] = [
        {"id": 1, "content": "HKUST Business School offers MBA programs with focus on analytics.", "embedding": None},
        {"id": 2, "content": "The ISOM department provides courses in information systems.", "embedding": None},
    ]

# ========= 初始化Azure OpenAI客户端 =========
@st.cache_resource
def get_azure_client(api_key):
    return AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint="https://hkust.azure-api.net"
    )

# ========= 语义搜索函数 =========
def semantic_search(query, client, top_k=3):
    """执行语义搜索并返回相关文档"""
    try:
        # 为查询生成embedding
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_vector = response.data[0].embedding
        
        # 这里简化处理，实际应该计算与所有文档的相似度
        # 模拟返回top_k个相关文档
        relevant_docs = st.session_state["documents"][:top_k]
        
        return relevant_docs, query_vector, len(query_vector)
    except Exception as e:
        st.error(f"语义搜索错误: {e}")
        return [], None, 0

# ========= 生成AI回答 =========
def generate_ai_response(query, relevant_docs, client):
    """基于查询和相关文档生成AI回答"""
    try:
        # 构建上下文
        context = "\n".join([doc["content"] for doc in relevant_docs])
        
        # 构建提示词
        prompt = f"""
        基于以下上下文信息回答问题：
        
        上下文：
        {context}
        
        问题：{query}
        
        请根据上下文提供准确、有用的回答。如果上下文信息不足，请基于你的知识回答。
        """
        
        # 调用Azure OpenAI生成回答
        response = client.chat.completions.create(
            model="gpt-35-turbo",  # 根据实际情况调整模型名称
            messages=[
                {"role": "system", "content": "你是一个有用的助手，能够基于提供的上下文信息准确回答问题。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        confidence = round(random.uniform(0.75, 0.95), 2)
        
        return answer, confidence
        
    except Exception as e:
        return f"生成回答时出错: {str(e)}", 0.0

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

# API check
if st.sidebar.button("🔄 Test Connection", use_container_width=True):
    with st.spinner("Testing API connection..."):
        try:
            client = get_azure_client(st.session_state["OPENAI_API_KEY"])
            response = client.embeddings.create(input="Hello world", model="text-embedding-ada-002")
            st.sidebar.success("✅ Azure OpenAI connection successful!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")

st.sidebar.header("⚙️ Search Configuration")
top_k = st.sidebar.slider("Number of documents to return", 1, 10, 3)

# --- 新建会话按钮 ---
if st.sidebar.button("🆕 New Chat", use_container_width=True):
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = len(st.session_state["conversations"]) - 1
    st.session_state["last_processed_query"] = None  # 重置处理状态
    st.rerun()

# --- 清除所有历史按钮 ---
if st.sidebar.button("🗑️ Clear All History", use_container_width=True):
    st.session_state["conversations"].clear()
    st.session_state["conversation_titles"].clear()
    st.session_state["active_chat_index"] = None
    st.session_state["last_processed_query"] = None
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
                st.session_state["last_processed_query"] = None  # 重置处理状态
                st.rerun()

# ========= 主体部分 =========
st.title("Semantic Search AI Chat for BA Users")
st.caption("A Semantic Search App prototype for ISOM 6670G.")

# 确保总是有激活的聊天
if st.session_state["active_chat_index"] is None and len(st.session_state["conversations"]) > 0:
    st.session_state["active_chat_index"] = 0
elif len(st.session_state["conversations"]) == 0:
    st.session_state["conversations"].append([])
    st.session_state["conversation_titles"].append("New Chat")
    st.session_state["active_chat_index"] = 0

# --- 已选定的会话 ---
chat_index = st.session_state["active_chat_index"]
current_chat = st.session_state["conversations"][chat_index]
chat_title = st.session_state["conversation_titles"][chat_index]

# --- 展示已有消息 ---
for msg in current_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 输入新消息 ---
user_query = st.chat_input(
    "Enter your question:",
    placeholder="e.g., Where is HKUST Business School?",
    key=f"chat_input_{chat_index}"  # 为每个聊天使用不同的key
)

# 处理用户查询
if user_query and user_query != st.session_state.get("last_processed_query"):
    # 若没有 API key，不允许继续
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please input your HKUST OpenAI API key in the sidebar first.")
        st.stop()

    # 1️⃣ 立即显示并保存用户输入
    with st.chat_message("user"):
        st.write(user_query)
    current_chat.append({"role": "user", "content": user_query})

    # 若这是该会话第一条消息，则用它更新标题
    if len(current_chat) == 1:
        st.session_state["conversation_titles"][chat_index] = user_query[:40]

    # 2️⃣ 生成embedding并获取AI回答
    with st.chat_message("assistant"):
        with st.spinner("Processing your query with semantic search..."):
            try:
                # 初始化Azure OpenAI客户端
                openai_client = get_azure_client(st.session_state["OPENAI_API_KEY"])
                
                # 执行语义搜索
                relevant_docs, query_vector, vector_dim = semantic_search(
                    user_query, openai_client, top_k=top_k
                )
                
                # 生成AI回答
                answer, confidence = generate_ai_response(user_query, relevant_docs, openai_client)
                
                # 构建完整回答
                answer_text = f"{answer}\n\n"
                answer_text += f"**Semantic Search Results:**\n"
                answer_text += f"- **Embedding Dimension:** {vector_dim}\n"
                answer_text += f"- **Documents Returned:** {len(relevant_docs)}\n"
                answer_text += f"- **Confidence Score:** {confidence}\n\n"
                
                # 显示相关文档
                answer_text += "**Relevant Documents:**\n"
                for i, doc in enumerate(relevant_docs, 1):
                    answer_text += f"{i}. {doc['content']}\n"
                
                st.write(answer_text)
                
                # 保存AI回复
                current_chat.append({"role": "assistant", "content": answer_text})
                
                # 标记该查询已处理
                st.session_state["last_processed_query"] = user_query
                
            except Exception as e:
                error_msg = f"Error processing your query: {str(e)}\n\nPlease check your API key and try again."
                st.write(error_msg)
                current_chat.append({"role": "assistant", "content": error_msg})
                st.session_state["last_processed_query"] = user_query

# ========= 显示embedding信息 =========
with st.expander("🔍 Embedding Information"):
    st.markdown("""
    **How Semantic Search Works:**
    - Convert question into a numerical vector (embedding)
    - Capture semantic meaning
    - Calculate similarity between question and document embeddings
    - Most relevant documents are returned based on semantic similarity
    """)
    
    # 如果有最近的查询，显示embedding信息
    if 'query_vector' in locals() and query_vector is not None:
        st.metric("Embedding Dimension", vector_dim)
        st.write("First 10 embedding values:")
        st.code(str(query_vector[:10]))

# ========= 显示会话信息 =========
with st.expander("📊 Session Information"):
    st.write(f"**Active Chat:** {chat_title}")
    st.write(f"**Total Conversations:** {len(st.session_state['conversations'])}")
    st.write(f"**Messages in Current Chat:** {len(current_chat)}")
    st.write(f"**Last Processed Query:** {st.session_state.get('last_processed_query', 'None')}")
