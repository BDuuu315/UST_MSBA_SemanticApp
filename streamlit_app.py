import streamlit as st
import random

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
    st.session_state["conversations"] = []  # 会话列表，每个会话为 list[{"role":"user/assistant", "content":...}]
if "active_chat_index" not in st.session_state:
    st.session_state["active_chat_index"] = None  # 当前激活的会话索引
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = None

# ========= Sidebar =========
st.sidebar.title("💬 Chat Sidebar")

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
if st.sidebar.button("➕ New Chat"):
    st.session_state["conversations"].append([])  # 新增一个空会话
    st.session_state["active_chat_index"] = len(st.session_state["conversations"]) - 1

# --- 展示历史会话列表 ---
st.sidebar.subheader("History")
if len(st.session_state["conversations"]) == 0:
    st.sidebar.info("No history yet. Click '➕ New Chat' to start.")
else:
    for i in range(len(st.session_state["conversations"])):
        label = f"Chat {i+1}"
        if i == st.session_state["active_chat_index"]:
            st.sidebar.button(label, key=f"chat_active_{i}", disabled=True)
        else:
            if st.sidebar.button(label, key=f"chat_{i}"):
                st.session_state["active_chat_index"] = i

# --- 清空历史按钮 ---
if st.sidebar.button("🗑️ Clear All History"):
    st.session_state["conversations"].clear()
    st.session_state["active_chat_index"] = None
    st.sidebar.success("All chat history cleared.")

st.sidebar.markdown("---")
st.sidebar.markdown("[Get your OpenAI API Key](https://platform.openai.com/account/api-keys)")

# ========= 主区内容 =========
st.title("Semantic Search AI Chat for BA Users")
st.caption("A Semantic Search App prototype for ISOM 6670G.")

# --- 当前会话内容 ---
if st.session_state["active_chat_index"] is None:
    st.info("👋 Click *'➕ New Chat'* in the sidebar to start a conversation.")
else:
    current_chat = st.session_state["conversations"][st.session_state["active_chat_index"]]

    # 显示历史消息
    for msg in current_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # --- 新输入 ---
    user_query = st.chat_input("Type your question here...")

    if user_query:
        if not st.session_state.get("OPENAI_API_KEY"):
            st.error("Please add your OpenAI API key in the sidebar first.")
        else:
            # 1️⃣ 保存用户问题
            current_chat.append({"role": "user", "content": user_query})

            # 2️⃣ 模拟系统回答
            with st.spinner("Processing..."):
                simulated_answer = (
                    "Our semantic engine retrieves and ranks documents "
                    "based on meaning similarity using embeddings."
                )
                confidence = round(random.uniform(0.75, 0.99), 2)
                answer_text = f"{simulated_answer}\n\n**Confidence Score:** {confidence}"

            # 3️⃣ 保存回答并显示
            current_chat.append({"role": "assistant", "content": answer_text})
            with st.chat_message("assistant"):
                st.write(answer_text)
