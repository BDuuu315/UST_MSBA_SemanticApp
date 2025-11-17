import streamlit as st
import random

# ========= 页面配置 =========
st.set_page_config(
    page_title="Semantic Search AI Chat",
    layout="wide",
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

# ========= Sidebar 控制面板 =========
st.sidebar.title("⚙️ Control Panel")

api_key = st.sidebar.text_input(
    "🔑 Enter your OpenAI API Key",
    type="password",
    help="You can get one at https://platform.openai.com/account/api-keys"
)
if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

# 初始化 chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "active_chat_index" not in st.session_state:
    st.session_state["active_chat_index"] = None  # 当前查看的对话编号

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Chat History")

# ------- 历史会话展示 -------
if len(st.session_state["chat_history"]) == 0:
    st.sidebar.info("No chat history yet.")
else:
    for i, msg in enumerate(st.session_state["chat_history"]):
        # 在每条消息前增加编号按钮（点击查看，不触发重复查询）
        if st.sidebar.button(f"🗨️ {i+1}. {msg['query'][:25]}"):
            st.session_state["active_chat_index"] = i

# 清除聊天记录按钮
if st.sidebar.button("🗑️ Clear History"):
    st.session_state["chat_history"] = []
    st.session_state["active_chat_index"] = None
    st.sidebar.success("Chat history cleared!")

st.sidebar.markdown("---")
st.sidebar.markdown("[Get an OpenAI API Key](https://platform.openai.com/account/api-keys)")

# ========= 主体部分 =========
st.title("Semantic Search AI Chat for BA Users")
st.caption("A Semantic Search App prototype for ISOM 6670G.")

# 如果点击了历史记录按钮，显示内容
if st.session_state["active_chat_index"] is not None:
    selected = st.session_state["chat_history"][st.session_state["active_chat_index"]]
    st.info(f"🕘 Viewing conversation #{st.session_state['active_chat_index'] + 1}")
    st.chat_message("user").write(selected["query"])
    st.chat_message("assistant").write(selected["answer"])
    st.caption(f"Confidence Score: {selected['confidence']}")
else:
    # ======= 输入新问题 =======
    user_query = st.chat_input("Type your question here...")

    if user_query:
        # 检查 API key
        if not st.session_state.get("OPENAI_API_KEY"):
            st.error("⚠️ Please add your OpenAI API key in the sidebar first.")
        else:
            with st.spinner("🔍 Processing your query..."):
                # 模拟 Semantic 搜索结果
                simulated_answer = (
                    "Our semantic engine retrieves and ranks documents "
                    "based on meaning similarity using embeddings."
                )
                confidence = round(random.uniform(0.75, 0.99), 2)

            # --- 显示并保存结果 ---
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(simulated_answer)
            st.caption(f"Confidence Score: {confidence}")

            # 保存进历史记录
            st.session_state["chat_history"].append({
                "query": user_query,
                "answer": simulated_answer,
                "confidence": confidence
            })
