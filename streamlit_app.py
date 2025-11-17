import streamlit as st
import random
import os

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

# ========= Sidebar 部分 =========
st.sidebar.title("Sidebar")

# --- 输入 OpenAI API Key ---
openai_api_key = st.sidebar.text_input(
    "Enter your UST OpenAI API Key",
    type="password",
    help="Check ISOM 6670G Syllabus to set up HKUST OpenAI account and get your OpenAI API Key"
)

# 在 session_state 中保存
if openai_api_key:
    st.session_state["OPENAI_API_KEY"] = openai_api_key

# --- 聊天历史记录 ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.sidebar.markdown("---")
st.sidebar.subheader("Chat History")

# 如果有历史记录，则显示
if len(st.session_state["chat_history"]) == 0:
    st.sidebar.info("No chat history yet.")
else:
    for i, msg in enumerate(st.session_state["chat_history"]):
        st.sidebar.button(f"{i+1}. {msg['query'][:20]}...")

# --- 清除历史按钮 ---
if st.sidebar.button("Clear History"):
    st.session_state["chat_history"] = []
    st.sidebar.success("Chat history cleared!")


# ========= 主体部分 =========
st.title("Semantic Search AI Chat for BA Users")
st.caption("A Semantic Search App prototype for ISOM 6670G.")

st.subheader("What is your question?")

# --- 输入框 ---
user_query = st.chat_input("Type your question here...")

if user_query:
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please add your OpenAI API key in the sidebar first.")
    else:
        st.session_state["chat_history"].append({"query": user_query})

        with st.spinner("Processing..."):
            # 模拟 Semantic 搜索结果 + 动态置信度
            simulated_backend_output = {
                "status": "success",
                "semantic_answer": (
                    "Our semantic engine retrieves and ranks documents "
                    "based on meaning similarity using embeddings."
                ),
                "confidence": round(random.uniform(0.75, 0.99), 2)
            }

        if simulated_backend_output["status"] == "success":
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(simulated_backend_output["semantic_answer"])
            st.caption(f"**Confidence Score:** {simulated_backend_output['confidence']}")
        else:
            st.error("Backend error. Please try again later.")

# ========= 页脚 =========
#st.markdown("---")
#st.markdown(
#    """
#    <div style='text-align: center; color: gray; font-size: small;'>
#    © 2025 HKUST ISOM 6670G Semantic Search Demo | Streamlit Front End
#    </div>
#    """,
#    unsafe_allow_html=True
#)
