import streamlit as st
import random
import os

#page set
st.set_page_config(
    page_title="Semantic Search AI Chat",
    layout="centered",
    initial_sidebar_state="collapsed",
)

#Logo
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

#Sidebar
st.sidebar.title("Sidebar")

#OpenAI Key
openai_api_key = st.sidebar.text_input(
    "Enter your UST OpenAI API Key",
    type="password",
    help="Check ISOM 6670G Syllabus to set up HKUST OpenAI account and get your OpenAI API Key"
)

#session state save
if openai_api_key:
    st.session_state["OPENAI_API_KEY"] = openai_api_key

#history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "active_chat_index" not in st.session_state:
    st.session_state["active_chat_index"] = None  # 当前查看的对话编号
st.sidebar.markdown("---")
st.sidebar.subheader("Chat History")

#show history

if len(st.session_state["chat_history"]) == 0:
    st.sidebar.info("No chat history yet.")
else:
    for i, msg in enumerate(st.session_state["chat_history"]):
        # 在每条消息前增加编号按钮（点击查看，不触发重复查询）
        if st.sidebar.button(f"🗨️ {i+1}. {msg['query'][:25]}"):
            st.session_state["active_chat_index"] = i

#clear history
if st.sidebar.button("Clear History"):
    st.session_state["chat_history"] = []
    st.session_state["active_chat_index"] = None
    st.sidebar.success("Chat history cleared!")


#main part
st.title("Semantic Search AI Chat for BA Users")
st.caption("A Semantic Search App prototype for ISOM 6670G.")

st.subheader("What is your question?")

if st.session_state["active_chat_index"] is not None:
    selected = st.session_state["chat_history"][st.session_state["active_chat_index"]]
    st.info(f"🕘 Viewing conversation #{st.session_state['active_chat_index'] + 1}")
    st.chat_message("user").write(selected["query"])
    st.chat_message("assistant").write(selected["answer"])
    st.caption(f"Confidence Score: {selected['confidence']}")
else:
    user_query = st.text_input(
    label="Enter your question:",
    placeholder="e.g., Where is HKUST Business School",
    help="Type your natural language question here."
    )
    
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
