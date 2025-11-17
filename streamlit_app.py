import streamlit as st
import requests
import json
import random
import os

#First Page
st.markdown(
    """
    <style>
    .logo { 
        position: absolute;
        top: 15px;
        left: 15px;
        z-index: 100; 
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.image("Logo_USTBusinessSchool.svg", width=120, output_format="SVG")

st.set_page_config(
    page_title="Semantic Search AI App for BA Users",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 输入 OpenAI API Key ---
openai_api_key = st.sidebar.text_input(
    "Enter your UST OpenAI API Key",
    type="password",
    help="You can get you API Key via syllabus instruction"
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


st.title("Semantic Search AI App for BA Users")
st.markdown("A Semantic Search App for ISOM 6670G.")

st.subheader("What is your question?")
user_query = st.text_input(
    label="Enter your question:",
    placeholder="e.g., Where is HKUST Business School",
    help="Type your natural language question here."
)

# Button for Search
if st.button("Search"):
    if not user_query:
        st.warning("Please enter a question before submitting.")
    else:
        #
        payload = {"query": user_query}

        st.info("Processing...")

        # ------------------------------
        # 调用
        # ------------------------------
        # ❗当有后端API时，放开下方注释：
        # response = requests.post("http://localhost:8000/api/search", json=payload)
        # result = response.json()

        
        simulated_backend_output = {
            "status": "success",
            "semantic_answer": "Our Semantic search works by comparing the meaning of entered question with document embeddings.",
            "confidence": round(random.uniform(0.75, 0.99), 2)
        }

        # ------------------------------
        # Result
        # ------------------------------
        if simulated_backend_output["status"] == "success":
            st.success("Query Processed Successfully!")
            st.subheader("Semantic Result:")
            st.write(simulated_backend_output["semantic_answer"])
            st.caption(f"Confidence Score: {simulated_backend_output['confidence']}")
        else:
            st.error("Backend returned an error. Please try again.")


