import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import pandas as pd

# ----------------------------
# 配置项
# ----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "sample-movies"  # 你的索引名（与控制台一致）

# ----------------------------
# 初始化客户端
# ----------------------------
@st.cache_resource
def get_clients():
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    return openai_client, index

openai_client, index = get_clients()

# ----------------------------
# 页面配置
# ----------------------------
st.set_page_config(
    page_title="🎬 Movie Semantic Search",
    page_icon="🎞️",
    layout="wide"
)
st.title("🎬 Movie Semantic Search (Powered by Pinecone + OpenAI)")

# ----------------------------
# 用户输入
# ----------------------------
query_text = st.text_input(
    "🔍 输入你的语义搜索内容（例如：'电影中主角保护外星种族'）",
    placeholder="try: 'About aliens and a human connecting emotionally'",
)

k = st.slider("返回 Top-K 结果", min_value=1, max_value=10, value=5)

# ----------------------------
# 执行搜索
# ----------------------------
if st.button("开始搜索") and query_text.strip():
    with st.spinner("Embedding + Searching..."):
        # 1️⃣ 生成 query embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        query_embedding = response.data[0].embedding

        # 2️⃣ 查询 Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

    st.success(f"找到 {len(results.matches)} 条相关记录")

    # ----------------------------
    # 展示结果
    # ----------------------------
    for i, match in enumerate(results.matches):
        meta = match.metadata
        with st.container():
            st.markdown(f"### 🏷️ {i+1}")
            st.markdown(f"**ID:** `{match.id}`")
            st.markdown(f"**SCORE:** `{match.score:.5f}`")
            st.markdown(
                f"""
                - **title:** *{meta.get('title', 'N/A')}*  
                - **year:** {meta.get('year', 'N/A')}  
                - **genre:** {meta.get('genre', 'N/A')}  
                - **box-office:** {meta.get('box-office', 'N/A'):,}  
                - **summary:** {meta.get('summary', 'N/A')}
                """
            )
            st.divider()

# ----------------------------
# 底部信息
# ----------------------------
st.markdown(
    "<div style='text-align:center; font-size:0.9em; color:gray;'>Built with ❤️ using Pinecone + OpenAI + Streamlit</div>",
    unsafe_allow_html=True
)
