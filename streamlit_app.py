from pinecone import Pinecone

# --- 新增: Pinecone初始化 ---
# 你可以选择在代码里直接写 key，或在 sidebar 添加输入框
PINECONE_API_KEY = st.sidebar.text_input(
    "Enter your Pinecone API Key",
    type="password",
    help="Paste your Pinecone API key here."
)
PINECONE_INDEX_NAME = st.sidebar.text_input(
    "Enter your Pinecone Index Name",
    value="sample-movies",  # 默认名
    help="The name of your Pinecone index."
)
top_k = st.sidebar.slider("Number of documents to return", 1, 10, 3)

# 初始化 Pinecone 客户端（缓存资源）
@st.cache_resource
def get_pinecone_client(api_key):
    pc = Pinecone(api_key=api_key)
    return pc

# 在主对话逻辑中，替换原有模拟搜索部分
if user_query:
    if not st.session_state.get("OPENAI_API_KEY"):
        st.error("Please input your HKUST OpenAI API key first.")
        st.stop()

    if not PINECONE_API_KEY:
        st.error("Please input your Pinecone API key in the sidebar.")
        st.stop()

    # --- 显示 & 缓存用户输入 ---
    st.chat_message("user").write(user_query)
    current_chat.append({"role": "user", "content": user_query})

    if len(current_chat) == 1:
        st.session_state["conversation_titles"][chat_index] = user_query[:40]

    with st.spinner("Embedding + Searching Pinecone..."):
        try:
            # 初始化 Azure OpenAI
            openai_client = get_azure_client(st.session_state["OPENAI_API_KEY"])
            # 生成 embedding
            response = openai_client.embeddings.create(
                input=user_query,
                model="text-embedding-ada-002"
            )
            query_vector = response.data[0].embedding

            # 初始化 Pinecone
            pc = get_pinecone_client(PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)

            # 🔍 执行语义搜索
            pinecone_results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )

            if len(pinecone_results.matches) == 0:
                answer_text = "No results found in Pinecone index."
            else:
                # 将搜索结果格式化为文本
                answer_text = "### 🔎 Semantic Search Results\n\n"
                for rank, match in enumerate(pinecone_results.matches, start=1):
                    meta = match.metadata or {}
                    answer_text += (
                        f"**{rank}. {meta.get('title', 'Unknown Title')}** "
                        f"(Score: {match.score:.4f})\n"
                        f"- Genre: {meta.get('genre', 'N/A')}\n"
                        f"- Year: {meta.get('year', 'N/A')}\n"
                        f"- Box Office: {meta.get('box-office', 'N/A')}\n"
                        f"- Summary: {meta.get('summary', 'N/A')[:250]}...\n\n"
                    )
        except Exception as e:
            answer_text = f"⚠️ Error searching Pinecone: {e}"

    st.chat_message("assistant").markdown(answer_text)
    current_chat.append({"role": "assistant", "content": answer_text})
