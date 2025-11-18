def build_augmented_prompt(user_query: str, search_results) -> str:
    """
    Combine the user's query with retrieved documents from Pinecone
    to form a retrieval-augmented (RAG) prompt.
    """
    context_list = []

    for i, match in enumerate(search_results.matches, start=1):
        context_text = match.metadata.get("text", "")
        context_list.append(f"[Document {i}]\n{context_text}")

    context_block = "\n\n".join(context_list)

    augmented_prompt = f"""
You are an intelligent assistant. Please answer the user's question
strictly based on the context provided below.

Guidelines:
1. Only use the information from the **Context** section to answer.
2. Do NOT fabricate or guess. 
3. If the answer is not present in the context, reply with:
   "The provided context does not contain the answer."

User Query:
{user_query}

Context:
{context_block}
""".strip()

    return augmented_prompt


# Example usage:
aug_prompt = build_augmented_prompt(query, search_results)
print(aug_prompt[:600] + " ...")
