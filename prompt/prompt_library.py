from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt for document analysis
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# Prompt for document comparison
document_comparison_prompt = ChatPromptTemplate.from_template("""
You are a precise document comparison assistant.

You will be given content from TWO documents:
- The first document is labelled <<REFERENCE DOCUMENT>>
- The second document is labelled <<ACTUAL DOCUMENT>>

YOUR TASK:
1. Carefully compare the content of the REFERENCE document against the ACTUAL document.
2. Identify every meaningful difference — changed values, added text, removed text, reworded sentences, etc.
3. For each difference, identify the section/topic it belongs to.
4. If the documents are identical or have NO meaningful differences, you MUST output an empty JSON array: []

STRICT RULES:
- Do NOT mention "PDF", "page number", or any format-specific term. Refer to document sections by topic or heading.
- Do NOT make up differences that do not exist in the provided text.
- Do NOT include differences that are only whitespace or formatting changes.
- Output ONLY valid JSON — no markdown fences, no explanations outside the JSON.

Input documents:

{combined_docs}

{format_instruction}
""")


# Prompt for contextual question rewriting
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Central dictionary to register prompts
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}