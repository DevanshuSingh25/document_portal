from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ================= DOCUMENT ANALYSIS PROMPT =================
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert document analysis AI. Your task is to extract structured metadata from the given document text.

CRITICAL INSTRUCTIONS (STRICTLY FOLLOW):
- You MUST return ONLY valid JSON. Nothing else.
- Do NOT include explanations, reasoning, notes, markdown, or code blocks.
- Do NOT include thinking tags like <think> or any hidden reasoning.
- Every field must be present.
- If a value is missing or unclear, return "Not Available".
- If a value is not explicitly present in the text, return "Not Available". Do NOT infer or guess.

OUTPUT FORMAT — return exactly this JSON structure:
{{
  "Title": "<document title or 'Not Available'>",
  "Author": "<all authors comma-separated, or 'Not Available'>",
  "DateCreated": "<creation date or 'Not Available'>",
  "LastModifiedDate": "<last modified date or 'Not Available'>",
  "Publisher": "<publisher name or 'Not Available'>",
  "Language": "<language name>",
  "PageCount": "<number as string, or 'Not Available'>",
  "SentimentTone": "<exactly one of: Positive | Negative | Neutral>",
  "Summary": [
    "<bullet sentence 1>",
    "<bullet sentence 2>",
    "<bullet sentence 3>"
  ]
}}

GUIDELINES:
- Summary must have 3 to 5 concise bullet sentences.
- Each sentence must be under 25 words.
- Each bullet must capture a key idea or contribution.
- Avoid generic phrases like "This document discusses...".
- SentimentTone must be exactly one of: Positive, Negative, Neutral.

FIELD EXTRACTION PRIORITY:
- Title and Author are highest priority — extract if present anywhere.
- Dates may appear in headers, footers, or references.
- Publisher may appear as conference, journal, or organization name.

IMPORTANT:
- This may be a chunk of a larger document.
- Extract ONLY from the provided text.
- Do NOT assume missing context.

FINAL CHECK (MANDATORY):
- Ensure output is valid JSON
- Ensure all fields are present
- Ensure no extra text is included
- Ensure Summary has 3–5 items
- Ensure SentimentTone is valid

DOCUMENT TEXT:
{document_text}
""")


# ================= DOCUMENT COMPARISON PROMPT =================
document_comparison_prompt = ChatPromptTemplate.from_template("""
You are a precise document comparison assistant.

You will be given content from TWO documents:
- The first document is labelled <<REFERENCE DOCUMENT>>
- The second document is labelled <<ACTUAL DOCUMENT>>

YOUR TASK:
1. Carefully compare the REFERENCE document with the ACTUAL document.
2. Identify ALL meaningful differences:
   - Changed values
   - Added content
   - Removed content
   - Reworded sentences
3. Group differences by topic or logical section.

STRICT RULES:
- Output ONLY valid JSON (no markdown, no explanations).
- If there are NO meaningful differences, return: []
- Do NOT mention page numbers, formatting, or file types (e.g., PDF).
- Do NOT include whitespace-only differences.
- Do NOT hallucinate or invent differences.

OUTPUT FORMAT:
{format_instructions}

INPUT DOCUMENTS:
{combined_docs}

FINAL CHECK:
- Ensure output is valid JSON
- Ensure no extra text outside JSON
""")


# ================= CONTEXTUAL QUESTION REWRITE =================
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the latest user query, rewrite the query into a standalone question. "
        "The rewritten question must be understandable without previous context. "
        "Do NOT answer the question — only rewrite it if necessary. "
        "If it is already clear, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ================= CONTEXT-BASED QA =================
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI assistant answering questions based ONLY on the provided context.\n\n"
        "RULES:\n"
        "- Use only the given context to answer.\n"
        "- If the answer is not present, respond with: 'I don't know.'\n"
        "- Do NOT hallucinate or guess.\n"
        "- Keep the answer concise (max 3 sentences).\n\n"
        "CONTEXT:\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ================= PROMPT REGISTRY =================
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}