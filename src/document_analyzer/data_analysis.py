import re
import json
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompt.prompt_library import PROMPT_REGISTRY

EXPECTED_KEYS = {"Title", "Author", "DateCreated", "LastModifiedDate",
                 "Publisher", "Language", "PageCount", "SentimentTone", "Summary"}

# Max characters sent to the LLM per chunk — 70B has a large context but
# sending cleaner, focused text gives better results than raw noise.
MAX_CHUNK_CHARS = 12_000


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters before sending to LLM."""
    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs to a single space per line
    text = "\n".join(line.strip() for line in text.splitlines())
    # Remove non-printable characters (keep standard punctuation)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text.strip()


def _extract_json(raw: str) -> dict:
    """
    Try to extract a valid JSON object from the LLM's raw string output.
    Tries multiple strategies so minor formatting deviations don't fail silently.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()

    # Strategy 1: direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: find first {...} block
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 3: return partial result with raw as summary
    return {
        "Title": "Not Available",
        "Author": "Not Available",
        "DateCreated": "Not Available",
        "LastModifiedDate": "Not Available",
        "Publisher": "Not Available",
        "Language": "Not Available",
        "PageCount": "Not Available",
        "SentimentTone": "Neutral",
        "Summary": [raw[:500]] if raw else ["No content extracted."],
        "_parse_error": True,
    }


def _merge_results(results: list[dict]) -> dict:
    """
    Merge metadata extracted from multiple chunks.
    Strategy:
    - For scalar fields: take the first non-'Not Available' value found across chunks.
    - For Summary (list): collect all unique bullet points, cap at 5.
    - For SentimentTone: majority vote.
    """
    merged = {k: "Not Available" for k in EXPECTED_KEYS}
    all_summaries = []
    sentiment_votes: dict[str, int] = {}

    for r in results:
        if not isinstance(r, dict):
            continue
        for key in EXPECTED_KEYS - {"Summary", "SentimentTone"}:
            val = r.get(key, "Not Available")
            if merged[key] == "Not Available" and val and val != "Not Available":
                merged[key] = val

        summary = r.get("Summary", [])
        if isinstance(summary, list):
            all_summaries.extend(s for s in summary if s and s not in all_summaries)
        elif isinstance(summary, str) and summary:
            all_summaries.append(summary)

        tone = r.get("SentimentTone", "Neutral")
        sentiment_votes[tone] = sentiment_votes.get(tone, 0) + 1

    # Pick majority sentiment
    merged["SentimentTone"] = max(sentiment_votes, key=sentiment_votes.get) if sentiment_votes else "Neutral"
    # Cap summary to 5 unique bullets
    merged["Summary"] = all_summaries[:5] if all_summaries else ["Not Available"]
    return merged


class DocumentAnalyzer:
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.llm = ModelLoader().load_llm()
            self.prompt = PROMPT_REGISTRY["document_analysis"]
            # StrOutputParser — no OutputFixingParser that silently returns {}
            self.chain = self.prompt | self.llm | StrOutputParser()
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_CHUNK_CHARS,
                chunk_overlap=500,
            )
            self.log.info("DocumentAnalyzer initialized")
        except Exception as e:
            self.log.error("DocumentAnalyzer init failed", error=str(e))
            raise DocumentPortalException("Error in DocumentAnalyzer initialization", sys)

    def analyze_document(self, document_text: str) -> dict:
        try:
            # ── Pre-process: clean text before sending ──────────────────────
            cleaned = _clean_text(document_text)
            self.log.info("analyze_document: text cleaned", original_chars=len(document_text), cleaned_chars=len(cleaned))

            chunks = self.splitter.split_text(cleaned)
            self.log.info("analyze_document: text chunked", total_chunks=len(chunks))

            results: list[dict] = []

            # Process up to 3 chunks (rate-limit aware)
            for i, chunk in enumerate(chunks[:3]):
                self.log.info("analyze_document: processing chunk", chunk_number=i + 1, chars=len(chunk))
                try:
                    raw = self.chain.invoke({"document_text": chunk})
                    self.log.info(
                        "analyze_document: LLM responded",
                        chunk_number=i + 1,
                        response_length=len(raw),
                        preview=raw[:200],
                    )
                    parsed = _extract_json(raw)
                    if parsed.get("_parse_error"):
                        self.log.warning("analyze_document: JSON extraction fell back to raw", chunk_number=i + 1)
                    results.append(parsed)
                except Exception as chunk_err:
                    self.log.error("analyze_document: chunk failed — skipping", chunk_number=i + 1, error=str(chunk_err))
                    continue

            if not results:
                raise DocumentPortalException("All chunks failed to produce a result", sys)

            final = _merge_results(results)
            # Remove internal debug key before returning
            final.pop("_parse_error", None)

            self.log.info("analyze_document: completed", keys=list(final.keys()))
            return final

        except DocumentPortalException:
            raise
        except Exception as e:
            self.log.error("analyze_document: unexpected error", error=str(e))
            raise DocumentPortalException("Metadata extraction failed", sys)
