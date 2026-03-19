import re
import sys
import json
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType


class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        # Use plain string output — avoids OutputFixingParser silently producing []
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=500,
        )
        self.log.info("DocumentComparatorLLM initialized")

    # ──────────────────────────────────────────────────
    # Internal: try to pull a list of {Section, Changes}
    # dicts out of whatever the LLM returned.
    # ──────────────────────────────────────────────────
    def _extract_rows(self, raw: str) -> list[dict]:
        """
        Try multiple parsing strategies to extract rows from LLM output:
        1. JSON array  -> parse directly
        2. JSON object with a list value -> unwrap first list
        3. Markdown table -> convert rows
        4. Numbered plain-text bullets -> convert to rows
        """
        cleaned = raw.strip()

        # ── Strategy 1: look for a JSON array anywhere in the string ──
        json_match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and parsed:
                    self.log.info("_extract_rows: parsed via JSON array", rows=len(parsed))
                    return parsed
            except json.JSONDecodeError:
                self.log.warning("_extract_rows: JSON array match found but failed to parse")

        # ── Strategy 2: look for a JSON object with a list value ──
        obj_match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group())
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list) and v:
                            self.log.info("_extract_rows: parsed via JSON object key", rows=len(v))
                            return v
            except json.JSONDecodeError:
                self.log.warning("_extract_rows: JSON object match found but failed to parse")

        # ── Strategy 3: markdown table ──
        rows = []
        lines = cleaned.splitlines()
        in_table = False
        for line in lines:
            if "|" in line and "---" not in line:
                cols = [c.strip() for c in line.split("|") if c.strip()]
                if len(cols) >= 2:
                    if not in_table:
                        in_table = True  # header row — skip
                        continue
                    rows.append({"Section": cols[0], "Changes": " | ".join(cols[1:])})
        if rows:
            self.log.info("_extract_rows: parsed via markdown table", rows=len(rows))
            return rows

        # ── Strategy 4: numbered bullet list ──
        rows = []
        for line in lines:
            m = re.match(r"^\s*[\d\*\-]+[\.\)]\s*(.+?)[:\-]\s*(.+)$", line)
            if m:
                rows.append({"Section": m.group(1).strip(), "Changes": m.group(2).strip()})
        if rows:
            self.log.info("_extract_rows: parsed via bullet list", rows=len(rows))
            return rows

        # ── Fallback: wrap entire response as one row ──
        self.log.warning(
            "_extract_rows: all strategies failed — wrapping raw output as single row",
            preview=cleaned[:200],
        )
        return [{"Section": "Full comparison", "Changes": cleaned}]

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            chunks = self.splitter.split_text(combined_docs)
            self.log.info("Document chunked for comparison", total_chunks=len(chunks))

            all_rows: list[dict] = []

            # Process up to 3 chunks to stay within rate limits
            for i, chunk in enumerate(chunks[:3]):
                self.log.info("Processing comparison chunk", chunk_number=i + 1, chunk_size=len(chunk))
                inputs = {
                    "combined_docs": chunk,
                    "format_instruction": (
                        'Return a JSON array. Each element must have exactly two keys: '
                        '"Section" (describe which part/topic of the document changed) '
                        'and "Changes" (describe what changed in detail). '
                        'Example: [{"Section": "Introduction", "Changes": "Paragraph reworded"}]'
                    ),
                }
                try:
                    raw_response = self.chain.invoke(inputs)
                    self.log.info(
                        "Chunk LLM response received",
                        chunk_number=i + 1,
                        response_length=len(raw_response),
                        preview=raw_response[:300],
                    )
                    rows = self._extract_rows(raw_response)
                    all_rows.extend(rows)
                    self.log.info("Chunk compared successfully", chunk_number=i + 1, rows_extracted=len(rows))
                except Exception as chunk_err:
                    self.log.error(
                        "Error processing chunk — skipping",
                        chunk_number=i + 1,
                        error=str(chunk_err),
                    )
                    continue

            self.log.info("Comparison completed", total_rows=len(all_rows))
            return self._format_response(all_rows)

        except Exception as e:
            self.log.error("Error in compare_documents", error=str(e))
            raise DocumentPortalException("Error comparing documents", sys)

    def _format_response(self, rows: list[dict]) -> pd.DataFrame:
        try:
            if not rows:
                self.log.warning("_format_response: no rows extracted — returning empty DataFrame")
                return pd.DataFrame(columns=["Section", "Changes"])
            df = pd.DataFrame(rows)
            # Ensure expected columns always present even if LLM used different keys
            for col in ["Section", "Changes"]:
                if col not in df.columns:
                    df[col] = ""
            self.log.info("DataFrame built", shape=str(df.shape), columns=list(df.columns))
            return df
        except Exception as e:
            self.log.error("Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)
