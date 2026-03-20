"""
Two-pass Map-Reduce document comparator:

Pass 1 (Map):  Extract structured key facts from each document independently.
               This gives the LLM focused, clean input instead of raw mixed text.

Pass 2 (Reduce): Compare the extracted fact sets and produce a list of differences.
                 The LLM receives a much smaller, semantically clean input.
"""
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


MAX_EXTRACT_CHARS = 4000     # chars sent per doc in extraction pass
MAX_COMPARE_CHARS = 8000     # chars of combined extracted facts for compare pass


class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        loader = ModelLoader()
        # Both passes use text mode (no JSON object mode — output is an array)
        llm = loader.load_llm_text()

        # Pass 1: fact extraction chain (JSON object mode is fine here — returns a dict)
        self.extract_chain = (
            PROMPT_REGISTRY["document_extraction"] | loader.load_llm() | StrOutputParser()
        )

        # Pass 2: comparison chain
        self.compare_chain = (
            PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value] | llm | StrOutputParser()
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_EXTRACT_CHARS,
            chunk_overlap=200,
        )
        self.log.info("DocumentComparatorLLM initialized (two-pass Map-Reduce)")

    # ── Pass 1: Extract structured facts from a single document ────────────────
    def _extract_facts(self, text: str, role: str) -> dict:
        """
        Ask the LLM to extract key facts grouped by topic from one document.
        Returns a dict like {"Metrics": ["Accuracy: 0.94", ...], "Config": [...]}
        """
        # Use only the first meaningful chunk for extraction
        chunk = text[:MAX_EXTRACT_CHARS].strip()
        self.log.info("_extract_facts: extracting facts", role=role, chars=len(chunk))
        try:
            raw = self.extract_chain.invoke({"document_text": chunk, "role": role})
            self.log.info("_extract_facts: LLM responded", role=role, preview=raw[:200])
            # Strip markdown fences if present
            raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
            # Find JSON object
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                parsed = json.loads(m.group())
                if isinstance(parsed, dict):
                    self.log.info("_extract_facts: parsed OK", role=role, topics=list(parsed.keys()))
                    return parsed
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str:
                self.log.error("_extract_facts: rate limit hit, failing immediately")
                raise
            self.log.warning("_extract_facts: failed, falling back to raw text", role=role, error=str(e))
        # Fallback: return plain text wrapped in dict
        return {"General": [chunk[:2000]]}

    def _facts_to_text(self, facts: dict, label: str) -> str:
        """Convert a facts dict to a labelled text block for the compare prompt."""
        lines = [f"<< {label} >>"]
        for topic, items in facts.items():
            lines.append(f"\n[{topic}]")
            for item in items:
                lines.append(f"  - {item}")
        return "\n".join(lines)

    # ── Pass 2: Compare extracted facts ──────────────────────────────────────
    def _extract_rows(self, raw: str) -> list[dict]:
        """Robust parser — tries JSON array, then object, then markdown table, then bullets."""
        cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()

        # Strategy 1: JSON array
        m = re.search(r"\[[\s\S]*\]", cleaned)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list):
                    self.log.info("_extract_rows: parsed via JSON array", rows=len(parsed))
                    return parsed
            except json.JSONDecodeError:
                pass

        # Strategy 2: JSON object with a list value
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, dict):
                    for v in parsed.values():
                        if isinstance(v, list) and v:
                            return v
            except json.JSONDecodeError:
                pass

        # Strategy 3: Markdown table
        rows, in_table = [], False
        for line in cleaned.splitlines():
            if "|" in line and "---" not in line:
                cols = [c.strip() for c in line.split("|") if c.strip()]
                if len(cols) >= 2:
                    if not in_table:
                        in_table = True
                        continue
                    rows.append({"Section": cols[0], "Changes": " | ".join(cols[1:])})
        if rows:
            self.log.info("_extract_rows: parsed via markdown table", rows=len(rows))
            return rows

        # Strategy 4: Numbered / bullet list
        rows = []
        for line in cleaned.splitlines():
            m2 = re.match(r"^\s*[\d\*\-]+[\.:\)]\s*\*{0,2}(.+?)\*{0,2}[:\-]\s*(.+)$", line)
            if m2:
                rows.append({"Section": m2.group(1).strip(), "Changes": m2.group(2).strip()})
        if rows:
            self.log.info("_extract_rows: parsed via bullet list", rows=len(rows))
            return rows

        # Fallback
        self.log.warning("_extract_rows: all strategies failed", preview=cleaned[:200])
        return [{"Section": "Full comparison", "Changes": cleaned}] if cleaned else []

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        """
        Two-pass approach:
        1. Split combined_docs back into reference / actual, extract facts from each.
        2. Feed extracted facts into the comparison prompt.
        """
        try:
            # ── Split reference and actual sections from the combined text ──
            ref_text, act_text = self._split_combined(combined_docs)
            self.log.info("compare_documents: split complete",
                          ref_chars=len(ref_text), act_chars=len(act_text))

            # ── Pass 1: Extract facts from each document independently ──
            ref_facts = self._extract_facts(ref_text, role="REFERENCE")
            act_facts = self._extract_facts(act_text, role="ACTUAL")

            # ── Build focused compare input ──
            compare_input = (
                self._facts_to_text(ref_facts, "REFERENCE DOCUMENT")
                + "\n\n"
                + self._facts_to_text(act_facts, "ACTUAL DOCUMENT")
            )
            compare_input = compare_input[:MAX_COMPARE_CHARS]
            self.log.info("compare_documents: compare input built", chars=len(compare_input))

            # ── Pass 2: Compare the extracted facts ──
            self.log.info("compare_documents: calling LLM for comparison")
            raw_response = self.compare_chain.invoke({"combined_docs": compare_input})
            self.log.info("compare_documents: LLM responded",
                          response_length=len(raw_response), preview=raw_response[:300])

            rows = self._extract_rows(raw_response)
            self.log.info("compare_documents: completed", rows=len(rows))
            return self._format_response(rows)

        except DocumentPortalException:
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str:
                self.log.error("compare_documents: rate limit hit")
                raise
            self.log.error("compare_documents: error", error=str(e))
            raise DocumentPortalException("Error comparing documents", sys)

    def _split_combined(self, combined_docs: str) -> tuple[str, str]:
        """
        Split the combined_docs string back into reference and actual text.
        Falls back to 50/50 split if markers are missing.
        """
        ref_marker = "<<REFERENCE DOCUMENT>>"
        act_marker = "<<ACTUAL DOCUMENT>>"

        if ref_marker in combined_docs and act_marker in combined_docs:
            parts = combined_docs.split(act_marker, 1)
            ref_text = parts[0].replace(ref_marker, "").strip()
            act_text = parts[1].strip()
            return ref_text, act_text

        # Fallback: split at midpoint
        self.log.warning("_split_combined: document markers not found — using midpoint split")
        mid = len(combined_docs) // 2
        return combined_docs[:mid], combined_docs[mid:]

    def _format_response(self, rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["Section", "Changes"])
        df = pd.DataFrame(rows)
        for col in ["Section", "Changes"]:
            if col not in df.columns:
                df[col] = ""
        # Deduplicate rows
        df = df.drop_duplicates(subset=["Section", "Changes"])
        self.log.info("DataFrame built", shape=str(df.shape))
        return df
