from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Dict, Any

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

log = CustomLogger().get_logger(__name__)

# ─────────────────────────────────────────────────────────────────
# Supported file extensions across the whole application.
# Add new extensions here and implement their loader below.
# ─────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".ppt",
    ".pptx",
    ".md",
    ".xlsx",
    ".csv",
    ".db",
    ".sqlite",
}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Load documents from a list of file paths.
    Routes each file to the appropriate loader based on its extension.
    Returns a flat list of LangChain Document objects.
    """
    docs: List[Document] = []
    paths = list(paths)
    log.info("load_documents: starting", file_count=len(paths))

    for p in paths:
        ext = p.suffix.lower()
        log.info("load_documents: processing file", file=str(p), extension=ext)

        try:
            if ext == ".pdf":
                log.debug("load_documents: using PyPDFLoader", file=str(p))
                loader = PyPDFLoader(str(p))
                loaded = loader.load()

            elif ext == ".docx":
                log.debug("load_documents: using Docx2txtLoader", file=str(p))
                loader = Docx2txtLoader(str(p))
                loaded = loader.load()

            elif ext == ".txt":
                log.debug("load_documents: using TextLoader", file=str(p))
                loader = TextLoader(str(p), encoding="utf-8")
                loaded = loader.load()

            elif ext == ".csv":
                log.debug("load_documents: using CSVLoader", file=str(p))
                loader = CSVLoader(str(p), encoding="utf-8")
                loaded = loader.load()

            elif ext == ".md":
                log.debug("load_documents: using UnstructuredMarkdownLoader", file=str(p))
                loader = UnstructuredMarkdownLoader(str(p))
                loaded = loader.load()

            elif ext in {".ppt", ".pptx"}:
                log.debug("load_documents: using PPT connector", file=str(p))
                from connectors.ppt_connector import load_ppt
                loaded = load_ppt(p)

            elif ext == ".xlsx":
                log.debug("load_documents: using XLSX connector", file=str(p))
                from connectors.xlsx_connector import load_xlsx
                loaded = load_xlsx(p)

            elif ext in {".db", ".sqlite"}:
                log.debug("load_documents: using SQL connector", file=str(p))
                from connectors.sql_connector import load_sqlite
                loaded = load_sqlite(p)

            else:
                log.warning(
                    "load_documents: unsupported extension — skipping",
                    file=str(p),
                    extension=ext,
                )
                continue

            log.info(
                "load_documents: file loaded successfully",
                file=str(p),
                documents_produced=len(loaded),
            )
            docs.extend(loaded)

        except DocumentPortalException:
            # Connector already logged the error — re-raise to halt the request
            raise
        except Exception as e:
            log.error(
                "load_documents: unexpected error loading file",
                file=str(p),
                error=str(e),
            )
            raise DocumentPortalException(f"Error loading {p.name}", e) from e

    log.info(
        "load_documents: all files processed",
        total_documents=len(docs),
        files_processed=len(paths),
    )
    return docs


def concat_for_analysis(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        parts.append(f"\n--- SOURCE: {src} ---\n{d.page_content}")
    return "\n".join(parts)

def concat_for_comparison(ref_docs: List[Document], act_docs: List[Document]) -> str:
    left = concat_for_analysis(ref_docs)
    right = concat_for_analysis(act_docs)
    return f"<<REFERENCE_DOCUMENTS>>\n{left}\n\n<<ACTUAL_DOCUMENTS>>\n{right}"