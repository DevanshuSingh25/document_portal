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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

from utils.file_io import _session_id, save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concat_for_comparison

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt",
    ".ppt", ".pptx", ".md",
    ".xlsx", ".csv",
    ".db", ".sqlite",
}

# FAISS Manager (load-or-create)
class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}
        
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}
        

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None
        
    def _exists(self)-> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()
    
    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        
        
    def add_documents(self,docs: List[Document]):
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")
        
        new_docs: List[Document] = []
        
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)
            
        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)
    
    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs
        
        
class ChatIngestor:
    def __init__( self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.model_loader = ModelLoader()
            
            self.use_session = use_session_dirs
            self.session_id = session_id or _session_id()
            
            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)
            
            self.log.info("ChatIngestor initialized",
                          session_id=self.session_id,
                          temp_dir=str(self.temp_dir),
                          faiss_dir=str(self.faiss_dir),
                          sessionized=self.use_session)
        except Exception as e:
            self.log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e
            
        
    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base
        
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        self.log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks
    
    def built_retriver( self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            fm = FaissManager(self.faiss_dir, self.model_loader)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
                
            added = fm.add_documents(chunks)
            self.log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        except Exception as e:
            self.log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e

            
        
            
class DocHandler:
    """
    Document save + read (page-wise) for analysis.
    Supports all extensions listed in SUPPORTED_EXTENSIONS.
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.log = CustomLogger().get_logger(__name__)
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or _session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        self.log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_pdf(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Invalid file type '{ext}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            self.log.info("Document saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            self.log.error("Failed to save document", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save document: {str(e)}", e) from e

    def read_pdf(self, pdf_path: str) -> str:
        try:
            file_path = Path(pdf_path)
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                text_chunks = []
                with fitz.open(pdf_path) as doc:
                    for page_num in range(doc.page_count):
                        page = doc.load_page(page_num)
                        text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")  # type: ignore
                text = "\n".join(text_chunks)
                self.log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
                return text
            elif ext in SUPPORTED_EXTENSIONS:
                docs = load_documents([file_path])
                text = "\n".join(d.page_content for d in docs)
                self.log.info("Document read successfully", path=pdf_path, session_id=self.session_id)
                return text
            else:
                raise ValueError(
                    f"Unsupported file type '{ext}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
        except Exception as e:
            self.log.error("Failed to read document", error=str(e), path=pdf_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process document: {pdf_path}", e) from e
class DocumentComparator:
    """
    Save, read & combine PDFs for comparison with session-based versioning.
    """
    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.session_id = session_id or _session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.log.info("DocumentComparator initialized", session_path=str(self.session_path))

    def save_uploaded_files(self, reference_file, actual_file):
        try:
            ref_name = Path(reference_file.name).name
            act_name = Path(actual_file.name).name

            # ── Prefix with reference_/actual_ so same-name files never overwrite ──
            ref_path = self.session_path / f"reference_{ref_name}"
            act_path = self.session_path / f"actual_{act_name}"

            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                orig_name = Path(fobj.name).name
                ext = Path(orig_name).suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    raise ValueError(
                        f"Invalid file type '{ext}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                    )
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())

            self.log.info(
                "Files saved with role prefixes",
                reference=str(ref_path),
                actual=str(act_path),
                session=self.session_id,
            )
            return ref_path, act_path
        except Exception as e:
            self.log.error("Error saving files", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error saving files", e) from e

    def read_pdf(self, pdf_path: Path) -> str:
        try:
            ext = pdf_path.suffix.lower()
            if ext == ".pdf":
                with fitz.open(pdf_path) as doc:
                    if doc.is_encrypted:
                        raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                    parts = []
                    for page_num in range(doc.page_count):
                        page = doc.load_page(page_num)
                        text = page.get_text()  # type: ignore
                        if text.strip():
                            parts.append(f"\n --- Page {page_num + 1} --- \n{text}")
                self.log.info("PDF read successfully", file=str(pdf_path), pages=len(parts))
                return "\n".join(parts)
            elif ext in SUPPORTED_EXTENSIONS:
                docs = load_documents([pdf_path])
                text = "\n".join(d.page_content for d in docs)
                self.log.info("Document read successfully", file=str(pdf_path))
                return text
            else:
                raise ValueError(
                    f"Unsupported file type '{ext}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
        except Exception as e:
            self.log.error("Error reading document", file=str(pdf_path), error=str(e))
            raise DocumentPortalException("Error reading document", e) from e

    def combine_documents(self) -> str:
        """
        Build the combined text that is sent to the comparator LLM.
        - Looks for files prefixed with 'reference_' and 'actual_' in the session folder.
        - Clearly labels each section so the LLM always knows which is which.
        - If both documents are byte-for-byte identical, returns a sentinel string
          that bypasses the LLM entirely and returns "No differences found."
        """
        import hashlib

        try:
            # Find the reference and actual files in this session folder
            all_files = list(self.session_path.iterdir())
            ref_files = [f for f in all_files if f.is_file() and f.name.startswith("reference_")]
            act_files = [f for f in all_files if f.is_file() and f.name.startswith("actual_")]

            if not ref_files or not act_files:
                # Fallback: treat first file as reference, second as actual (sorted order)
                valid = sorted(
                    f for f in all_files
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
                )
                if len(valid) < 2:
                    raise ValueError(
                        "Session folder must contain exactly 2 documents to compare."
                    )
                ref_files = [valid[0]]
                act_files = [valid[1]]
                self.log.warning(
                    "combine_documents: no role-prefixed files found — using sorted fallback",
                    reference=str(ref_files[0]),
                    actual=str(act_files[0]),
                )

            ref_path = ref_files[0]
            act_path = act_files[0]

            self.log.info(
                "combine_documents: reading files",
                reference=ref_path.name,
                actual=act_path.name,
            )

            ref_text = self.read_pdf(ref_path)
            act_text = self.read_pdf(act_path)

            # ── Identical-content guard ──────────────────────────────────────
            # MD5 of normalised whitespace to catch trivially identical uploads
            ref_hash = hashlib.md5(ref_text.strip().encode()).hexdigest()
            act_hash = hashlib.md5(act_text.strip().encode()).hexdigest()

            if ref_hash == act_hash:
                self.log.info(
                    "combine_documents: both documents are identical (hash match) "
                    "— returning sentinel",
                    reference=ref_path.name,
                    actual=act_path.name,
                )
                return "__IDENTICAL_DOCUMENTS__"
            # ────────────────────────────────────────────────────────────────

            combined_text = (
                "<<REFERENCE DOCUMENT>>\n"
                f"Filename: {ref_path.name.replace('reference_', '')}\n\n"
                f"{ref_text}\n\n"
                "<<ACTUAL DOCUMENT>>\n"
                f"Filename: {act_path.name.replace('actual_', '')}\n\n"
                f"{act_text}"
            )

            self.log.info(
                "combine_documents: documents combined",
                ref_chars=len(ref_text),
                act_chars=len(act_text),
                combined_chars=len(combined_text),
                session=self.session_id,
            )
            return combined_text

        except DocumentPortalException:
            raise
        except Exception as e:
            self.log.error("combine_documents: error", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", e) from e


    def clean_old_sessions(self, keep_latest: int = 3):
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                self.log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            self.log.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("Error cleaning old sessions", e) from e