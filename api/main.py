import os
import asyncio
import logging
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from logger.custom_logger import CustomLogger

log = CustomLogger().get_logger(__name__)

# Also capture LangChain / httpx / tenacity retry logs so they appear in the terminal
logging.basicConfig(level=logging.INFO)
for noisy in ("httpx", "openai", "langchain", "tenacity"):
    logging.getLogger(noisy).setLevel(logging.INFO)

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
    FaissManager,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

app = FastAPI(title="IntelliDoc AI API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────

def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg or "retrying" in msg


async def _run_with_cancel(request: Request, fn, *args, label: str = "task"):
    """
    Run a blocking callable `fn(*args)` in a thread pool via asyncio.to_thread().
    Polls request.is_disconnected() every 0.5 s.
    If the client disconnects (e.g. user clicked Stop), cancels the task and
    raises HTTPException(499) so the endpoint can return immediately.
    """
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    log.info(f"{label}: LLM task started")

    try:
        while not task.done():
            if await request.is_disconnected():
                log.warning(f"{label}: client disconnected — cancelling LLM task")
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                raise HTTPException(status_code=499, detail="Request cancelled by client")
            await asyncio.sleep(0.5)

        result = await task
        log.info(f"{label}: LLM task completed")
        return result

    except HTTPException:
        raise
    except Exception as e:
        if _is_rate_limit_error(e):
            log.warning(f"{label}: LLM rate limit hit (429)", error=str(e))
            raise HTTPException(status_code=429, detail=f"Rate limit reached: {e}")
        log.error(f"{label}: LLM task failed", error=str(e))
        raise


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    resp = templates.TemplateResponse(request=request, name="index.html", context={"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}


# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(request: Request, file: UploadFile = File(...)) -> Any:
    try:
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = _read_pdf_via_handler(dh, saved_path)
        log.info("analyze: document saved and read, calling LLM", chars=len(text))

        def _do_analyze():
            return DocumentAnalyzer().analyze_document(text)

        result = await _run_with_cancel(request, _do_analyze, label="analyze")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        if _is_rate_limit_error(e):
            log.warning("analyze: LLM rate limit hit (429)", error=str(e))
            raise HTTPException(status_code=429, detail=f"Rate limit reached: {e}")
        log.error("analyze: unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(
    request: Request,
    reference: UploadFile = File(...),
    actual: UploadFile = File(...),
) -> Any:
    try:
        dc = DocumentComparator()
        dc.save_uploaded_files(FastAPIFileAdapter(reference), FastAPIFileAdapter(actual))
        combined_text = dc.combine_documents()

        if combined_text == "__IDENTICAL_DOCUMENTS__":
            log.info("compare: documents are identical — skipping LLM")
            return {
                "rows": [],
                "session_id": dc.session_id,
                "message": "Documents are identical — no differences found.",
            }

        log.info("compare: documents differ, calling LLM", chars=len(combined_text))

        def _do_compare():
            comp = DocumentComparatorLLM()
            df = comp.compare_documents(combined_text)
            rename_map = {}
            for col in df.columns:
                if col.lower() in ("page", "section"):
                    rename_map[col] = "Section"
                elif col.lower() == "changes":
                    rename_map[col] = "Changes"
            if rename_map:
                df = df.rename(columns=rename_map)
            return df.to_dict(orient="records")

        rows = await _run_with_cancel(request, _do_compare, label="compare")
        return {"rows": rows, "session_id": dc.session_id}

    except HTTPException:
        raise
    except Exception as e:
        if _is_rate_limit_error(e):
            log.warning("compare: LLM rate limit hit (429)", error=str(e))
            raise HTTPException(status_code=429, detail=f"Rate limit reached: {e}")
        log.error("compare: unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")


# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        log.info("chat/index: files received, building FAISS index", file_count=len(wrapped))

        def _do_index():
            ci.built_retriver(wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k)

        await _run_with_cancel(request, _do_index, label="chat/index")
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}

    except HTTPException:
        raise
    except Exception as e:
        if _is_rate_limit_error(e):
            log.warning("chat/index: LLM rate limit hit (429)", error=str(e))
            raise HTTPException(status_code=429, detail=f"Rate limit reached: {e}")
        log.error("chat/index: unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    request: Request,
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        log.info("chat/query: question received, calling RAG", question=question[:80])

        def _do_query():
            rag = ConversationalRAG(session_id=session_id)
            rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)
            return rag.invoke(question, chat_history=[])

        response = await _run_with_cancel(request, _do_query, label="chat/query")
        return {"answer": response, "session_id": session_id, "k": k, "engine": "LCEL-RAG"}

    except HTTPException:
        raise
    except Exception as e:
        if _is_rate_limit_error(e):
            log.warning("chat/query: LLM rate limit hit (429)", error=str(e))
            raise HTTPException(status_code=429, detail=f"Rate limit reached: {e}")
        log.error("chat/query: unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ── Helpers ─────────────────────────────────────────────────────────

class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .read() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def read(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()
    def getbuffer(self) -> bytes:
        return self.read()

def _read_pdf_via_handler(handler: DocHandler, path: str) -> str:
    if hasattr(handler, "read_pdf"):
        return handler.read_pdf(path)  # type: ignore
    if hasattr(handler, "read_"):
        return handler.read_(path)  # type: ignore
    raise RuntimeError("DocHandler has neither read_pdf nor read_ method.")

# uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload