from __future__ import annotations
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, List

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.document_ops import SUPPORTED_EXTENSIONS

log = CustomLogger().get_logger(__name__)


def _session_id(prefix: str = "session") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """Save uploaded files and return their local paths."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for uf in uploaded_files:
            name = getattr(uf, "name", "file")
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=name)
                continue
            out = target_dir / f"{uuid.uuid4().hex[:8]}{ext}"
            with open(out, "wb") as f:
                f.write(uf.read() if hasattr(uf, "read") else uf.getbuffer())
            saved.append(out)
            log.info("File saved", name=name, path=str(out))
        return saved
    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e))
        raise DocumentPortalException("Failed to save uploaded files", e) from e