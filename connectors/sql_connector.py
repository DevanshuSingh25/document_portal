"""
SQL / SQLite Connector
-----------------------
Loads a .db or .sqlite file and reads all user-defined tables.
Each table is returned as a separate LangChain Document where the
content is the table name, column headers, and all rows formatted
as a readable text table.

Uses sqlalchemy for database connection (supports sqlite natively).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import sqlalchemy as sa
from langchain_core.documents import Document

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

log = CustomLogger().get_logger(__name__)


def _rows_to_text(table_name: str, columns: List[str], rows: list) -> str:
    """
    Format a list of rows into a readable text table.
    """
    header = " | ".join(columns)
    separator = " | ".join(["---"] * len(columns))
    lines = [f"## Table: {table_name}", header, separator]

    for row in rows:
        row_str = " | ".join(str(val) if val is not None else "NULL" for val in row)
        lines.append(row_str)

    return "\n".join(lines)


def load_sqlite(path: Path) -> List[Document]:
    """
    Load all tables from a .db or .sqlite file and return one Document per table.

    Args:
        path: Absolute path to the SQLite database file.

    Returns:
        List of Document objects, one per table.
    """
    log.info("SQL connector: starting load", file=str(path))
    connection_string = f"sqlite:///{str(path)}"
    engine = None

    try:
        engine = sa.create_engine(connection_string)
        inspector = sa.inspect(engine)
        table_names = inspector.get_table_names()

        log.info(
            "SQL connector: database opened",
            file=str(path),
            table_count=len(table_names),
            tables=table_names,
        )

        if not table_names:
            log.warning(
                "SQL connector: database has no tables",
                file=str(path),
            )
            return []

        docs: List[Document] = []

        with engine.connect() as conn:
            for table_name in table_names:
                log.info(
                    "SQL connector: reading table",
                    file=str(path),
                    table=table_name,
                )
                try:
                    result = conn.execute(sa.text(f"SELECT * FROM \"{table_name}\""))
                    columns = list(result.keys())
                    rows = result.fetchall()

                    log.info(
                        "SQL connector: table read complete",
                        table=table_name,
                        row_count=len(rows),
                        column_count=len(columns),
                        columns=columns,
                    )

                    if not rows:
                        log.warning(
                            "SQL connector: table is empty — skipping",
                            table=table_name,
                        )
                        continue

                    content = _rows_to_text(table_name, columns, rows)
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": str(path),
                                "file_type": path.suffix.lower(),
                                "table_name": table_name,
                                "row_count": len(rows),
                                "column_count": len(columns),
                            },
                        )
                    )

                except Exception as table_err:
                    log.error(
                        "SQL connector: failed to read table — skipping",
                        table=table_name,
                        error=str(table_err),
                    )
                    continue

        log.info(
            "SQL connector: load complete",
            file=str(path),
            documents_created=len(docs),
        )
        return docs

    except Exception as e:
        log.error("SQL connector: failed to connect to database", file=str(path), error=str(e))
        raise DocumentPortalException(f"Failed to load SQLite file: {path.name}", e) from e
    finally:
        if engine:
            engine.dispose()
            log.debug("SQL connector: engine disposed", file=str(path))
