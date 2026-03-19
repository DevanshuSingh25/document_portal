from connectors.ppt_connector import load_ppt
from connectors.xlsx_connector import load_xlsx
from connectors.sql_connector import load_sqlite

__all__ = ["load_ppt", "load_xlsx", "load_sqlite"]
