# tests/test_unit_cases.py

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# 1. Home route test
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text


# 2. Analyze PDF document
def test_analyze_pdf():
    files = {"file": ("sample.pdf", b"%PDF-1.4 dummy pdf content")}
    response = client.post("/analyze", files=files)
    assert response.status_code in [200, 400]


# 3. Analyze DOCX document
def test_analyze_docx():
    files = {"file": ("sample.docx", b"dummy docx content")}
    response = client.post("/analyze", files=files)
    assert response.status_code in [200, 400]


# 4. Analyze TXT document
def test_analyze_txt():
    files = {"file": ("sample.txt", b"This is a test text file")}
    response = client.post("/analyze", files=files)
    assert response.status_code in [200, 400]


# 5. Reject unsupported file type
def test_analyze_invalid_file_type():
    files = {"file": ("sample.exe", b"invalid file")}
    response = client.post("/analyze", files=files)
    assert response.status_code in [400, 422]


# 6. Compare two PDF files
def test_compare_pdf_files():
    files = {
        "reference_file": ("ref.pdf", b"%PDF-1.4 reference"),
        "actual_file": ("actual.pdf", b"%PDF-1.4 actual")
    }
    response = client.post("/compare", files=files)
    assert response.status_code in [200, 400]


# 7. Compare DOCX and TXT files
def test_compare_docx_txt():
    files = {
        "reference_file": ("ref.docx", b"docx content"),
        "actual_file": ("actual.txt", b"text content")
    }
    response = client.post("/compare", files=files)
    assert response.status_code in [200, 400]


# 8. Chat index PDF document
def test_chat_index_pdf():
    files = {"file": ("doc.pdf", b"%PDF-1.4 content")}
    response = client.post("/chat/index", files=files)
    assert response.status_code in [200, 400]


# 9. Chat index DOCX document
def test_chat_index_docx():
    files = {"file": ("doc.docx", b"docx test")}
    response = client.post("/chat/index", files=files)
    assert response.status_code in [200, 400]


# 10. Chat query with session
def test_chat_query():
    payload = {
        "question": "What is this document about?",
        "session_id": "test_session",
        "history": []
    }
    response = client.post("/chat/query", json=payload)
    assert response.status_code in [200, 400]


# 11. Invalid route
def test_invalid_route():
    response = client.get("/invalid")
    assert response.status_code == 404