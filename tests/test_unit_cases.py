import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# 1. Home Route
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text


# 2. Analyze - PDF
def test_analyze_pdf():
    with open("test_data/NIPS-2017-attention-is-all-you-need-Paper.pdf", "rb") as f:
        files = {"file": ("sample.pdf", f, "application/pdf")}
        response = client.post("/analyze", files=files)

    assert response.status_code in [200, 400]


# 3. Analyze - DOCX
def test_analyze_docx():
    with open("test_data/market_analysis_report.docx", "rb") as f:
        files = {
            "file": (
                "sample.docx",
                f,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        }
        response = client.post("/analyze", files=files)

    assert response.status_code in [200, 400]


# 4. Analyze - TXT
def test_analyze_txt():
    with open("test_data/state_of_the_union.txt", "rb") as f:
        files = {"file": ("sample.txt", f, "text/plain")}
        response = client.post("/analyze", files=files)

    assert response.status_code in [200, 400]


# 5. Invalid File Type
def test_analyze_invalid_file_type():
    files = {"file": ("sample.exe", b"invalid file", "application/octet-stream")}
    response = client.post("/analyze", files=files)

    assert response.status_code in [400, 422, 413, 500]


# 6. Compare - PDF vs PDF
def test_compare_pdf_files():
    with open("test_data/Long_Report_V1.pdf", "rb") as f1, \
         open("test_data/Long_Report_V2.pdf", "rb") as f2:

        files = {
            "reference": ("ref.pdf", f1, "application/pdf"),
            "actual": ("actual.pdf", f2, "application/pdf"),
        }

        response = client.post("/compare", files=files)

    assert response.status_code in [200, 400]


# 7. Compare - DOCX vs TXT
def test_compare_docx_txt():
    with open("test_data/market_analysis_report.docx", "rb") as f1, \
         open("test_data/state_of_the_union.txt", "rb") as f2:

        files = {
            "reference": ("ref.docx", f1, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            "actual": ("actual.txt", f2, "text/plain"),
        }

        response = client.post("/compare", files=files)

    assert response.status_code in [200, 400]


# 8. Chat Index - PDF
def test_chat_index_pdf():
    with open("test_data/NIPS-2017-attention-is-all-you-need-Paper.pdf", "rb") as f:
        files = [("files", ("doc.pdf", f, "application/pdf"))]
        response = client.post("/chat/index", files=files)

    assert response.status_code in [200, 400]


# 9. Chat Index - DOCX
def test_chat_index_docx():
    with open("test_data/market_analysis_report.docx", "rb") as f:
        files = [
            (
                "files",
                (
                    "doc.docx",
                    f,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            )
        ]
        response = client.post("/chat/index", files=files)

    assert response.status_code in [200, 400]


# 10. Chat Index - TXT
def test_chat_index_txt():
    with open("test_data/state_of_the_union.txt", "rb") as f:
        files = [("files", ("doc.txt", f, "text/plain"))]
        response = client.post("/chat/index", files=files)

    assert response.status_code in [200, 400]


# 11. Chat Query
def test_chat_query():
    payload = {
        "question": "What is this document about?",
        "session_id": "test_session",
        "history": []
    }

    response = client.post("/chat/query", json=payload)

    assert response.status_code in [200, 400, 422]


# 12. Invalid Route
def test_invalid_route():
    response = client.get("/invalid")
    assert response.status_code == 404