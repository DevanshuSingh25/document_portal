import importlib.metadata
packages = [
    "langchain",
    "python-dotenv",
    "ipykernel",
    "langchain_groq",
    "sentence-transformers",
    "langchain_community",
    "pypdf",
    "faiss-cpu",
    "structlog",
    "PyMuPDF",
    "PyYAML",
    "pandas",
    "streamlit",
    "langchain-core[tracing]",
    "pytest",
    "docx2txt",
    "fastapi",
    "uvicorn",
    "python-multipart"

]
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} (not installed)")

# # serve static & templates
# app.mount("/static", StaticFiles(directory="../static"), name="static")
# templates = Jinja2Templates(directory="../templates")