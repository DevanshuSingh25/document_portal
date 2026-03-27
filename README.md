# IntelliDoc AI

## 📖 About the Project
IntelliDoc AI is a full-stack, AI-powered document processing portal. It provides users with a comprehensive set of tools to upload documents, extract and chunk text, perform document analysis, and compare multiple documents using an advanced Retrieval-Augmented Generation (RAG) chatbot. 

## ✨ Features
- **Document Upload & Processing**: Seamlessly upload files and extract textual content.
- **Smart Chunking**: Automatically breaks down large documents into manageable segments.
- **Document Analysis & Comparison**: AI-driven insights to analyze and compare uploaded documents.
- **RAG Chatbot**: Interactive chatbot powered by LLaMA and a FAISS vector database to answer queries based on document context.
- **Automated CI/CD**: Fully automated pipeline via GitHub Actions.
- **Scalable Infrastructure**: Deployed securely on AWS ECS Fargate with Load Balancing.

## 🏗️ System Architecture

![System Architecture Diagram](architecture.png)

### Architecture Explanation
The architecture follows a clean, modern layered approach:
1. **Client Layer**: A web application serving as the primary interface for users to interact with the system.
2. **FastAPI Backend**: The core application server handling Document Uploads, Text Extraction & Chunking, Document Comparison, and the RAG Chatbot.
3. **LLM Integration**: Integrates with Groq API to utilize the ultra-fast LLaMA model for advanced conversational interactions and generated context.
4. **RAG Pipeline**: Responsible for Embedding Generation via HuggingFace's MiniLM and Similarity Search using FAISS Vector DB for accurate document retrieval.
5. **CI/CD Pipeline**: A seamless GitHub Actions workflow that handles Code Build & Test, Docker Image creation, and pushes updates straight to an AWS ECR.
6. **AWS Infrastructure**: The deployed service utilizes ECR, ECS Fargate, and an Application Load Balancer to guarantee high availability. Security is tightly managed using AWS Secrets Manager and IAM, with CloudWatch for logs.

## 🛠️ Tech Stack (High Level)
- **Backend Application**: Python, FastAPI
- **AI / ML / RAG**: LangChain, LLaMA (via Groq API), HuggingFace (MiniLM), FAISS Vector DB
- **Document Processing**: PyMuPDF, Unstructured, PyPDF
- **CI / CD Infrastructure**: GitHub Actions, Docker
- **Cloud / AWS**: ECS Fargate, ECR, Application Load Balancer, Secrets Manager, IAM, CloudWatch

## 🚀 How to Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/DevanshuSingh25/document_portal.git
cd Document_portal
```

2. **Create a Virtual Environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Set Up Environment Variables**
Create a `.env` file in the root directory based on the provided `.env.copy` and populate it with your specific API keys (e.g., Groq API, AWS credentials).

5. **Start up the application**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

6. **Access the application**
Open your browser and navigate to http://localhost:8080.