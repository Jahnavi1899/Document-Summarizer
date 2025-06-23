# Document Summarizer and Q&A Web Application

A powerful web application that allows users to upload PDF documents, get AI-generated summaries, and interact with the content through a chatbot interface. The application leverages Ollama with the Llama 3.2 1b model for efficient and accurate document processing and Q&A capabilities.

## Features

1. **Document Upload & Processing**
   - Upload PDF documents with drag-and-drop interface
   - Automatic document text extraction and chunking
   - Real-time processing status updates

2. **AI-Powered Summarization**
   - Generate concise summaries of uploaded documents
   - Maintain key information and context
   - Fast and efficient processing using Llama 3.2 1b model

3. **Interactive Q&A with RAG**
   - Ask questions about the uploaded document
   - Get accurate, context-aware responses powered by Retrieval-Augmented Generation (RAG)
   - Natural language interaction with document context
   - Conversation memory for follow-up questions

4. **Background Processing**
   - Asynchronous document processing with Celery
   - Real-time status updates via API polling
   - Scalable worker architecture

## Tech Stack

### Frontend
- **React.js** with TypeScript for the user interface
- **Modern, responsive design** with CSS3
- **Real-time updates** and interactions
- **Nginx** for serving static files and API proxying

### Backend
- **FastAPI** for high-performance API endpoints
- **Python-based processing pipeline** with async support
- **File handling and storage** with proper validation
- **Background task processing** with Celery

### AI/ML Components
- **Ollama** for local LLM deployment
- **Llama 3.2 1B model** for document processing and Q&A
- **LangChain** for document processing and RAG integration
- **Vector embeddings** for semantic search
- **Conversational memory** for chat context

### Infrastructure
- **Docker** for containerization
- **Redis** for message broker and caching
- **MongoDB** (cloud service) for document and embedding storage
- **Celery** for background task processing

## Project Structure

```
Document-Summarizer/
├── app_frontend/          # React frontend application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API services
│   │   └── types.ts       # TypeScript definitions
│   └── package.json
├── app_backend/           # FastAPI backend server
│   ├── api.py            # Main API endpoints
│   ├── chains.py         # LangChain RAG chains
│   ├── tasks.py          # Celery background tasks
│   ├── db.py             # Database operations
│   └── llm_config.py     # LLM configuration
├── uploads/              # Temporary file storage
├── requirements.txt      # Python dependencies
├── Dockerfile            # Backend container
├── Dockerfile.worker     # Celery worker container
├── Dockerfile.frontend   # Frontend container
├── docker-compose.yml    # Service orchestration
└── nginx.conf           # Nginx configuration
```

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- Cloud MongoDB service (MongoDB Atlas, AWS DocumentDB, etc.)
- At least 8GB RAM (for Ollama model)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Document-Summarizer
   ```

2. **Create environment file:**

3. **Start all services:**
   ```bash
   docker compose up -d
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

### Environment Variables

Create a `.env` file with the following variables:

```bash
# MongoDB Configuration (Cloud Service)
MONGO_URI
DB_NAME

# MongoDB Collections
EMBEDDINGS_COLLECTION_NAME
TASK_STATUS_COLLECTION_NAME

# MongoDB Atlas Vector Search
ATLAS_VECTOR_SEARCH_INDEX_NAME

# Celery Configuration
CELERY_BROKER_URL
CELERY_RESULT_BACKEND

# Ollama Configuration
OLLAMA_BASE_URL
OLLAMA_MODEL

# Application Configuration
UPLOAD_DIR
MAX_FILE_SIZE
```

## Container Architecture

The application is containerized with separate services:

- **Frontend Container**: React app served by Nginx with API proxy
- **Backend Container**: FastAPI application
- **Worker Container**: Celery background task processor
- **Redis Container**: Message broker and result backend
- **Ollama Container**: LLM inference server
- **MongoDB**: Cloud service (external)

## Usage

1. **Upload a Document:**
   - Navigate to http://localhost:3000
   - Drag and drop or select a PDF file
   - Monitor processing status in real-time

2. **View Summary:**
   - Once processing is complete, view the AI-generated summary
   - Summary highlights key points and main topics

3. **Ask Questions:**
   - Use the chat interface to ask questions about the document
   - Get context-aware answers based on the document content
   - Ask follow-up questions for deeper insights

## API Endpoints

- `POST /upload_file` - Upload and process a document
- `GET /status/{task_id}` - Get processing status
- `POST /chat` - Ask questions about a document
- `GET /documents` - List all uploaded documents
- `DELETE /documents/{task_id}` - Delete a document

## Development Setup

### Local Development

1. **Backend Setup:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app_backend.api:app --reload
   ```

2. **Frontend Setup:**
   ```bash
   cd app_frontend
   npm install
   npm start
   ```

3. **Start Ollama:**
   ```bash
   ollama serve
   ollama pull llama3.2:1b
   ```

### Docker Development

```bash

docker compose up -d

# View logs
docker compose logs -f

# Scale workers
docker compose up -d --scale worker=3
```

## Monitoring and Management

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f worker
```

### Service Management
```bash
# Restart services
docker compose restart

# Stop all services
docker compose down

# Rebuild containers
docker compose build --no-cache
```

## Performance Optimization

- **Worker Scaling**: Scale Celery workers based on load
- **Model Selection**: Use smaller models for faster inference
- **Caching**: Redis provides task result caching
- **Vector Search**: Optimized embeddings for fast retrieval


## Future Enhancements

1. **User Management**
   - User authentication and authorization
   - Document sharing and collaboration
   - User-specific document history

2. **Enhanced Features**
   - Support for multiple file formats (DOCX, TXT, etc.)
   - Batch document processing
   - Advanced search and filtering
   - Export summaries and conversations

3. **Performance Improvements**
   - Model quantization for faster inference
   - Advanced caching strategies
   - Load balancing for high availability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
