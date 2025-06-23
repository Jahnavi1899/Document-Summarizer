# Docker Setup - Document Summarizer

Simple containerization for the Document Summarizer application with separate containers for different services and cloud MongoDB.

## Quick Start

1. **Create environment file:**
   ```bash
   cp env.example .env
   # Edit .env file with your cloud MongoDB connection string
   ```

2. **Build and start all services:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop services:**
   ```bash
   docker-compose down
   ```

## Services

- **Frontend** (Port 3000) - React application
- **Backend** (Port 8000) - FastAPI server
- **Worker** - Celery background tasks (separate container)
- **Redis** (Port 6379) - Message broker (separate container)
- **Ollama** (Port 11434) - LLM server
- **MongoDB** - Cloud service (external)

## Container Architecture

The application is split into separate containers for better scalability and resource management:

- **Backend Container**: Handles HTTP requests and API endpoints
- **Worker Container**: Processes background tasks (document processing, summarization)
- **Redis Container**: Message broker and result backend for Celery
- **Ollama Container**: LLM inference server
- **Frontend Container**: React application with Nginx
- **MongoDB**: Cloud service (MongoDB Atlas, AWS DocumentDB, etc.)

## Environment Variables

The application uses environment variables from a `.env` file. Copy `env.example` to `.env` and customize as needed:

- `MONGO_URI` - **Required**: Your cloud MongoDB connection string
- `DB_NAME` - Database name
- `OLLAMA_MODEL` - Ollama model to use (default: llama2:1b)
- `EMBEDDINGS_COLLECTION_NAME` - MongoDB collection for embeddings
- `ATLAS_VECTOR_SEARCH_INDEX_NAME` - Vector search index name

### MongoDB Connection String Examples:

**MongoDB Atlas:**
```
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/document_summarizer?retryWrites=true&w=majority
```

**AWS DocumentDB:**
```
MONGO_URI=mongodb://username:password@your-docdb-cluster.cluster-xxxxx.region.docdb.amazonaws.com:27017/document_summarizer
```

**Azure Cosmos DB:**
```
MONGO_URI=mongodb://username:password@your-cosmos-account.mongo.cosmos.azure.com:10255/document_summarizer?ssl=true&replicaSet=globaldb
```

## Files

- `Dockerfile` - Backend API container
- `Dockerfile.worker` - Celery worker container
- `Dockerfile.frontend` - Frontend container
- `docker-compose.yml` - Service orchestration
- `.dockerignore` - Excluded files
- `env.example` - Environment variables template

## Scaling

You can scale individual services independently:

```bash
# Scale worker instances
docker-compose up -d --scale worker=3

# Scale backend instances
docker-compose up -d --scale backend=2
```

## Notes

- Redis data is persisted in a Docker volume
- Ollama models are cached in a Docker volume
- Uploaded files are stored in the `./uploads` directory
- The specified Ollama model will be downloaded automatically on first run
- MongoDB data is stored in your cloud service 