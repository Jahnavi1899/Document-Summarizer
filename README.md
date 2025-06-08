# Document Summarizer and Q&A Web Application

A powerful web application that allows users to upload PDF documents, get AI-generated summaries, and interact with the content through a chatbot interface. The application leverages Ollama with the Llama 3.2 1b model for efficient and accurate document processing and Q&A capabilities.

## Features

1. **Document Upload & Processing**
   - Upload PDF documents
   - Automatic document text extraction

2. **AI-Powered Summarization**
   - Generate concise summaries of uploaded documents
   - Maintain key information and context
   - Fast and efficient processing using Llama 3.2 1b model

3. **Interactive Q&A**
   - Ask questions about the uploaded document
   - Get accurate, context-aware responses powered by Llama 3.2 1b
   - Natural language interaction

## Tech Stack

### Frontend
- React.js for the user interface
- Modern, responsive design
- Real-time updates and interactions

### Backend
- FastAPI for high-performance API endpoints
- Python-based processing pipeline
- File handling and storage

### AI/ML Components
- Ollama for local LLM deployment
- Llama 3.2 1b model for document processing and Q&A
- LangChain for document processing and integration
- Advanced NLP capabilities

## Project Structure

```
Document-Summarizer/
├── app-frontend/          # React frontend application
├── app-backend/           # FastAPI backend server
├── app-data/             # Data storage and processing
└── requirements.txt      # Python dependencies
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm or yarn
- Ollama installed on your system

### Ollama Setup
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Llama 3.2 1b model:
   ```bash
   ollama pull llama2:1b
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
   The server will run on `http://localhost:11434` by default

### Backend Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   cd app-backend
   uvicorn backend:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd app-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Usage

1. Ensure Ollama server is running (`ollama serve`)
2. Open the application in your web browser (default: http://localhost:3000)
3. Upload a PDF document using the file upload interface
4. Wait for the document to be processed and summarized
5. Use the chat interface to ask questions about the document

## Future Enhancements

1. User Authentication
   - Login and signup functionality
   - User profile management
   - Document history

2. Additional Features
   - Support for text files and web pages
   - Enhanced document validation
   - Batch processing capabilities
   - Support for different Ollama models

3. Performance Improvements
   - Caching mechanisms
   - Optimized document processing
   - Enhanced response times
   - Model optimization for faster inference

## References

1. DeepLearning.AI - LangChain Chat with Your Data
2. [Build a Simple RAG Chatbot with LangChain](https://medium.com/credera-engineering/build-a-simple-rag-chatbot-with-langchain-b96b233e1b2a)
3. [Ollama Documentation](https://github.com/ollama/ollama)
4. Claude
5. ChatGPT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
