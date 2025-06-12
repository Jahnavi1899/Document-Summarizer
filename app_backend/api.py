import uuid
import os
import logging
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
from dotenv import load_dotenv
import datetime

load_dotenv()

from celery_app import celery_app
from app_backend.tasks import process_summary_task, process_rag_task
from app_backend.db import get_mongodb_client
from app_backend.chains import get_conversational_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*", "http://localhost:3000", "localhost:3000"],#["http://localhost:3000", "localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

TEMP_FILE_PATH = "uploads"
os.makedirs(TEMP_FILE_PATH, exist_ok=True) # Ensure temp_uploads directory exists

class ChatbotInput(BaseModel):
    question: str
    document_id: str # Required for chat context

@app.get("/")
def display_text():
    return {"message":"Hello"}

# to upload the file
@app.post("/upload_file")
async def upload_file(file: UploadFile):
    # global CLEAR_HISTORY, START_CONV_FLAG, UPLOADED_FILE_NAME
    api_mongo_client = None
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        contents = await file.read()
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        temp_file_path = os.path.join(TEMP_FILE_PATH, unique_filename)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(contents)
        
        current_document_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        # Initialize status record in MongoDB
        api_mongo_client = get_mongodb_client()
        task_status_collection = api_mongo_client[os.getenv("DB_NAME")]['task_status']
        task_status_collection.insert_one(
            {
                '_id': task_id,
                'document_id': current_document_id,
                'filename': file.filename,
                'overall_status': 'uploaded',
                'summary_status': 'pending', # Status for summarization
                'rag_status': 'pending',     # Status for RAG processing
                'summary_text': None,
                'rag_ready': False,
                'error_message': None,
                'created_at': datetime.datetime.now(datetime.timezone.utc),
                'last_updated_at': datetime.datetime.now(datetime.timezone.utc)
            }
        )
        
        # --- DISPATCH BOTH TASKS CONCURRENTLY ---
        process_summary_task.delay(temp_file_path, current_document_id, task_id)
        process_rag_task.delay(temp_file_path, current_document_id, task_id)
        
        logger.info(f"File upload received for {file.filename}. Task {task_id} initiated. Summary and RAG tasks dispatched.")

        return JSONResponse(
            content={
                "message": "Document processing initiated.",
                "task_id": task_id,
                "document_id": current_document_id,
                "status_endpoint": f"/status/{task_id}"
            },
            status_code=202
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in upload_file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during file upload initiation.")
    # finally:
    #     if api_mongo_client:
    #         api_mongo_client.close()

@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    api_mongo_client = None
    try:
        api_mongo_client = get_mongodb_client()
        task_status_collection = api_mongo_client[os.getenv("DB_NAME")]['task_status']
        
        status_doc = task_status_collection.find_one({'_id': task_id})
        
        if not status_doc:
            raise HTTPException(status_code=404, detail="Task ID not found.")
        
        # Prepare for JSON serialization (remove _id if desired, ensure timestamps are strings)
        status_to_return = {
            'task_id': str(status_doc['_id']),
            'document_id': str(status_doc.get('document_id')),
            'overall_status': status_doc.get('overall_status'),
            'summary_status': status_doc.get('summary_status'),
            'rag_status': status_doc.get('rag_status'),
            'summary_text': status_doc.get('summary_text'),
            'rag_ready': status_doc.get('rag_ready', False),
            'error': status_doc.get('error_message'),
            'filename': status_doc.get('filename'),
            'created_at': status_doc.get('created_at').isoformat() if status_doc.get('created_at') else None,
            'last_updated_at': status_doc.get('last_updated_at').isoformat() if status_doc.get('last_updated_at') else None,
        }
        
        return status_to_return

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error checking task status for {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error checking task status.")
    # finally:
    #     if api_mongo_client:
    #         api_mongo_client.close()

@app.post("/chat")
async def chatbot(request: ChatbotInput):
    api_mongo_client = None
    try:
        if not request.document_id:
            raise HTTPException(status_code=400, detail="document_id is required for chat.")

        api_mongo_client = get_mongodb_client()
        task_status_collection = api_mongo_client[os.getenv("DB_NAME")]['task_status']
        
        # Check if RAG is ready for this document
        status_doc = task_status_collection.find_one({'document_id': request.document_id})
        
        if not status_doc:
            raise HTTPException(status_code=404, detail=f"Document with ID {request.document_id} not found or not processed.")
        
        if not status_doc.get('rag_ready'):
            raise HTTPException(status_code=400, detail=f"Document '{status_doc.get('filename')}' (ID: {request.document_id}) is not yet ready for Q&A. Status: {status_doc.get('rag_status')}")

        # Initialize the conversational chain
        # get_conversational_rag_chain will handle its own LLM/DB client setup
        conversational_rag_chain = get_conversational_rag_chain(document_id=request.document_id)
        
        response = conversational_rag_chain.invoke({"question": request.question})
        return {"answer": response["answer"]}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in chat for document {request.document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during chat.")
    # finally:
    #     if api_mongo_client:
    #         api_mongo_client.close()

# @app.post("/chat")
# def chatbot(request: ChatbotInput):
#     try:
#         chain = get_conversational_rag_chain()
#         response = chain.invoke({"question": request.question})
#         return {"answer": response["answer"]}
#     except Exception as e:
#         logger.error(f"Error in chat: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
