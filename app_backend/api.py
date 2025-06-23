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

class DebugRetrieverInput(BaseModel):
    query: str

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
                'overall_status': 'processing',
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

        logger.info(f"Starting chat for document {request.document_id}, question: {request.question}")
        
        # Initialize the conversational chain
        # get_conversational_rag_chain will handle its own LLM/DB client setup
        conversational_rag_chain = get_conversational_rag_chain(document_id=request.document_id)
        
        logger.info(f"Chain initialized, invoking with question: {request.question}")
        response = conversational_rag_chain.invoke({"question": request.question})
        
        logger.info(f"Chat response generated for document {request.document_id}")
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

@app.get("/documents")
async def get_documents():
    """Get list of all uploaded documents with their processing status"""
    api_mongo_client = None
    try:
        api_mongo_client = get_mongodb_client()
        task_status_collection = api_mongo_client[os.getenv("DB_NAME")]['task_status']
        
        # Get all documents, sorted by creation date (newest first)
        documents = list(task_status_collection.find(
            {},
            {
                '_id': 1,
                'document_id': 1,
                'filename': 1,
                'overall_status': 1,
                'summary_status': 1,
                'rag_status': 1,
                'summary_text': 1,
                'rag_ready': 1,
                'created_at': 1,
                'last_updated_at': 1
            }
        ).sort('created_at', -1))
        
        # Prepare response data
        documents_list = []
        for doc in documents:
            documents_list.append({
                'task_id': str(doc['_id']),
                'document_id': str(doc.get('document_id')),
                'filename': doc.get('filename'),
                'overall_status': doc.get('overall_status'),
                'summary_status': doc.get('summary_status'),
                'rag_status': doc.get('rag_status'),
                'summary_text': doc.get('summary_text'),
                'rag_ready': doc.get('rag_ready', False),
                'created_at': doc.get('created_at').isoformat() if doc.get('created_at') else None,
                'last_updated_at': doc.get('last_updated_at').isoformat() if doc.get('last_updated_at') else None,
            })
        
        return {"documents": documents_list}
        
    except Exception as e:
        logger.error(f"Error fetching documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching documents.")
    finally:
        pass

@app.delete("/documents/{task_id}")
async def delete_document(task_id: str):
    """Delete a document and all its associated data from the database"""
    api_mongo_client = None
    try:
        api_mongo_client = get_mongodb_client()
        task_status_collection = api_mongo_client[os.getenv("DB_NAME")]['task_status']
        embeddings_collection = api_mongo_client[os.getenv("DB_NAME")][os.getenv("EMBEDDINGS_COLLECTION_NAME")]
        
        # First, get the document info to find the document_id
        document_doc = task_status_collection.find_one({'_id': task_id})
        if not document_doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        
        document_id = document_doc.get('document_id')
        filename = document_doc.get('filename')
        
        # Debug: Check the actual structure of embeddings for this document
        sample_embedding = embeddings_collection.find_one({"document_id": document_id})
        if sample_embedding:
            logger.info(f"Sample embedding structure for document_id {document_id}: {list(sample_embedding.keys())}")
            if 'metadata' in sample_embedding:
                logger.info(f"Metadata structure: {sample_embedding['metadata']}")
        
        # Delete from task_status collection
        task_result = task_status_collection.delete_one({'_id': task_id})
        
        # Try different deletion strategies for embeddings
        embeddings_result = None
        
        # Strategy 1: Try direct document_id field
        embeddings_result = embeddings_collection.delete_many({"document_id": document_id})
        if embeddings_result.deleted_count > 0:
            logger.info(f"Deleted {embeddings_result.deleted_count} embeddings using direct document_id field")
        else:
            # Strategy 2: Try metadata.document_id field
            embeddings_result = embeddings_collection.delete_many({"metadata.document_id": document_id})
            if embeddings_result.deleted_count > 0:
                logger.info(f"Deleted {embeddings_result.deleted_count} embeddings using metadata.document_id field")
            else:
                # Strategy 3: Try metadata field directly
                embeddings_result = embeddings_collection.delete_many({"metadata": {"document_id": document_id}})
                if embeddings_result.deleted_count > 0:
                    logger.info(f"Deleted {embeddings_result.deleted_count} embeddings using metadata object")
                else:
                    logger.warning(f"No embeddings found for document_id {document_id}")
                    embeddings_result = type('obj', (object,), {'deleted_count': 0})()
        
        # Delete the actual file if it exists
        try:
            file_path = os.path.join(TEMP_FILE_PATH, f"{task_id}_{filename}")
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete file {file_path}: {e}")
        
        logger.info(f"Deleted document {filename} (task_id: {task_id}, document_id: {document_id})")
        logger.info(f"Removed {embeddings_result.deleted_count} embeddings")
        
        return {
            "message": "Document deleted successfully",
            "deleted_task_id": task_id,
            "deleted_document_id": document_id,
            "deleted_filename": filename,
            "embeddings_removed": embeddings_result.deleted_count
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting document {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during document deletion.")
    finally:
        pass

@app.delete("/documents")
async def delete_all_documents():
    """Delete all documents and their associated data from the database"""
    api_mongo_client = None
    try:
        api_mongo_client = get_mongodb_client()
        task_status_collection = api_mongo_client[os.getenv("DB_NAME")]['task_status']
        embeddings_collection = api_mongo_client[os.getenv("DB_NAME")][os.getenv("EMBEDDINGS_COLLECTION_NAME")]
        
        # Get all documents to delete their files
        all_documents = list(task_status_collection.find({}, {'_id': 1, 'filename': 1}))
        
        # Delete all from task_status collection
        task_result = task_status_collection.delete_many({})
        
        # Delete all embeddings - try different strategies
        embeddings_result = None
        
        # Strategy 1: Try direct document_id field
        embeddings_result = embeddings_collection.delete_many({"document_id": {"$exists": True}})
        if embeddings_result.deleted_count > 0:
            logger.info(f"Deleted {embeddings_result.deleted_count} embeddings using direct document_id field")
        else:
            # Strategy 2: Try metadata.document_id field
            embeddings_result = embeddings_collection.delete_many({"metadata.document_id": {"$exists": True}})
            if embeddings_result.deleted_count > 0:
                logger.info(f"Deleted {embeddings_result.deleted_count} embeddings using metadata.document_id field")
            else:
                # Strategy 3: Delete all documents in the collection
                embeddings_result = embeddings_collection.delete_many({})
                logger.info(f"Deleted {embeddings_result.deleted_count} embeddings using delete all")
        
        # Delete all files in the uploads directory
        files_deleted = 0
        try:
            for filename in os.listdir(TEMP_FILE_PATH):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(TEMP_FILE_PATH, filename)
                    os.remove(file_path)
                    files_deleted += 1
        except Exception as e:
            logger.warning(f"Could not delete some files: {e}")
        
        logger.info(f"Deleted all documents: {task_result.deleted_count} task records, {embeddings_result.deleted_count} embeddings, {files_deleted} files")
        
        return {
            "message": "All documents deleted successfully",
            "documents_removed": task_result.deleted_count,
            "embeddings_removed": embeddings_result.deleted_count,
            "files_removed": files_deleted
        }
        
    except Exception as e:
        logger.error(f"Error deleting all documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during bulk deletion.")
    finally:
        pass

@app.get("/debug/embeddings/{document_id}")
async def debug_embeddings_structure(document_id: str):
    """Debug endpoint to check the structure of embeddings for a specific document"""
    api_mongo_client = None
    try:
        api_mongo_client = get_mongodb_client()
        embeddings_collection = api_mongo_client[os.getenv("DB_NAME")][os.getenv("EMBEDDINGS_COLLECTION_NAME")]
        
        # Find all embeddings for this document_id
        embeddings = list(embeddings_collection.find({"metadata.document_id": document_id}).limit(3))
        
        if not embeddings:
            # Try alternative queries
            embeddings_alt1 = list(embeddings_collection.find({"metadata.document_id": document_id}).limit(3))
            embeddings_alt2 = list(embeddings_collection.find({"metadata": {"document_id": document_id}}).limit(3))
            
            return {
                "document_id": document_id,
                "direct_query_count": len(embeddings),
                "metadata_dot_query_count": len(embeddings_alt1),
                "metadata_object_query_count": len(embeddings_alt2),
                "sample_embeddings": {
                    "direct": embeddings[0] if embeddings else None,
                    "metadata_dot": embeddings_alt1[0] if embeddings_alt1 else None,
                    "metadata_object": embeddings_alt2[0] if embeddings_alt2 else None
                }
            }
        
        # Return structure of first embedding
        sample_embedding = embeddings[0]
        return {
            "document_id": document_id,
            "total_embeddings_found": len(embeddings),
            "sample_embedding_keys": list(sample_embedding.keys()),
            "sample_embedding": {
                k: v for k, v in sample_embedding.items() 
                if k not in ['embedding']  # Exclude the large embedding vector
            }
        }
        
    except Exception as e:
        logger.error(f"Error debugging embeddings for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during debug.")
    finally:
        pass

@app.post("/debug/retriever/{document_id}")
async def debug_retriever(document_id: str, input_data: DebugRetrieverInput):
    """Debug endpoint to test the retriever functionality directly"""
    try:
        from app_backend.chains import CustomDocumentIdRetriever
        from app_backend.db import get_vector_store
        
        vector_store = get_vector_store()
        retriever = CustomDocumentIdRetriever(
            vector_store_instance=vector_store,
            document_id_filter=document_id
        )
        
        # Test the retriever
        documents = retriever._get_relevant_documents(input_data.query)
        
        return {
            "document_id": document_id,
            "query": input_data.query,
            "documents_retrieved": len(documents),
            "documents": [
                {
                    "page_content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        }
        
    except Exception as e:
        logger.error(f"Error testing retriever for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retriever test failed: {str(e)}")
