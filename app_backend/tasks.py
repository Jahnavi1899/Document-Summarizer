# your_project/tasks.py
import os
import uuid
import logging
from celery import Celery # This needs to be imported, or from .celery_app import celery_app
from pymongo import MongoClient
import datetime
from dotenv import load_dotenv
load_dotenv()

print(f"DEBUG: OLLAMA_LLM_MODEL from tasks.py: '{os.getenv('OLLAMA_MODEL')}'")
print(f"DEBUG: OLLAMA_BASE_URL from tasks.py: '{os.getenv('OLLAMA_BASE_URL')}'")
print(f"DEBUG: MONGODB_DB_NAME from tasks.py: '{os.getenv('DB_NAME')}'") # Also check this one

# Assume celery_app is defined in celery_app.py at project root
from celery_app import celery_app 

# --- Import core LangChain components for tasks ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document # For Document object

# --- Import configs for worker-specific client initialization ---
from app_backend.llm_config import get_llm as _get_llm_worker
from app_backend.llm_config import get_embedding_model as _get_embedding_model_worker
from app_backend.db import get_mongodb_client as _get_mongodb_client_worker
from app_backend.db import get_vector_store as _get_vector_store_worker


logger = logging.getLogger(__name__)

# --- Helper to update task status in MongoDB ---
def _update_task_status(task_id: str, updates: dict):
    client = None
    try:
        client = _get_mongodb_client_worker()
        task_status_collection = client[os.getenv("DB_NAME")]['task_status']
        task_status_collection.update_one(
            {'_id': task_id},
            {'$set': {**updates, 'last_updated_at': datetime.datetime.utcnow()}},
            upsert=True
        )
    except Exception as e:
        logger.error(f"Failed to update task status for {task_id}: {e}", exc_info=True)
    finally:
        pass
        # if client:
        #     client.close()

# --- Celery Task Definitions ---

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60) # Retry up to 3 times with 60s delay
def process_summary_task(self, file_path: str, document_id: str, task_id: str):
    logger.info(f"Task {task_id}: Starting summary processing for document {document_id}")
    _update_task_status(task_id, {'summary_status': 'processing', 'overall_status': 'processing'})
    try:
        # 1. Chunk for summarization
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_text = ""
        for doc in docs:
            all_text += doc.page_content + "\n\n"
        
        CONTEXT_WINDOW = 131072
        CHUNK_SIZE = CONTEXT_WINDOW // 3
        CHUNK_OVERLAP = CHUNK_SIZE // 4
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks_for_summary = text_splitter.create_documents([all_text])

        # 2. Summarize text
        llm = _get_llm_worker() # Get LLM for worker
        chunk_summary_prompt_template = """Please provide a concise summary of the following chunk of text. 
        Focus on extracting the main points and important details. Text: {text}\nConcise Summary:"""

        complete_summary_prompt_template = """You are provided with several individual summaries from different sections of a larger document. 
        Your task is to combine these individual summaries into a single, comprehensive, and coherent overall summary of the entire original document. 
        Ensure the final summary flows naturally and accurately reflects the content of all provided summaries. 
        Do NOT include any formatting like JSON, bullet points, or numbering.
        Just provide the complete summary as a continuous paragraph or set of paragraphs. 
        Individual Summaries to Combine: ```{text}```\nComplete Document Summary:"""
        
        chunk_summary_prompt = PromptTemplate(template=chunk_summary_prompt_template, input_variables=["text"])
        complete_summary_template = PromptTemplate(template=complete_summary_prompt_template, input_variables=["text"])
        
        summarize_chain = load_summarize_chain(
            llm=llm, chain_type="map_reduce", map_prompt=chunk_summary_prompt,
            combine_prompt=complete_summary_template, return_intermediate_steps=False,
            token_max=120000
        )
        summary_result = summarize_chain.invoke({"input_documents": chunks_for_summary}, return_only_outputs=True)['output_text']
        
        _update_task_status(task_id, {'summary_status': 'completed', 'summary_text': summary_result})
        logger.info(f"Task {task_id}: Summary completed for document {document_id}.")

        # Chain to the RAG task immediately after summary (assuming it's sequential)
        # It's crucial that this task is called with .delay() to be run by Celery
        # process_rag_task.delay(file_path, document_id, task_id)

    except Exception as e:
        logger.error(f"Task {task_id}: Summary processing failed for document {document_id}: {e}", exc_info=True)
        _update_task_status(task_id, {'summary_status': 'failed', 'error_message': str(e), 'overall_status': 'failed'})
        raise self.retry(exc=e) # Re-raise for Celery retry mechanism
    finally:
        pass


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_rag_task(self, file_path: str, document_id: str, task_id: str):
    logger.info(f"Task {task_id}: Starting RAG processing for document {document_id}")
    _update_task_status(task_id, {'rag_status': 'processing', 'overall_status': 'processing'})
    client = None
    try:
        # 1. Create RAG chunks
        loader = PyPDFLoader(file_path)
        docs = loader.load() # Each doc in docs stores the content of a page

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        rag_chunks = []
        for i, page_doc in enumerate(docs):
            page_content = page_doc.page_content
            chunks_from_page = text_splitter.create_documents(
                [page_content],
                metadatas=[{
                    # No user_id for now as we're focusing on async processing first
                    "document_id": document_id,
                    "source": os.path.basename(file_path),
                    "page": page_doc.metadata.get('page', i)
                }]
            )
            rag_chunks.extend(chunks_from_page)

        # 2. Embed and Store in MongoDB
        client = _get_mongodb_client_worker()
        vector_store = _get_vector_store_worker()
        vector_store.add_documents(rag_chunks) # This handles embedding and insertion

        _update_task_status(task_id, {'rag_status': 'completed', 'rag_ready': True, 'overall_status': 'completed'})
        logger.info(f"Task {task_id}: RAG processing completed for document {document_id}.")

    except Exception as e:
        logger.error(f"Task {task_id}: RAG processing failed for document {document_id}: {e}", exc_info=True)
        _update_task_status(task_id, {'rag_status': 'failed', 'error_message': str(e), 'overall_status': 'failed'})
        raise self.retry(exc=e) # Re-raise for Celery retry mechanism
    finally:
        pass
        # if client:
        #     client.close()
        # Clean up temp file after all processing for it is done
        # if os.path.exists(file_path):
        #      os.remove(file_path)
        #      logger.info(f"Task {task_id}: Removed temporary file {file_path}")