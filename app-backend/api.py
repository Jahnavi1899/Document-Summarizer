from fastapi import FastAPI, UploadFile, HTTPException, requests
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from backend import *
from db import *
import bson

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*", "http://localhost:3000", "localhost:3000"],#["http://localhost:3000", "localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def display_text():
    return {"message":"Hello"}

# to upload the file
@app.post("/upload_file")
async def upload_file(file: UploadFile):
    global CLEAR_HISTORY, START_CONV_FLAG, UPLOADED_FILE_NAME
    try:
        contents = await file.read() # read file contents
        temp_file_path = os.path.join(TEMP_FILE_PATH, file.filename)

        os.makedirs(TEMP_FILE_PATH, exist_ok=True) # make folder if path does not exist

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(contents)

        UPLOADED_FILE_NAME = file.filename.split('.pdf')[0]

        add_document_chunks_to_db(temp_file_path)
        insertDocument('User1', UPLOADED_FILE_NAME, bson.Binary(contents)) # insert the uploaded document into the db
    
        # loader = PyPDFLoader(temp_file_path)
        # docs = loader.load() # each doc in docs stores the content of a page
        # print("Len of docs:", len(docs))
    
        # page_content = [doc.page_content for doc in docs]
        # print(page_content)
        # create document chunks

    except Exception as e:
        # UPLOADED_FILE_NAME = ""
        CLEAR_HISTORY = False
        logger.error(f"Error processing file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chatbot(request: ChatbotInput):
    print("Inside chat api")
    question = request.question
    user = request.user
    filename = request.filename
    response = conversational_chatbot(request)

    return {"answer": response}
# @app.post("/upload_file")
# async def upload_file(file: UploadFile):
#     global CLEAR_HISTORY, START_CONV_FLAG, UPLOADED_FILE_NAME
#     try:
#         contents = await file.read() # read file contents
#         temp_file_path = os.path.join(TEMP_FILE_PATH, file.filename)

#         os.makedirs(TEMP_FILE_PATH, exist_ok=True) # make folder if path does not exist

#         with open(temp_file_path, "wb") as temp_file:
#             temp_file.write(contents)

#         UPLOADED_FILE_NAME = file.filename.split('.pdf')[0]
#         insertDocument('User1', UPLOADED_FILE_NAME, bson.Binary(contents)) # insert the uploaded document into the db
    
#         loader = PyPDFLoader(temp_file_path)
#         docs = loader.load() # each doc in docs stores the content of a page
#         print("Len of docs:", len(docs))
    
#         page_content = [doc.page_content for doc in docs]
#         # print(page_content)
#         document_chunks = create_chunks(page_content)
#         print(len(document_chunks))
#         # file_model_map.update_file_info(UPLOADED_FILE_NAME,'chunks',chunks)
#         logger.info(f"In upload_file method after creating chunks:{file_model_map.list_files()}")

#         # logger.info(f"The model mapping:{file_model_map.file_model_map}")
#         chunk_summary = [summarize_text(document_chunk, summary_tokenizer, summary_model) for document_chunk in document_chunks]
#         concatenated_summary = " ".join(chunk_summary)
#         # print(concatenated_summary)

#         final_summary = summarize_text(concatenated_summary, summary_tokenizer, summary_model, 1024)

#         # logger.info(f"In upload_file method after summarizing text:{file_model_map.list_files()}")
        
#         # CLEAR_HISTORY = True
#         # START_CONV_FLAG = True
#         return {"summary": final_summary}
    
#     except Exception as e:
#         # UPLOADED_FILE_NAME = ""
#         CLEAR_HISTORY = False
#         logger.error(f"Error processing file upload: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

