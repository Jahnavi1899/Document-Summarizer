import requests
from PyPDF2 import PdfReader
from PyPDF2 import errors
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

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

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
SUMMARIZATION_MODEL_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

PROMPT_TEMPLATE = ''' Text to summarizer: {text}
Please provide an abstractive summary of the above text. The summary should be concise, coherent, and capture the main points.
Summary:
'''

TEMP_FILE_PATH = "uploads"
EXTRACTED_TEXT_PATH = "extracted_text"
CLEAR_HISTORY = False
UPLOADED_FILE_NAME = ""
START_CONV_FLAG = False
qa_models = {}

class TextRequest(BaseModel):
    text:str

class QuestionAnswerRequest(BaseModel):
    question:str
    filename:str

def query(payload):
    try:
        response = requests.post(SUMMARIZATION_MODEL_API_URL, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        logger.error(f"Error processing the request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def convert_to_text(file_path, filename, save_dir):
    pdf_text = ""
    if filename.lower().endswith('.pdf'):
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

    os.makedirs(save_dir, exist_ok=True)
    text_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(save_dir, text_filename)

    # Write text to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pdf_text)
    return pdf_text

def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    text_chunks = text_splitter.split_text(text)
    #text_chunks = text_splitter.split_documents(docs)
    return text_chunks

# @app.post("/summarize_text")
def summarize_text(text_data, max_length):
    input_data = PROMPT_TEMPLATE.format(text = text_data)
    payload = {"inputs": input_data, "parameters":{"max_length": max_length, "min_length":50}}
    response = query(payload)
    #print(response)
    summarized_text = response[0]["summary_text"]
    return summarized_text

def create_embeddings(filename):
    loader = TextLoader(filename)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator=" ")
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)

    return embeddings, vectordb

def create_conversational_model(filename):
    filepath = EXTRACTED_TEXT_PATH + '/' + filename.split('.')[0] + '.txt'
    logger.info(filepath)

    embeddings, vectordb = create_embeddings(filepath)
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.8,
        top_k=50,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    retriever=vectordb.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )

    return qa

@app.get("/")
def display_text():
    return {"message":"Hello"}

@app.post("/upload_file")
async def upload_file(file: UploadFile):
    global CLEAR_HISTORY, START_CONV_FLAG, UPLOADED_FILE_NAME
    try:
        contents = await file.read()
        logger.info(f"Received file: {file.filename}")
        temp_file_path = output_path = os.path.join(TEMP_FILE_PATH, file.filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(contents)

        logger.info(f"size:{len(contents)}")
        text = convert_to_text(temp_file_path, file.filename, EXTRACTED_TEXT_PATH)

        # loader = PyPDFLoader(temp_file_path)
        # pages = loader.load()
       
        summarized_chunks = []
        document_chunks = create_chunks(text)
        for chunk in document_chunks:
            chunk_summary = summarize_text(chunk, 1000)
            summarized_chunks.append(chunk_summary)

        if len(summarized_chunks) > 1:
            combined_summary = " ".join(summarized_chunks)
            final_summary = summarize_text(combined_summary, 1000)

        UPLOADED_FILE_NAME = file.filename
        CLEAR_HISTORY = True
        START_CONV_FLAG = True
        return {"summary": final_summary}
    
    except Exception as e:
        # UPLOADED_FILE_NAME = ""
        CLEAR_HISTORY = False
        logger.error(f"Error processing file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/question-answer")
def answer_question(request: QuestionAnswerRequest):
    question = request.question
    filename = request.filename

    filepath = EXTRACTED_TEXT_PATH + '/' + filename.split('.')[0] + '.txt'
    logger.info(filepath)
    embeddings, vectordb = create_embeddings(filepath)
    #repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.8,
        top_k=50,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    template="""
    Use the following piece of context to answer the questions about their life.
    Use following piece of context to answer the question. 
    If you dont know the answer, just say you don't know.
    Keep the answer within 3 sentences and complete the answer.

    Context: {context}
    Question: {question}
    Answer:

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    rag_chain = (
        {"context":vectordb.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(question)
    logger.info(result)
    # logger.info(similar_docs)
   
    return {"answer":result}
    
@app.post("/chatbot")
def chatbot(request: QuestionAnswerRequest):
    question = request.question
    filename = request.filename
    try:
        if filename not in qa_models:
            qa = create_conversational_model(filename)
            qa_models[filename] = qa
        else:
            qa = qa_models[filename]

        result = qa.invoke(question)

        return {"answer": result['answer']}
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Connection aborted. Please try again.")

# @app.post("/clear-chat-history")
# def clear_chat_history
