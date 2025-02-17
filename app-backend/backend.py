from PyPDF2 import PdfReader
from PyPDF2 import errors
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer
from data_models import FileModelManage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

repo_id = os.getenv("repo_id")
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

PROMPT_TEMPLATE = ''' Text to summarizer: {text}
Please provide an abstractive summary of the above text. The summary should be concise, coherent, and capture the main points.
Summary:
'''

TEMP_FILE_PATH = "uploads"
EXTRACTED_TEXT_PATH = "extracted_text"
CLEAR_HISTORY = False
UPLOADED_FILE_NAME = ""
START_CONV_FLAG = False
file_model_map = FileModelManage()
tokenizer = AutoTokenizer.from_pretrained(repo_id)

class TextRequest(BaseModel):
    text:str

class QuestionAnswerRequest(BaseModel):
    question:str
    filename:str

# def query(payload):
#     try:
#         response = requests.post(SUMMARIZATION_MODEL_API_URL, headers=headers, json=payload)
#         return response.json()
#     except Exception as e:
#         logger.error(f"Error processing the request: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# def convert_to_text(file_path, filename, save_dir):
#     pdf_text = ""
#     if filename.lower().endswith('.pdf'):
#         pdf_reader = PdfReader(file_path)
#         for page in pdf_reader.pages:
#             pdf_text += page.extract_text()

#     os.makedirs(save_dir, exist_ok=True)
#     text_filename = os.path.splitext(filename)[0] + ".txt"
#     output_path = os.path.join(save_dir, text_filename)

#     # Write text to file
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(pdf_text)
#     return pdf_text

def create_chunks(docs):
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 1000,
    #     chunk_overlap = 200
    # )
    text_splitter = TokenTextSplitter(
        chunk_size = 400,
        chunk_overlap = 50
    )
    # text_chunks = text_splitter.split_text(docs)
    text_chunks = text_splitter.split_documents(docs)
    return text_chunks

def LLM(repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.8,
        top_k=50,
        huggingfacehub_api_token=huggingface_api_token,
        # model_kwargs={'max_length': 32768}
    )

    return llm

# @app.post("/summarize_text")
def summarize_text(text_data_chunks):
    llm = LLM(repo_id)

    file_model_map.update_file_info(UPLOADED_FILE_NAME, 'model', llm)
    # logger.info(f"In summarize_text method:{file_model_map.get_file_data(UPLOADED_FILE_NAME)}")

    chunk_summary_prompt_template = """
        Please provide a summary of the following chunk of text that includes the main points and any important details.
        {text}
    """

    complete_summary_prompt_template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response which covers the key points of the text.
              ```{text}```
              SUMMARY:
              """
    
    chunk_summary_prompt = PromptTemplate(
        template=chunk_summary_prompt_template, input_variables=["text"]
    )

    complete_summary_template = PromptTemplate(
        template=complete_summary_prompt_template, input_variables=["text"]
    )

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=chunk_summary_prompt,
        combine_prompt=complete_summary_template,
        return_intermediate_steps=True
    )
    result = chain.invoke({"input_documents": text_data_chunks}, return_only_outputs=True)

    return result['output_text']

def create_embeddings(filename):
   
    logger.info(f"Inside create_embeddings: {file_model_map.list_files()}")
    docs = file_model_map.get_file_info(filename, 'chunks')
    # logger.info(docs)
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)

    return embeddings, vectordb

def create_conversational_model(filename):
    logger.info(f"Inside create_converstaion_model: {file_model_map.list_files()}")
    embeddings, vectordb = create_embeddings(UPLOADED_FILE_NAME)
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.8,
        top_k=50,
        huggingfacehub_api_token=huggingface_api_token
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
    logger.info(f"Inside create_converstaion_model - after: {file_model_map.list_files()}")

    return qa, memory


