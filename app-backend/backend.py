from PyPDF2 import PdfReader
from PyPDF2 import errors
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, SentenceTransformersTokenTextSplitter
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_models import FileModelManage
from db import *
from langchain_mongodb import MongoDBAtlasVectorSearch
from uuid import uuid4
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# repo_id = os.getenv("repo_id")
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# summarization model
summary_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summary_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
llm_repo_id = os.getenv("repo_id")

PROMPT = """ You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer
    the question. If you don't know the answer, say that you
    don't know. Use three sentences maximum and keep the
    answer concise.
    \n\n
    {context}"""

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

store = {}

TEMP_FILE_PATH = "uploads"
EXTRACTED_TEXT_PATH = "extracted_text"
CLEAR_HISTORY = False
UPLOADED_FILE_NAME = ""
START_CONV_FLAG = False
file_model_map = FileModelManage()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class ChatbotInput(BaseModel):
    question:str
    user:str
    filename:str

def LLM(repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.8,
        top_k=50,
        huggingfacehub_api_token=huggingface_api_token,
        # model_kwargs={'max_length': 32768}
    )

    return llm

def create_chunks(pages, max_tokens = 1024, overlap=50, tokenizer=summary_tokenizer):
    page_chunks = []
    for page in pages:
        tokens = tokenizer.encode(page, add_special_tokens=False)
        chunks = []
        start = 0
        # print(tokens)
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk = tokens[start:end]

            start = end - overlap if end < len(tokens) else len(tokens)
            chunks.append(chunk)
        # print(chunks)
        for chunk in chunks:
            page_chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))

    return page_chunks
        
    # max_chunk_size = 1024
    # overlap_size = 50
    # tokenized_docs = []
    # print("Before applying tokenizer")
    # for text in docs:
    #     tokenized_doc = tokenizer(text, return_tensors='pt', truncation=False, padding=True)
    #     tokenized_docs.append(tokenized_doc)
    # print("After applying tokenizer")
    # tokens = tokenized_docs['input_ids'][0].tolist()
    # chunks = []
    # current_chunk = []

    # for token in tokens:
    #     current_chunk.append(token)
    #     if len(current_chunk) >= max_chunk_size:
    #         chunks.append(current_chunk)
    #         # Keep the overlap by starting the new chunk from the last 'overlap_size' tokens
    #         current_chunk = current_chunk[-overlap_size:]

    # if current_chunk:
    #     chunks.append(current_chunk)

    # # Decode the tokenized chunks back into text
    # chunked_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    # return chunked_texts
    # # text_splitter = RecursiveCharacterTextSplitter(
    # #     chunk_size = 1000,
    # #     chunk_overlap = 200
    # # )
    # text_splitter = TokenTextSplitter(
    #     chunk_size = 400,
    #     chunk_overlap = 50
    # )
    # # text_chunks = text_splitter.split_text(docs)
    # text_chunks = text_splitter.split_documents(docs)
    # return text_chunks

def add_document_chunks_to_db(file_path):
    # load the document
    loader = PyPDFLoader(file_path)
    docs = loader.load() # each doc in docs stores the content of a page

    # split the entore document into chunks based on chunk size - this chunk size is based on embedding model context size
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=50)
    all_splits = text_splitter.split_documents(docs)

    vector_store_db = vector_store()

    uuids = [str(uuid4()) for _ in range(len(all_splits))]

    # adding each document embeddings into the vector database
    vector_store_db.add_documents(documents=all_splits, ids=uuids)

def vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = initDB()
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

    MONGODB_COLLECTION = db[COLLECTION_NAME]

    vector_store = MongoDBAtlasVectorSearch(
        collection=MONGODB_COLLECTION,
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine",
    )

    return vector_store

def summarize_text(text, tokenizer, model, max_summary_tokens=300):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    chunks_summary = model.generate(
        inputs["input_ids"],
        max_length=max_summary_tokens,
        min_length=50,
        num_beams=4,
    )

    summary = tokenizer.decode(chunks_summary[0], skip_special_tokens=True)
    return summary

    # llm = LLM(repo_id)

    # file_model_map.update_file_info(UPLOADED_FILE_NAME, 'model', llm)
    # # logger.info(f"In summarize_text method:{file_model_map.get_file_data(UPLOADED_FILE_NAME)}")

    # chunk_summary_prompt_template = """
    #     Please provide a summary of the following chunk of text that includes the main points and any important details.
    #     {text}
    # """

    # complete_summary_prompt_template = """
    #           Write a concise summary of the following text delimited by triple backquotes.
    #           Return your response which covers the key points of the text.
    #           ```{text}```
    #           SUMMARY:
    #           """
    
    # chunk_summary_prompt = PromptTemplate(
    #     template=chunk_summary_prompt_template, input_variables=["text"]
    # )

    # complete_summary_template = PromptTemplate(
    #     template=complete_summary_prompt_template, input_variables=["text"]
    # )

    # chain = load_summarize_chain(
    #     llm=llm,
    #     chain_type="map_reduce",
    #     map_prompt=chunk_summary_prompt,
    #     combine_prompt=complete_summary_template,
    #     return_intermediate_steps=True
    # )
    # result = chain.invoke({"input_documents": text_data_chunks}, return_only_outputs=True)

    # return result['output_text']
    inputs = summary_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    # Generate summary
    summary_ids = summary_model.generate(inputs['input_ids'], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    # Decode and return the summary
    return summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# def create_embeddings(filename):
   
#     logger.info(f"Inside create_embeddings: {file_model_map.list_files()}")
#     docs = file_model_map.get_file_info(filename, 'chunks')
#     # logger.info(docs)
#     embeddings = HuggingFaceEmbeddings()
#     vectordb = FAISS.from_documents(docs, embeddings)

#     return embeddings, vectordb

# def retrieve():
    
    # vector_db = vector_store()
    # retrieved_docs = vector_db.similarity_search(state["question"])
    # return {"context": retrieved_docs}

def generate():
    llm = LLM(llm_repo_id)
    retriever = vector_store().as_retriever()
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    
    
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    # docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # messages = prompt_template.invoke({"question": state["question"], "context": docs_content})
    
    # response = llm.invoke(messages)
    return rag_chain

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_conversational_chain():
    # graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    # graph_builder.add_edge(START, "retrieve")
    # graph = graph_builder.compile()

    # return graph
    rag_chain = generate()

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain
    
def conversational_chatbot(request):
    question = request.question
    session_id = request.user
    filename = request.filename

    result = create_conversational_chain().invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    # print("Inside conversational_chatbot")
    # print(request.question)
    # graph = create_conversational_chain()
    # result = graph.invoke({"question": request.question})

    # for step in graph.stream(
    #     {"question": request.question}, stream_mode="updates"
    # ):
    #     print(f"{step}\n\n----------------\n")

    return result["answer"]

