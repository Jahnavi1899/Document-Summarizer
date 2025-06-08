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
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama.llms import OllamaLLM
#from langchain_huggingface import HuggingFaceEndpoint
# from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data_models import FileModelManage
from db import *
from langchain_mongodb import MongoDBAtlasVectorSearch
from uuid import uuid4
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# from langgraph.graph import START, StateGraph
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# repo_id = os.getenv("repo_id")
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# summarization model
# summary_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
# summary_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
# llm_repo_id = os.getenv("repo_id")

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
    # user:str
    # filename:str

db = initDB()

def LLM():
    llm = OllamaLLM(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("OLLAMA_MODEL"),
        temperature=0.8,
        num_ctx=131072
    )
    return llm

# summarization
def create_chunks(file_path):
    """
    Process a PDF file and create text chunks for summarization.
    Args:
        file_path: Path to the PDF file
    Returns:
        List of Document objects
    """
    try:
        # Load the PDF document
        loader = PyPDFLoader(file_path)
        docs = loader.load()  # each doc in docs stores the content of a page
  
        # Extract text from all pages
        all_text = ""
        for doc in docs:
            all_text += doc.page_content + "\n\n"
        
        # Calculate optimal chunk size based on model's context window
        # Using 1/3 of context window to leave room for prompts and responses
        CONTEXT_WINDOW = 131072 # Llama model context window
        CHUNK_SIZE = CONTEXT_WINDOW // 3  # ~1300 tokens
        CHUNK_OVERLAP = CHUNK_SIZE // 4  # 25% overlap for better context preservation
        
        # Split text into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Split on these characters in order
        )
        
        # Create chunks as Document objects
        chunks = text_splitter.create_documents([all_text])
        
        # Store chunks in file_model_map for later use
        file_model_map.update_file_info(UPLOADED_FILE_NAME, 'chunks', chunks)
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error in create_chunks: {e}")
        raise e

def summarize_text(text_data_chunks):
    """
    Summarize text using map-reduce approach with Llama model.
    Args:
        text_data_chunks: Text chunks to summarize (can be a single string or list of chunks)

    Returns:
        Summarized text
    """
    llm = LLM()
    file_model_map.update_file_info(UPLOADED_FILE_NAME, 'model', llm)

    chunk_summary_prompt_template = """
        Please provide a concise summary of the following chunk of text. Focus on extracting the main points and important details.
        
        Text:
        {text}
        
        Concise Summary:
    """

    complete_summary_prompt_template = """
        You are provided with several individual summaries from different sections of a larger document.
        Your task is to combine these individual summaries into a single, comprehensive, and coherent overall summary of the entire original document.
        Ensure the final summary flows naturally and accurately reflects the content of all provided summaries.
        Do NOT include any formatting like JSON, bullet points, or numbering. Just provide the complete summary as a continuous paragraph or set of paragraphs.
        
        Individual Summaries to Combine:
        ```{text}```
        
        Complete Document Summary:
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
        return_intermediate_steps=False,
        token_max=120000,
    )
    result = chain.invoke({"input_documents": text_data_chunks}, return_only_outputs=True)

    return result['output_text']

# chatbot
def get_embedding_model():
    """Initializes and returns the Ollama embedding model."""
    embedding_model = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        model=os.getenv("OLLAMA_EMBEDDING_MODEL")
    )
    return embedding_model

def get_vector_store():
    COLLECTION_NAME = os.getenv("EMBEDDINGS_COLLECTION_NAME")
    ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

    DB_COLLECTION = db[COLLECTION_NAME]

    vector_store = MongoDBAtlasVectorSearch(
        collection=DB_COLLECTION,
        embedding=get_embedding_model(),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    return vector_store

def create_rag_chunks(file_path: str, document_id: str):
    """
    Processes a PDF file into smaller, granular chunks suitable for RAG.
    Each chunk will be a LangChain Document with metadata including the document_id.

    Args:
        file_path: Path to the PDF file.
        document_id: A unique ID for the document (e.g., a hash or UUID).
    Returns:
        List of LangChain Document objects, ready for embedding.
    """
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load() # Each doc in docs stores the content of a page

        # Combine page content into a single string for recursive splitting
        all_text = ""
        for i, doc in enumerate(docs):
            all_text += doc.page_content + f"\n\n--- Page {i+1} ---\n\n" # Add page metadata for context

        # RAG Chunking: Smaller, more granular chunks
        # Aim for 500-1500 tokens per chunk for good retrieval precision
        # Use 10% overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Optimal for RAG retrieval
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Create chunks as Document objects, preserving source metadata
        # Add a custom 'document_id' to each chunk's metadata
        rag_chunks = text_splitter.create_documents(
            [all_text],
            metadatas=[{"document_id": document_id, "source": os.path.basename(file_path)}] * len(text_splitter.split_text(all_text))
            # The above line is a bit simplistic for metadata; you might want to adjust
            # if `docs` originally had page numbers, etc.
            # A more robust way: iterate through original docs, split each, and preserve original metadata.
            # Example:
            # rag_chunks = []
            # for i, doc_page in enumerate(docs):
            #    page_chunks = text_splitter.create_documents([doc_page.page_content], metadatas=[{"document_id": document_id, "source": doc_page.metadata.get('source', os.path.basename(file_path)), "page": doc_page.metadata.get('page')}])
            #    rag_chunks.extend(page_chunks)

        )
        logger.info(f"Created {len(rag_chunks)} RAG chunks for document '{document_id}'.")
        return rag_chunks
        
    except Exception as e:
        logger.error(f"Error in create_rag_chunks: {e}")
        raise e
    
def embed_and_store_chunks(chunks):
    """
    Generates embeddings for chunks and stores them in MongoDB Atlas Vector Search.
    
    Args:
        chunks: List of LangChain Document objects.
    """
    vector_store = get_vector_store()
    logger.info(f"Storing {len(chunks)} chunks in MongoDB Atlas Vector Search...")
    vector_store.add_documents(chunks) # This handles embedding and insertion
    logger.info("Chunks successfully embedded and stored.")

def get_conversational_rag_chain():
    """
    Creates a LangChain ConversationalRetrievalChain for RAG with memory.
    """
    # This prompt is crucial for integrating chat history
    # It tells the LLM to consider both chat history and retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based ONLY on the following context and the conversation history. If the answer is not found in the context, clearly state 'I cannot answer this question based on the provided document(s).'. Do not make up answers.\n\nChat History:\n{chat_history}\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])

    # The ConversationalRetrievalChain automatically manages passing history and context
    # It also has an internal 'question generator' that rewrites follow-up questions
    # to be standalone, using the chat history.
    
    # You might want to customize the question generator if needed
    # For now, let's use the default behavior.
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        output_key="answer"  # Specify which output to store in memory
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM(),
        retriever=get_vector_store().as_retriever(search_kwargs={"k": 5}),
        memory=memory, # Attach the memory here
        combine_docs_chain_kwargs={"prompt": qa_prompt}, # Use our custom prompt for QA
        return_source_documents=True # Optional: Returns the retrieved docs
    )
    return chain

# def add_document_chunks_to_db(file_path):
#     # load the document
#     loader = PyPDFLoader(file_path)
#     docs = loader.load() # each doc in docs stores the content of a page

#     # split the entire document into chunks based on chunk size - this chunk size is based on embedding model context size
#     text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=50)
#     all_splits = text_splitter.split_documents(docs)

#     vector_store_db = vector_store()

#     uuids = [str(uuid4()) for _ in range(len(all_splits))]

#     # adding each document embeddings into the vector database
#     vector_store_db.add_documents(documents=all_splits, ids=uuids)



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
    llm = LLM()
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

