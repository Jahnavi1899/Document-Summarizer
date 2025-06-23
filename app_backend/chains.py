# your_project/chains.py
import os
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from app_backend.llm_config import get_llm, get_embedding_model
from app_backend.db import get_vector_store, get_mongodb_client
from langchain.schema.retriever import BaseRetriever # For CustomRetriever
from langchain_core.documents import Document # For Document object in custom retriever
from langchain_mongodb import MongoDBAtlasVectorSearch

logger = logging.getLogger(__name__)

# Re-defining this here, it's ok as it's needed for the custom retriever.
def get_embedding_model_for_retriever():
    """Returns the embedding model, specifically for the custom retriever's use."""
    # from app_backend.llm_config import get_embedding_model # Import here to avoid circular if needed
    return get_embedding_model()

# Custom Retriever Class to filter by document_id (and potentially user_id later)
class CustomDocumentIdRetriever(BaseRetriever):
    vector_store_instance: MongoDBAtlasVectorSearch
    document_id_filter: str
    
    class Config: # Pydantic config for BaseModel, if you use it for validation
        arbitrary_types_allowed = True # Allows non-Pydantic types like MongoDBAtlasVectorSearch

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        return await self._get_relevant_documents(query)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        try:
            logger.info(f"Retriever called with query: '{query}' for document_id: {self.document_id_filter}")
            
            # 1. Convert input question to embedding using the same model
            embedding_model = get_embedding_model_for_retriever()
            query_embedding = embedding_model.embed_query(query)
            logger.info(f"Query converted to embedding with length: {len(query_embedding)}")
            
            # 2. Fetch top 5 closest embedding documents from DB for the given file
            client = get_mongodb_client()
            collection = client[os.getenv("DB_NAME")][os.getenv("EMBEDDINGS_COLLECTION_NAME")]
            
            # Use MongoDB Atlas Vector Search to find closest embeddings
            pipeline = [
                {
                    "$vectorSearch": {
                        "queryVector": query_embedding,
                        "path": "embedding",
                        "numCandidates": 100,
                        "limit": 5,
                        "index": os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME"),
                        "filter": {"document_id": self.document_id_filter}
                    }
                },
                {"$project": {"_id": 0, "text": 1, "document_id": 1, "source": 1, "page": 1}}
            ]
            
            results = list(collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} closest documents")
            
            # 3. Convert to LangChain Document objects
            documents = []
            for result in results:
                if 'text' in result and result['text']:
                    # Create metadata from the actual fields
                    metadata = {
                        "document_id": result.get("document_id"),
                        "source": result.get("source"),
                        "page": result.get("page")
                    }
                    
                    doc = Document(
                        page_content=result['text'],
                        metadata=metadata
                    )
                    documents.append(doc)
                    logger.info(f"Added document with content: {result['text'][:100]}...")
            
            logger.info(f"Returning {len(documents)} documents as context")
            return documents
            
        except Exception as e:
            logger.error(f"Error in custom retriever: {e}", exc_info=True)
            return []


def get_conversational_rag_chain(document_id: str): # Now accepts document_id
    """
    Creates a LangChain ConversationalRetrievalChain for RAG with memory, filtered by document_id.
    """
    llm = get_llm()
    vector_store = get_vector_store() # Get the pre-initialized vector store

    # Initialize memory (stores last 'k' interactions for a given chain instance)
    # This memory is per-request, NOT persistent across user sessions yet.
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True, 
        k=5,
        output_key="answer"  # Explicitly set which key to store from chain output
    )

    # Use our custom retriever for document_id filtering
    retriever = CustomDocumentIdRetriever(
        vector_store_instance=vector_store,
        document_id_filter=document_id
    )

    # This prompt is crucial for integrating chat history
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based ONLY on the provided context from a document. 

IMPORTANT RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say "I cannot answer this question based on the provided document."
3. Do not make up or infer information not explicitly stated in the context
4. Be precise and accurate in your responses
5. If the context is insufficient, acknowledge the limitations

Context: {context}

Chat History: {chat_history}

Question: {question}"""),
        ("human", "{question}"), # Aligned with ConversationalRetrievalChain's expected key
    ])
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever, # Use the custom retriever
        memory=memory, # Attach the memory here
        combine_docs_chain_kwargs={"prompt": qa_prompt}, # Use our custom prompt for QA
        return_source_documents=True # Optional: Returns the retrieved docs
    )
    return chain