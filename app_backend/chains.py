# your_project/chains.py
import os
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from app_backend.llm_config import get_llm, get_embedding_model
from app_backend.db import get_vector_store
from langchain.schema.retriever import BaseRetriever # For CustomRetriever
from langchain_core.documents import Document # For Document object in custom retriever
from pymongo import MongoClient # For direct mongo access in custom retriever
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
        client = None
        try:
            client = MongoClient(os.getenv("MONGODB_ATLAS_URI")) # New client for this call
            # Use the global embedding model or create a new one here if necessary
            embedding_model = get_embedding_model_for_retriever()
            query_embedding = embedding_model.embed_query(query)
            
            collection = client[os.getenv("DB_NAME")][os.getenv("MONGODB_COLLECTION_NAME")]
            
            # The Atlas Vector Search aggregation pipeline
            results = collection.aggregate([
                {
                    "$vectorSearch": {
                        "queryVector": query_embedding,
                        "path": "embedding",
                        "numCandidates": 100, # Number of nearest neighbors to consider
                        "limit": 5,           # Number of results to return
                        "index": os.getenv("MONGODB_VECTOR_INDEX_NAME"),
                        "filter": { "document_id": self.document_id_filter } # Apply the filter here
                    }
                },
                { "$project": { "_id": 0, "page_content": "$text", "metadata": "$metadata" } } # Project for LangChain Document format
            ])
            
            # Convert MongoDB results to LangChain Document objects
            return [Document(page_content=r['page_content'], metadata=r.get('metadata', {})) for r in results]
        except Exception as e:
            logger.error(f"Error in custom retriever: {e}", exc_info=True)
            return [] # Return empty list on error
        finally:
            if client:
                client.close()


def get_conversational_rag_chain(document_id: str): # Now accepts document_id
    """
    Creates a LangChain ConversationalRetrievalChain for RAG with memory, filtered by document_id.
    """
    llm = get_llm()
    vector_store = get_vector_store() # Get the pre-initialized vector store

    # Initialize memory (stores last 'k' interactions for a given chain instance)
    # This memory is per-request, NOT persistent across user sessions yet.
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

    # Use our custom retriever for document_id filtering
    retriever = CustomDocumentIdRetriever(
        vector_store_instance=vector_store,
        document_id_filter=document_id
    )

    # This prompt is crucial for integrating chat history
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based ONLY on the following context and the conversation history. If the answer is not found in the context, clearly state 'I cannot answer this question based on the provided document(s).'. Do not make up answers.\n\nChat History:\n{chat_history}\n\nContext:\n{context}"),
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