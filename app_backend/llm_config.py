# your_project/llm_config.py
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# Use a global dictionary to store singletons, if preferred,
# or simply return new instances each time (which is fine for these clients)
_llm_instance = None
_embedding_model_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_MODEL"),
            temperature=0.3,
            num_ctx=131072, # Your 128K context window
        )
    return _llm_instance

def get_embedding_model():
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_EMBEDDING_MODEL")
        )
    return _embedding_model_instance