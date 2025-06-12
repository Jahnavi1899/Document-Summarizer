from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pymongo
from dotenv import load_dotenv
import os
import sys
from langchain_mongodb import MongoDBAtlasVectorSearch
from app_backend.llm_config import get_embedding_model

load_dotenv()

_mongo_client_instance = None
_vector_store_instance = {} # Store vector store instances per collection/index if needed

def get_mongodb_client():
    global _mongo_client_instance
    if _mongo_client_instance is None:
        try:
            username = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD") 
            # uri = f"mongodb+srv://{username}:{db_password}@cluster0.ouq4j.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
            uri = f"mongodb+srv://{username}:{db_password}@cluster0.1h6evjy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
            _mongo_client_instance = MongoClient(
                os.getenv("DB_URI"),
                tls=True,
                tlsAllowInvalidCertificates=True
            )
            print("Connection to db is done")
    
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise e
    return _mongo_client_instance

def get_vector_store():
    # In a real app, you might want to pass collection_name or index_name dynamically
    # For now, it uses env vars
    db_name = os.getenv("DB_NAME")
    collection_name = os.getenv("EMBEDDINGS_COLLECTION_NAME")
    index_name = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

    # Use a combined key for the instance cache
    instance_key = f"{db_name}-{collection_name}-{index_name}"
    
    global _vector_store_instance
    if instance_key not in _vector_store_instance:
        client = get_mongodb_client()
        collection = client[db_name][collection_name]
        _vector_store_instance[instance_key] = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=get_embedding_model(), # Use the centralized embedding model
            index_name=index_name
        )
    return _vector_store_instance[instance_key]

# # Create a new client and connect to the server
# def initDB():
#     try:
#         username = os.getenv("DB_USER")
#         db_password = os.getenv("DB_PASSWORD") 
#         uri = f"mongodb+srv://{username}:{db_password}@cluster0.ouq4j.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
#         client = MongoClient(
#             uri,
#             tls=True,
#             tlsAllowInvalidCertificates=True
#         )
#         DB_NAME = os.getenv("DB_NAME")
#         db = client[DB_NAME]
#         print("Connection to db is done")
#         return db
    
#     except Exception as e:
#         print(f"Error connecting to MongoDB: {e}")
#         raise e

# def insertDocument(username, filename, file):
#     db = initDB()
#     collection1 = db['files']

#     try:
#         doc = {'user': username, 'filename':filename, 'file': file}
#         result = collection1.insert_one(doc)
#     except Exception as e:
#         print(f"Error:{e}")
#         sys.exit(1)
#     else:
#         print("Inserted 1 document")

