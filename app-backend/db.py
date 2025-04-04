
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pymongo
from dotenv import load_dotenv
import os
import sys

load_dotenv()

# Create a new client and connect to the server
def initDB():
    try:
        username = os.getenv("db_username")
        db_password = os.getenv("db_password") 
        uri = f"mongodb+srv://{username}:{db_password}@cluster0.ouq4j.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        client = MongoClient(uri)
        DB_NAME = os.getenv("DB_NAME")
        db = client[DB_NAME]
        print("Connection to db is done")
        return db
    
    except pymongo.errors.ConfigurationError:
        print("Incorrect URI host received. Please check your URI string.")
        sys.exit(1)

def insertDocument(username, filename, file):
    db = initDB()
    collection1 = db['files']

    try:
        doc = {'user': username, 'filename':filename, 'file': file}
        result = collection1.insert_one(doc)
    except Exception as e:
        print(f"Error:{e}")
        sys.exit(1)
    else:
        print("Inserted 1 document")