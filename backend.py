import requests
from PyPDF2 import PdfReader
from PyPDF2 import errors
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
MODEL_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class TextRequest(BaseModel):
    text:str

def query(payload):
    response = requests.post(MODEL_API_URL, headers=headers, json=payload)
    return response.json()

@app.get("/")
def display_text():
    return {"message":"Hello"}

@app.post("/summarize_text")
async def summarize_text(text_data: TextRequest):
    payload = {"inputs": text_data.text, "parameters":{"max_length": 500, "min_length":100}}
    response = query(payload)
    summarized_text = response[0]["summary_text"]
    return {"message":summarized_text}



# def read_document(file_path):
#     try:
#         with open(file_path, 'rb') as file:
#             pdf_reader = PdfReader(file)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         return text
#     except FileNotFoundError:
#         print("Please upload the file again")
#     except errors.PdfReadError as e:
#         print(f"Error reading PDF file: {e}")

# extracted_text = read_document("uploads/transformers.pdf")
