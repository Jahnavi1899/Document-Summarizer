from fastapi import FastAPI, UploadFile, HTTPException, requests
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from backend import *

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

@app.get("/")
def display_text():
    return {"message":"Hello"}

# to upload the file
@app.post("/upload_file")
async def upload_file(file: UploadFile):
    global CLEAR_HISTORY, START_CONV_FLAG, UPLOADED_FILE_NAME
    try:
        contents = await file.read()
        temp_file_path = os.path.join(TEMP_FILE_PATH, file.filename)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(contents)

        UPLOADED_FILE_NAME = file.filename.split('.')[0]
        file_model_map.add_file_entry(UPLOADED_FILE_NAME)
        logger.info(f"In upload_file method:{file_model_map.list_files()}")


        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        chunks = create_chunks(docs)
        file_model_map.update_file_info(UPLOADED_FILE_NAME,'chunks',chunks)
        logger.info(f"In upload_file method after creating chunks:{file_model_map.list_files()}")

        # logger.info(f"The model mapping:{file_model_map.file_model_map}")
        final_summary = summarize_text(chunks) 
        logger.info(f"In upload_file method after summarizing text:{file_model_map.list_files()}")

        
        CLEAR_HISTORY = True
        START_CONV_FLAG = True
        return {"summary": final_summary}
    
    except Exception as e:
        # UPLOADED_FILE_NAME = ""
        CLEAR_HISTORY = False
        logger.error(f"Error processing file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/question-answer")
# def answer_question(request: QuestionAnswerRequest):
#     question = request.question
#     filename = request.filename

#     filepath = EXTRACTED_TEXT_PATH + '/' + filename.split('.')[0] + '.txt'
#     embeddings, vectordb = create_embeddings(filepath)
#     llm = file_model_map[UPLOADED_FILE_NAME]['model']

#     template="""
#     Use the following piece of context to answer the questions about their life.
#     Use following piece of context to answer the question. 
#     If you dont know the answer, just say you don't know.
#     Keep the answer within 3 sentences and complete the answer.

#     Context: {context}
#     Question: {question}
#     Answer:

#     """

#     prompt = PromptTemplate(
#         template=template,
#         input_variables=["context", "question"]
#     )

#     rag_chain = (
#         {"context":vectordb.as_retriever(), "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     result = rag_chain.invoke(question)
#     # logger.info(result)
#     # logger.info(similar_docs)
   
#     return {"answer":result}
    
@app.post("/chatbot")
def chatbot(request: QuestionAnswerRequest):
    question = request.question
    filename = request.filename
    try:
        fileName = filename.split('.')[0]
        logger.info(f"In chatbot method:{file_model_map.list_files()}")

        if file_model_map.get_file_info(fileName, 'qa') == None:
            qa, memory = create_conversational_model(filename)
            file_model_map.update_file_info(fileName, 'qa', qa)
            file_model_map.update_file_info(fileName, 'memory', memory)
        else:
            qa = file_model_map.get_file_info(fileName, 'qa')

        result = qa.invoke(question)

        return {"answer": result['answer']}
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Connection aborted. Please try again.")

# @app.post("/clear-chat-history")
# def clear_chat_history