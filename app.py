from fastapi import FastAPI, File, UploadFile
from main import Services
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specify your frontend URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# class UserQuery(BaseModel):
#     user_question:str
#     user_id:str


class PDFUploadRequest(BaseModel):
    file:UploadFile=File(...)

class UserQuery1(BaseModel):
    query:str

# class UserQueryPython(BaseModel):
#     query:str

# class CodeRequest(BaseModel):
#     code: str
    


# @app.post("/python")
# def chatbot(req:UserQueryPython):
#     ques=req.query
#     obj = Services()
#     result = obj.generate_python_code(ques)
#     return result
# obj = Services()
# @app.post("/resp")
# def chatbot(req:UserQuery):
#     ques=req.user_question
#     us=req.user_id
   
#     result = obj.chatbot_mainn(ques,us)
#     return result


@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    obj = Services()
    result = obj.upload(file)
    return result


@app.post("/query")
def query_documents(query:UserQuery1):
    obj=Services()
    result=obj.query_documents(query.query)
    return result

# @app.post("/execute")
# def execute_code(request: CodeRequest):
    
#     obj=Services()
#     result = obj.execute_python_code(request.code)
#     return {"result": result}
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)