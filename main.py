import io
import sys
import tempfile
from typing import Dict
import chromadb
from fastapi import File, HTTPException, UploadFile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document 
import os
from dotenv import load_dotenv
from pypdf import PdfReader

class Services:
    def __init__(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"]=os.getenv("OPEN_API_KEY")
        # os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
        # os.environ["LANGCHAIN_TRACING_V2"]="true"
        # os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
        self.llm=ChatOpenAI(model="gpt-4o")
        self.store:Dict[str, ChatMessageHistory]={}
        self.conversation_history = {}
       
    # def generate_python_code(self,ques):
    #     llm=ChatOpenAI(model="gpt-4o")
    #     # result=llm.invoke(ques)
    #     # Define the message templates
    #     system_message = SystemMessagePromptTemplate.from_template(
    #         "You are an expert AI Engineer. Generate Python code only based on the questions."
    #         "Ensure the output is valid Python code and starts with the required imports"
    #         "if applicable."
    #         "Please make sure the output should be in proper format and proper indentation"
    #         "please dont give examples in output"
    #     )
    #     user_message = HumanMessagePromptTemplate.from_template("{ques}")
        
    #     # Create the chat prompt template
    #     prompt = ChatPromptTemplate.from_messages([system_message, user_message])
    #     output_parser=StrOutputParser()
    #     chain=prompt|llm|output_parser
    #     response=chain.invoke({"ques": ques})
    #     return response
    
    # def session_store(self,user_id:str):
        
    #     if user_id not in self.store:
    #         self.store[user_id]=ChatMessageHistory()
    #     return self.store[user_id]

    
    
    # def chatbot_mainn(self,ques:str,user_id:str):
    #     # print("hi")
    #     chat_history = self.session_store(user_id)
        # print(chat_history)
       
              # result=llm.invoke(ques)
        # Define the message templates
         
       

        # system_message = SystemMessagePromptTemplate.from_template(
        #     "You are an expert AI Engineer. Provide me answers based on the questions."
        # )
        # user_message = HumanMessagePromptTemplate.from_template("{ques}")
        
        # Create the chat prompt template
        # prompt = ChatPromptTemplate.from_messages([system_message, user_message])
        # output_parser=StrOutputParser()
        
        # chain=prompt|llm|output_parser
        # print(chain)
        # response=chain.invoke({"ques": ques})
        # chat_history.add_user_message(ques)
        # chat_history.add_ai_message(response)
        # print(response)
        # return response
       
        # Retrieve or create chat history for the user
        
        # Retrieve previous messages and format them as a list of BaseMessage objects
        # formatted_history = chat_history.messages  # Access stored messages
        
        # # Add the new user message to the context
        # new_user_message = HumanMessage(content=ques)
        # context = formatted_history + [new_user_message]
        
        # # Generate a response using the LLM
        # response_message = self.llm(context)
        
        # # Update the chat history
        # chat_history.add_user_message(ques)
        # chat_history.add_ai_message(response_message.content)
        
        # return response_message.content
    
    
       

   
    def upload(self,file: UploadFile) -> str:
        try:
            file_content=file.file.read()
            with tempfile.NamedTemporaryFile(delete=False, mode="wb") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            loader=PyPDFLoader(temp_file_path)
            docs=loader.load() 
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
            final_documents=text_splitter.split_documents(docs)
            texts = [doc.page_content for doc in final_documents]
            documents = [Document(page_content=text) for text in texts] 
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            query_results=embeddings.embed_documents(texts)
            client=chromadb.Client()
            collection=client.create_collection("document-embeddings")
            for i,text in enumerate(texts):
                collection.add(
                    documents=[text],
                    embeddings=[query_results[i]],
                    metadatas=[{"source": f"doc_{i}"}],  # Metadata about each document (optional)
                    ids=[str(i)]
                   
                  
                )
            collection_data = {
            "documents": texts,
            "embeddings": query_results,
            "metadata": [{"source": f"doc_{i}"} for i in range(len(texts))],
            "ids": [str(i) for i in range(len(texts))]
             }

            return {"data_with_embeddings": collection_data}
    
            
            
            
            # db=Chroma.from_documents(documents,query_results)
            
    
        except Exception as e:
            return f"Error while processing the file: {str(e)}"
           
        



    def query_documents(self,user_query):
        try:

            client=chromadb.Client()
            collection=client.get_collection("document-embeddings")
            embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
            query_embeddings=embeddings.embed_query(user_query)


            results=collection.query(
                query_embeddings=[query_embeddings],
                n_results=5
            
            )
          

            response=[
                {
                    "document":results["documents"][i],
                    "metadata":results["metadatas"][i],
                    "similarity_score": results["distances"][i]
                }
                for i in range(len(results["documents"]))
            ]
            return {"query_results":response}


        except Exception as e:
            return {"error": f"Error while querying documents: {str(e)}"}
    


    # def execute_python_code(self,code: str):
    #     try:
    #         # Capture standard output and error
    #         output = io.StringIO()
    #         sys.stdout = output

    #         # Execute the code
    #         exec(code)

    #         # Get the result of execution
    #         result = output.getvalue()

    #         return result
    #     except Exception as e:
    #         return f"Error: {str(e)}"