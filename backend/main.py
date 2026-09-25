# main.py
# import os
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import uvicorn

# # LangChain Imports
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains import create_retrieval_chain

# # --- CONFIGURATION ---
# # Set your OpenAI API key in your environment variables:
# # export OPENAI_API_KEY="your-api-key-here"
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

# app = FastAPI(title="TinkerAI Backend", version="1.0")

# # --- MOCK VECTOR DB SETUP ---
# # In production, this would point to a persistent Chroma or Pinecone instance.
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vector_store = Chroma(
#     collection_name="appliance_manuals",
#     embedding_function=embeddings,
#     persist_directory="./chroma_db" 
# )

# # Initialize LLM
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

# # Define the RAG Prompt
# system_prompt = (
#     "You are an expert appliance repair assistant named TinkerAI. "
#     "Use the following pieces of retrieved technical manual context to answer the user's question. "
#     "If you don't know the answer or the context doesn't contain it, say that you don't know and advise calling a professional. "
#     "Use markdown for bolding warnings and numbering steps. \n\n"
#     "Context: {context}"
# )
# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}"),
# ])

# # Build the chains
# question_answer_chain = create_stuff_documents_chain(llm, prompt)

# # --- API MODELS ---
# class ChatRequest(BaseModel):
#     session_id: str
#     appliance_model: str
#     query: str

# class Citation(BaseModel):
#     page: str
#     source: str

# class ChatResponse(BaseModel):
#     response: str
#     citations: List[Citation]

# # --- ENDPOINTS ---
# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     try:
#         # 1. Create a retriever with a metadata filter for the specific appliance model
#         retriever = vector_store.as_retriever(
#             search_kwargs={
#                 "k": 3,
#                 "filter": {"model": request.appliance_model} # Assumes documents were added with this metadata
#             }
#         )
        
#         # 2. Create the retrieval chain
#         rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
#         # 3. Invoke the chain
#         # NOTE: If ChromaDB is empty, this will pass empty context.
#         result = rag_chain.invoke({"input": request.query})
        
#         # 4. Extract citations from the retrieved documents
#         citations = []
#         for doc in result.get("context", []):
#             page = doc.metadata.get("page", "Unknown")
#             source = doc.metadata.get("source", "Manual")
#             citations.append(Citation(page=str(page), source=source))
            
#         return ChatResponse(
#             response=result["answer"],
#             citations=citations
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# LangChain & Ollama Imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

app = FastAPI(title="TinkerAI Backend (Ollama)", version="1.0")

# --- MOCK VECTOR DB SETUP ---
# Point to local Ollama embeddings instead of OpenAI
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    collection_name="appliance_manuals",
    embedding_function=embeddings,
    persist_directory="./chroma_db" 
)

# Point LLM to local Llama 3.1 model instead of GPT-4o
llm = ChatOllama(model="llama3.1", temperature=0)

system_prompt = (
    "You are an expert appliance repair assistant named TinkerAI. "
    "Use the following pieces of retrieved technical manual context to answer the user's question. "
    "If you don't know the answer or the context doesn't contain it, say that you don't know and advise calling a professional. "
    "Use markdown for bolding warnings and numbering steps. \n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

# --- API MODELS ---
class ChatRequest(BaseModel):
    session_id: str
    appliance_model: str
    query: str

class Citation(BaseModel):
    page: str
    source: str

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]

# --- ENDPOINTS ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {"model": request.appliance_model}
            }
        )
        
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        result = rag_chain.invoke({"input": request.query})
        
        citations = []
        for doc in result.get("context", []):
            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "Manual")
            citations.append(Citation(page=str(page), source=source))
            
        return ChatResponse(
            response=result["answer"],
            citations=citations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)