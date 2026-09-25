# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma

# # --- CONFIGURATION ---
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# # Make sure you have a PDF file with this name in your project folder
# PDF_FILE_PATH = "sample_manual.pdf" 
# # This MUST match the model string requested by your iOS ViewModel
# APPLIANCE_MODEL = "Electrolux EFLS627UTT" 

# def populate_database():
#     if not os.path.exists(PDF_FILE_PATH):
#         print(f"Error: Could not find {PDF_FILE_PATH}. Please add a PDF to the folder.")
#         return

#     print(f"Loading {PDF_FILE_PATH}...")
#     loader = PyPDFLoader(PDF_FILE_PATH)
#     documents = loader.load()
    
#     print("Chunking document into smaller pieces...")
#     # Overlap ensures context isn't lost if a sentence is cut in half between chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200, 
#         length_function=len
#     )
#     chunks = text_splitter.split_documents(documents)
    
#     # Inject the required metadata for the FastAPI filter to find these chunks
#     for chunk in chunks:
#         chunk.metadata["model"] = APPLIANCE_MODEL
#         # PyPDFLoader automatically adds a 'page' metadata field, which our backend uses for citations!
        
#     print(f"Created {len(chunks)} chunks. Embedding and saving to ChromaDB...")
    
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     vector_store = Chroma(
#         collection_name="appliance_manuals",
#         embedding_function=embeddings,
#         persist_directory="./chroma_db"
#     )
    
#     vector_store.add_documents(chunks)
#     print("Success! The database is populated.")

# if __name__ == "__main__":
#     populate_database()

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
PDF_FILE_PATH = "sample_manual.pdf" 
# This MUST match the model string requested by your iOS ViewModel
APPLIANCE_MODEL = "Electrolux EFLS627UTT" 

def populate_database():
    if not os.path.exists(PDF_FILE_PATH):
        print(f"Error: Could not find {PDF_FILE_PATH}. Please add a PDF to the folder.")
        return

    print(f"Loading {PDF_FILE_PATH}...")
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()
    
    print("Chunking document into smaller pieces...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    for chunk in chunks:
        chunk.metadata["model"] = APPLIANCE_MODEL
        
    print(f"Created {len(chunks)} chunks. Embedding and saving to ChromaDB with Ollama...")
    
    # Using local Ollama embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name="appliance_manuals",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    vector_store.add_documents(chunks)
    print("Success! The database is populated.")

if __name__ == "__main__":
    populate_database()