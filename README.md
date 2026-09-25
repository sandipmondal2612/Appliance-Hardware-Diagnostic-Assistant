# TinkerAI 🛠️📱
An iOS AI Assistant that diagnoses appliance hardware issues by reading local PDF technical manuals using RAG (Retrieval-Augmented Generation). 

![TinkerAI Demo](link_to_your_short_video.gif)

## 🚀 Tech Stack
* **Frontend:** iOS SwiftUI, MVVM Architecture
* **Backend:** Python, FastAPI
* **AI & RAG:** LangChain, Local Llama 3.1 (via Ollama), ChromaDB
* **Embeddings:** nomic-embed-text

## 🧠 How it Works
1. The backend parses a 40+ page OEM technical manual, chunks the text, and stores it in a local Chroma vector database.
2. The iOS client takes the user's natural language query (e.g., "What is Error E20?").
3. The RAG pipeline retrieves the exact diagnostic steps and feeds them to Llama 3.1.
4. The SwiftUI app renders the formatted Markdown response, complete with page citations.

## 🛠️ Setup Instructions
### 1. Backend (Local AI)
* Install [Ollama](https://ollama.com/) and run `ollama pull llama3.1` and `ollama pull nomic-embed-text`.
* Navigate to `/backend`, create a virtual environment, and `pip install -r requirements.txt`.
* Run `python populate_db.py` to embed the sample PDF.
* Run `python main.py` to start the server.

### 2. iOS Frontend
* Open `/ios_app/TinkerAI.xcodeproj` in Xcode 15+.
* Ensure your Mac and iPhone (if testing on device) are on the same network, and build the project.

https://github.com/user-attachments/assets/a416317b-ab01-4aa8-a35b-678151653278



