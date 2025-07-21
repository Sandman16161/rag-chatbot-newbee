# rag-chatbot-newbee

# 📚 RAG PDF Chatbot – Ask Your Documents Anything!
This project is a **Retrieval-Augmented Generation (RAG)** based chatbot built using **LangChain**, **FAISS**, **OpenAI GPT**, and **Gradio**. It lets you upload and query PDF documents intelligently – ideal for document search, policy assistance, helpdesk automation, and more. Also, it keeps the history as well(the query doesn't get cleared after getting the soluton)

## 🚀 Features
- ✅ Ask questions directly from multiple PDFs
- ✅ Retrieves relevant chunks using FAISS vector search
- ✅ Uses OpenAI's `gpt-3.5-turbo` or `gpt-4` for accurate responses
- ✅ Source highlighting (shows which PDF & page the answer came from)
- ✅ Clean Gradio-based chat interface
- ✅ Deployed on Hugging Face Space

## 🛠️ Tech Stack
- LangChain – for chaining retrieval and generation
- FAISS – for efficient semantic search
- OpenAI – to generate natural answers
- PyPDFLoader – to extract text from PDFs
- Gradio – to build the chatbot UI
- Hugging Face Spaces – for free hosting and sharing

## 📂 Project Structure
├── app.py # Main application code (Gradio + RAG logic)
├── faiss_index/ # Saved FAISS vector store
├── admin_guide.pdf # Example input PDFs
├── it_help_manual.pdf # More input documents
├── ...
├── requirements.txt # Python dependencies
├── README.md # You're reading this!

## 📄 How It Works
1. **PDF Loading**: PDFs are parsed using `PyPDFLoader` from LangChain.
2. **Chunking**: Text is split into manageable chunks with context overlap.
3. **Embedding**: Chunks are embedded using `OpenAIEmbeddings`.
4. **Indexing**: Chunks are stored in a FAISS vector store.
5. **Retrieval**: Top relevant chunks are retrieved based on user query.
6. **Answering**: The LLM answers the query based on retrieved chunks.
7. **Streaming**: The response is streamed back using Gradio UI.

## 📦 Setup Instructions

1. Install dependencies
   pip install -r requirements.txt
""

3. Set your OpenAI API key
export OPENAI_API_KEY=your-key-here

4. Run the chatbot locally
python app.py

5. Add Your Own PDFs
Replace or add to these files in your project directory:
"admin_guide.pdf
hr_policies.pdf
it_help_manual.pdf
server_issues_guide.pdf"
The chatbot will automatically process them on startup and build a new FAISS index.

## Try it online:
This project is hosted on Hugging Face Spaces:
💡 Example Questions
- How do I reset my company password?
- What is the leave policy for new employees?
- Which ports does the server open by default?
- Steps to configure VPN on Windows?

🤖 Future Enhancements
 - File upload support in UI
 - Multi-document summarization
 - Memory-based conversation
 - Support for other file types (DOCX, TXT)

🙏 Credits
Built using:
- LangChain
- OpenAI API
- Gradio
- FAISS
