# ✈️ Airlines HR Policy Q&A Bot

This is a Streamlit app that answers questions about **Flykite Airlines HR Policies** using Retrieval-Augmented Generation (RAG).

## Features
- PDF stored on Hugging Face Hub
- Fetches PDF dynamically at runtime
- Uses FAISS + HuggingFace embeddings for retrieval
- Powered by Groq LLM for fast responses

## Setup
1. Clone the repo:
   git clone https://github.com/saiveena31/airlines-policy-chatbot.git
   cd airlines-policy-chatbot

2. Install dependencies:
   pip install -r requirements.txt

3. Set your Groq API key:
   export GROQ_API_KEY=your_api_key_here

4. Run the app:
   streamlit run deployment/app.py
