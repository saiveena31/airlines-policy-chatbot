# deployment/app.py
import os
import warnings
import streamlit as st

from huggingface_hub import hf_hub_download
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Streamlit page config
# --------------------------------------------------
st.set_page_config(
    page_title="Airlines HR Policy Q&A Bot",
    page_icon="✈️",
    layout="wide"
)

# --------------------------------------------------
# Cache: Load PDF + Build Vector Store
# --------------------------------------------------
@st.cache_resource
def load_vectorstore():
    # Download PDF from Hugging Face dataset repo
    pdf_path = hf_hub_download(
        repo_id="saiveena/airlines-chatbot",   # replace with your dataset repo
        filename="Dataset - Flykite Airlines_ HRP.pdf",
        repo_type="dataset"
    )

    reader = PdfReader(pdf_path)
    raw_text = "".join([page.extract_text() or "" for page in reader.pages])

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.create_documents([raw_text])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# --------------------------------------------------
# Cache: Load LLM
# --------------------------------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

# --------------------------------------------------
# Main App Logic
# --------------------------------------------------
def main():
    st.title("✈️ Airlines HR Policy Q&A Bot")
    st.write(
        "Ask questions about **Flykite Airlines HR Policies**. "
        "Answers are generated using Retrieval-Augmented Generation (RAG)."
    )

    # Validate API Key
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set. Please add it in Streamlit Cloud secrets.")
        st.stop()

    # Load resources
    with st.spinner("Loading knowledge base..."):
        vectorstore = load_vectorstore()
        llm = load_llm()

    # User input
    query = st.text_input(
        "Enter your HR policy question:",
        placeholder="e.g., What is the bereavement leave policy?"
    )

    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a valid question.")
            return

        with st.spinner("Searching policy documents..."):
            docs = vectorstore.similarity_search(query, k=3)
            context = "

".join([doc.page_content for doc in docs])

            prompt = f"""
You are an HR policy assistant for an airline company.
Answer the question strictly using the context below.
If the answer is not available, say "Not specified in the policy."

Context:
{context}

Question:
{query}

Answer:
"""
            response = llm.predict(prompt)

        st.subheader("Answer")
        st.write(response)

        with st.expander("📄 Retrieved Policy Context"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content)

# --------------------------------------------------
# Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    main()

