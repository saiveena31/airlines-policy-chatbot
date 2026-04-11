import os
import warnings
import streamlit as st

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Airlines HR Policy Q&A Bot",
    page_icon="✈️",
    layout="wide"
)

@st.cache_resource
def load_vectorstore(pdf_path: str):
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    documents = splitter.create_documents([raw_text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

def main():
    st.title("✈️ Airlines HR Policy Q&A Bot")
    st.write("Ask questions about **Flykite Airlines HR Policies** using Retrieval-Augmented Generation (RAG).")

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set. Please export it before running the app.")
        st.stop()

    with st.spinner("Loading knowledge base..."):
        # NOTE: replace with your actual PDF path or Hugging Face download logic
        vectorstore = load_vectorstore("deployment/data/Dataset - Flykite Airlines_ HRP.pdf")
        llm = load_llm()

    query = st.text_input("Enter your HR policy question:", placeholder="e.g., What is the bereavement leave policy?")
    if st.button("Get Answer"):
        if not query.strip():
            st.warning("Please enter a valid question.")
            return
        with st.spinner("Searching policy documents..."):
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""You are an HR policy assistant for an airline company.
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

if __name__ == "__main__":
    main()
