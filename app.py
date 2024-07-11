import streamlit as st
import fitz  # PyMuPDF
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import faiss
import numpy as np

# Function to read PDF and extract text
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Initialize RAG components
def initialize_rag():
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
    return tokenizer, model

# Function to get answer from RAG model
def get_answer(question, context, tokenizer, model):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    generated = model.generate(**inputs)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

# Streamlit app
st.title("RAG-based Information Retrieval from PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Reading PDF..."):
        pdf_text = read_pdf(uploaded_file)
    st.text_area("Extracted Text", value=pdf_text, height=300)

    # Initialize RAG model
    tokenizer, model = initialize_rag()

    # Ask question
    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Getting answer..."):
            answer = get_answer(question, pdf_text, tokenizer, model)
        st.write("Answer:", answer)
