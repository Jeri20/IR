import streamlit as st
import fitz  # PyMuPDF for PDF reading
import docx2txt
from transformers import pipeline

# Function to read PDF and extract text
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to read DOCX and extract text
def read_docx(file):
    return docx2txt.process(file)

# Function to summarize text using T5 model
def summarize_text_t5(text, summarizer):
    try:
        if not text.strip():
            return ""
        # Split text into chunks that fit model's max_length
        max_chunk_size = 512
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None

# Main function for the Streamlit app
def summarization_page():
    st.title("Text Summarization with T5")

    # Upload document
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]

        # Read PDF or DOCX and extract text
        if file_type == "pdf":
            document_text = read_pdf(uploaded_file)
        elif file_type == "docx":
            document_text = read_docx(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a PDF or DOCX file.")
            return
        
        # Initialize T5 summarizer pipeline
        summarizer_t5 = pipeline("summarization", model="t5-small")

        # Summarize the extracted text
        with st.spinner("Summarizing text..."):
            summary_t5 = summarize_text_t5(document_text, summarizer_t5)
        
        if summary_t5:
            st.subheader("Summary (T5):")
            st.write(summary_t5)

if __name__ == "__main__":
    summarization_page()
