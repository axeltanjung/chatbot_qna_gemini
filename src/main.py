# Credit to github/ardyadipta for the original code
# * Build RAG Application with Gemini using Langchain by Kardne
# * GenAI Project by Khrish Naik

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as generativeai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
generativeai.configure(api_key=api_key)

def extract_pdf_content(pdf_files):
    """
    Extracts the text content from a PDF file.

    Args:
        pdf_files (list): List of PDF file paths.

    Returns:
        str: Text content of the PDF file.
    """
    combined_text = ""
    for pdf in pdf_files:
        pdf_file = PdfReader(pdf)
        for page_num in range(len(pdf_file.pages)):
            page = pdf_file.pages[page_num]
            combined_text += page.extract_text()
    return combined_text

def split_text_into_chunks(content):
    """
    Splits the text content into chunks of 1000 characters.

    Args:
        content (str): Text content to be split.

    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    return splitter.split_text(content)

def generate_vector_index(text_segments):
    """
    Creates a vector index for the text segments and save it to a FAI
    
    Args:
        text_segments (list): List of text segments.
    
    Returns:
        FAISS: FAISS vector index locally as 'faiss_index_store'.
    """
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(text_segments, embed_model)
    vector_db.save_local("faiss_index_store")

def create_qa_chain():
    """
    Creates a question-answering chain using the pre-trained model (Gemini).
    
    Returns:
        QuestionAnsweringChain: Question-answering chain.
    """
    custom_prompt = """
    Please provide a detailed answer based on the context given. If the context does not contain the answer, simply state, 
    "The answer is not available in the provided context." Please do not guess.\n\n
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temprature=0.3)
    prompt_template = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
    qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
    return qa_chain

def handle_user_query(query):
    """
    Handles the user input searching for relevant document and generating answers to the questions.
    
    Args:
        query (str): User query.
    
    Returns:
        None: the answer to the question in streamlit.
    """
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_db = FAISS.load_local("faiss_index_store", embedding_model, allow_dangerous_deserialization=True)
    matched_docs = faiss_db.similarity_search(query)
    qa_chain = create_qa_chain()
    result = qa_chain({"input_documents": matched_docs, "question": query}, return_only_outputs=True)

    st.write("AI Response: ", result["output_text"])

def main():
    """
    Main function for running Streamlit application.

    Sets up the Streamlit application and handles the user input.

    Returns:
        None
    """
    st.set_page_config(page_title="RAG Application with Gemini using Langchain", page_icon="🔗")
    st.header("RAG Application with Gemini using Langchain")

    user_query = st.text_input("Enter a question based on your uploaded PDF:")

    if user_query:
        handle_user_query(user_query)

    with st.sidebar:
        st.title("Upload PDF File")
        pdf_files = st.file_uploader("Upload PDF File", type=["pdf"], accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                extracted_text = extract_pdf_content(pdf_files)
                text_segments = split_text_into_chunks(extracted_text)
                generate_vector_index(text_segments)
                st.success("PDFs processed successfully!")

if __name__ == "__main__":
    main()