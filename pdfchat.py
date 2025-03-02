from dotenv import load_dotenv
load_dotenv() ##loading all the environent varaibles
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate  # Corrected import here
from langchain.text_splitter import RecursiveCharacterTextSplitter #Corrected import here

import streamlit as st
import os
import google.generativeai as genai
import io

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_txt(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            # Ensure we're reading the file content as bytes
            pdf_bytes = pdf.read()

            # Create a file-like object from the bytes
            pdf_buffer = io.BytesIO(pdf_bytes)

            # Now, pass the buffer to PdfReader
            pdf_reader = PdfReader(pdf_buffer)

            for page in pdf_reader.pages:
                text += page.extract_text()

        except Exception as e:
            print(f"Error reading PDF: {e}")
            continue  # Skip to the next PDF if there's an error

    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunk=text_splitter.split_text(text)
    return chunk

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("FAISS_INDEX")


def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context,make sure you use the content provided in the context only.
    if the answer is not found on the context, just say "answer is not available in the context." Don't provide the wrong answers\n\n
    context:\n {context}? \n
    Question: \n {question}? \n
    Answer:"""

    model= ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)

    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    # Add allow_dangerous_deserialization=True here
    new_db=FAISS.load_local("FAISS_INDEX", embeddings, allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)

    chain=get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat With Multiple PDFs")
    st.header("Chat With Multiple PDFs Using Gemini")

    user_question=st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs= st.file_uploader("Upload the PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing"):
                raw_text=get_pdf_txt(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__=="__main__":
    main()