from dotenv import load_dotenv
load_dotenv() ##loading all the environent varaibles
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
import pandas as pd  # Import pandas for CSV/Excel processing
import numpy as np

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

INSIGHTS_PROMPT = """
You are a seasoned business intelligence analyst. Your role is to extract actionable insights from data. I will upload a CSV or Excel file. You must think step-by-step to identify valuable business insights, but your final output should ONLY be a numbered list of 5-8 key insights that could be valuable for business decision-making.

**Process (Do NOT include these steps in your output):**

1.  **Understand the Data:** What is the data about? What are the columns, and what do they represent? Describe the data (for your own understanding).
2.  **Look for Patterns:** Are there any obvious trends or correlations between columns? Are there any unexpected values or outliers? (Analyze and identify patterns.)
3.  **Consider Business Implications:** What do these patterns mean for the business? What decisions could be made based on these insights? (Interpret the patterns.)
4.  **Formulate Insights:** Identify 5-8 key insights that could be valuable for business decision-making. These insights should be specific, measurable, and actionable. Focus on:

    *   Significant trends
    *   Correlations between variables
    *   Outliers or anomalies
    *   Potential opportunities
    *   Areas of concern or risk

**Output Format:**

Present ONLY the insights as a numbered list. Each insight should be a single, well-formed sentence.

**Example:**
1.  Insight: Sales in the Northeast region are significantly lower than other regions. Implication: Investigate marketing strategies in the Northeast and consider targeted promotions.
2.  Insight: Customer churn rate is increasing among customers who have contacted support more than twice. Implication: Improve customer support processes to reduce repeat contacts.

**Data:**
{context}

**Insights:**

"""


# --- New functions for handling Excel and CSV data ---

def get_dataframe_txt(data_files):
    """
    Extracts text from pandas DataFrames derived from CSV and Excel files.

    Args:
        data_files: A list of uploaded file objects representing CSV or Excel files.

    Returns:
        A single string containing a text representation of all data.
    """
    all_data_text = ""
    for file in data_files:
        try:
            file_extension = file.name.split(".")[-1].lower()
            if file_extension == "csv":
                df = pd.read_csv(file)
            elif file_extension == "xlsx" or file_extension == "xls":
                df = pd.read_excel(file)
            else:
                st.warning(f"Unsupported file type: {file.name}.  Only CSV and Excel files are accepted.")
                continue  # Skip to the next file

            # Convert DataFrame to string representation
            all_data_text += df.to_string() + "\n\n"  # Add separators
        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")
            continue
    return all_data_text



# --- Modified functions to handle both PDFs and DataFrames ---

def get_document_text(docs, file_types):
    """
    Handles both PDF and DataFrame (Excel/CSV) files.
    """
    text = ""
    for doc, file_type in zip(docs, file_types):
        try:
            if file_type == "pdf":
                # PDF Processing
                pdf_bytes = doc.read()
                pdf_buffer = io.BytesIO(pdf_bytes)
                pdf_reader = PdfReader(pdf_buffer)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif file_type in ["csv", "xlsx", "xls"]:
                # DataFrame Processing
                try:
                    if file_type == "csv":
                        df = pd.read_csv(doc)
                    else:  # Excel files
                        df = pd.read_excel(doc)
                    text += df.to_string() + "\n\n"  # DataFrame to string
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    continue  # Skip to the next file
            else:
                st.warning(f"Unsupported file type. Skipping.")
        except Exception as e:
            st.error(f"Error processing document: {e}")
            continue
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("FAISS_INDEX")


def get_conversational_chain(prompt_template):
    prompt=PromptTemplate(template=prompt_template, input_variables=["context"])  # Removed question input var
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5, streaming=True)  # Enable streaming!
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_insights(data):
    """Generates insights using the INSIGHTS_PROMPT."""
    chain = get_conversational_chain(INSIGHTS_PROMPT)
    response = chain({"input_documents": data}, return_only_outputs=True)
    return response["output_text"]

def user_input(user_question, chain):  # Pass the chain as argument
    """Handles user input and displays streaming responses."""
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local("FAISS_INDEX", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    #Streaming response from the LLM
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})

def main():
    st.set_page_config("Chat With PDFs, Excel, and CSV")
    st.title("Chat With PDFs, Excel, and CSV Using Gemini")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for file uploads
    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload PDF, Excel, and CSV files",
            type=["pdf", "csv", "xlsx", "xls"],
            accept_multiple_files=True
        )
        process_button = st.button("Submit & Process")

    # Main screen: Insights display
    if "insights" in st.session_state:
        st.subheader("Business Insights:")
        st.write(st.session_state.insights) # Display insights

    # Process files and generate insights upon button click
    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing..."):
                # Separate files based on type
                pdf_docs = [f for f in uploaded_files if f.name.endswith(".pdf")]
                excel_csv_docs = [f for f in uploaded_files if f not in pdf_docs]

                file_types = []  # Keep track of file types

                all_docs = []
                all_docs.extend(pdf_docs)
                all_docs.extend(excel_csv_docs)

                for file in pdf_docs:
                    file_types.append("pdf")
                for file in excel_csv_docs:
                    file_types.append(file.name.split(".")[-1].lower())  # Get the extension

                raw_text = get_document_text(all_docs, file_types)  # Process all files
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)

                # Generate insights
                if excel_csv_docs: # Only generate insights if Excel/CSV files were uploaded
                        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
                        new_db=FAISS.load_local("FAISS_INDEX", embeddings, allow_dangerous_deserialization=True)
                        docs=new_db.similarity_search("Generate insights from the data:")

                        st.session_state.insights = generate_insights(docs)
                else:
                    st.session_state.insights = "No Excel/CSV files uploaded. Insights cannot be generated."
                st.rerun() #Rerun the streamlit


    # Chat interface at the bottom
    st.subheader("Chat with your Documents")
    chat_prompt = """
        You are a helpful and expert data analysis assistant. You will receive the COMPLETE data in text form representing the ENTIRE contents of a CSV or Excel file. It is CRUCIAL that you use ALL the provided data when answering questions and performing analysis. Do NOT assume any data is missing or irrelevant. Your goal is to answer user questions about the data and perform various data analysis tasks as requested, using the ENTIRE dataset.

**Instructions:**

1. **Understand the ENTIRE Data:** Carefully examine ALL the data provided. Identify ALL the columns, their data types, and the relationships between them across the ENTIRE dataset. Infer the context of the data based on ALL rows and columns. Do NOT disregard any part of the data.

2. **Answer User Questions using ALL the Data:** Answer user questions about the data as accurately and completely as possible. Provide specific details and insights based on ALL of the data.  Do not limit your analysis to only a subset of the data.

3. **Perform Data Analysis Tasks using ALL the Data:** You are capable of performing the following data analysis tasks, ALWAYS considering the ENTIRE dataset:
   * **Calculations:** Calculate averages, sums, minimums, maximums, and other statistical measures for numerical columns across the ENTIRE dataset. Perform calculations based on user-specified criteria (e.g., "average sales for product X across ALL regions").
   * **Column Analysis:** Describe the characteristics of specific columns across the ENTIRE dataset, including their data types, range of values, and any interesting patterns or anomalies. Provide summary statistics for numerical columns, considering ALL values.
   * **Fact Extraction:** Extract specific facts from the ENTIRE dataset based on user queries (e.g., "What is the total sales for January across ALL regions?", "How many customers are located in California according to ALL customer records?").
   * **Filtering and Sorting:** Filter the ENTIRE dataset based on user-specified criteria (e.g., "Show me ALL orders with a value greater than $100", "List the top 10 customers by revenue based on ALL revenue data").
   * **Trend Analysis:** Identify trends over time (if time-related data is present), analyzing the ENTIRE time period covered by the data.
   * **Correlation Analysis:** Determine correlations between different columns across the ENTIRE dataset.

4. **Provide Clear Explanations:** When providing answers or performing data analysis tasks, explain your reasoning and the steps you took to arrive at the result, referencing the fact that you used the ENTIRE dataset.

5. **Handle Missing Information:** If the answer to a user question is not directly available in the provided data (even after considering the ENTIRE dataset), attempt to derive the answer by performing calculations or analysis across the ENTIRE dataset. If it is truly impossible to answer the question based on the data, respond with "I cannot answer this question based on the available data, even when considering the entire dataset." Do not provide incorrect information.

6. **Example Interactions:**

   * **User:** What is the average sales price?
   * **Assistant:** The average sales price is $XX.XX, calculated by summing the sales prices for ALL transactions in the dataset and dividing by the number of transactions.

   * **User:**  What are the top 5 products by revenue?
   * **Assistant:** The top 5 products by revenue, considering ALL revenue data, are:
      1. Product A: $XXXX
      2. Product B: $XXXX
      3. Product C: $XXXX
      4. Product D: $XXXX
      5. Product E: $XXXX
      This was determined by calculating the revenue for each product and sorting in descending order, using the ENTIRE dataset.

   * **User:** How many customers are located in California?
   * **Assistant:** According to ALL customer records, there are XXX customers located in California.

7. **Prioritize Accuracy Using the ENTIRE Dataset:** Always prioritize accuracy and avoid making assumptions or providing speculative information. Rely solely on the ENTIRE dataset provided.

8. **Context is Key:** Remember that your analysis should be relevant to the likely business context of the data, considering the ENTIRE dataset.\n\n
        Context:\n {context}? \n
        Question: \n {question}? \n
        Answer:"""

    chat_chain = get_conversational_chain(chat_prompt) #Chain created
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question from the uploaded files:"):
        user_input(prompt, chat_chain)
        st.rerun() #Rerun the streamlit

if __name__ == "__main__":
    main()


