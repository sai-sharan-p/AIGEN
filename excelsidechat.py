from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
import uuid  # Import the uuid module

# --- Configuration and Prompts ---
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

Present ONLY the insights as a numbered list. Each insight should be a single, well-formed sentence.  Start each insight with a number and a period, like this:

1.  ...
2.  ...
3.  ...

**Example:**

1.  **Sales** in the **Northeast region** are significantly lower than other regions. Implication: Investigate marketing strategies in the Northeast and consider targeted promotions.
2.  **Customer churn rate** is increasing among customers who have contacted support more than twice. Implication: Improve customer support processes to reduce repeat contacts.

**Data:**
{context}

**Insights:**
"""

CHAT_PROMPT = """
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
        Answer:
"""

# --- Utility Functions ---

def get_dataframe_txt(data_files):
    """Extracts text from pandas DataFrames (CSV/Excel)."""
    all_data_text = ""
    for file in data_files:
        try:
            file_extension = file.name.split(".")[-1].lower()
            if file_extension == "csv":
                df = pd.read_csv(file)
            elif file_extension in ("xlsx", "xls"):
                df = pd.read_excel(file)
            else:
                st.warning(f"Unsupported file type: {file.name}. Only CSV/Excel.")
                continue
            all_data_text += df.to_string() + "\n\n"
        except Exception as e:
            st.error(f"Error processing file {file.name}: {e}")
            continue
    return all_data_text

def get_conversational_chain(prompt_template):
    """Creates a Langchain conversational chain."""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, streaming=True)
    llm_chain = LLMChain(llm=model, prompt=prompt)
    return llm_chain

def generate_insights(data):
    """Generates insights using the INSIGHTS_PROMPT."""
    prompt = PromptTemplate(template=INSIGHTS_PROMPT, input_variables=["context"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, streaming=False)
    llm_chain = LLMChain(llm=model, prompt=prompt)
    response = llm_chain({"context": data}, return_only_outputs=True)
    return response["text"]

def user_input(user_question, chain, data_string):
    """Handles user input and displays streaming responses."""
    try:
        with st.spinner("Generating response..."):
            response = chain(
                {"context": data_string, "question": user_question},
            )

        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": response["text"]})

    except Exception as e:
        st.error(f"An error occurred during chat: {e}")
        st.write("Please check the input and try again.")

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Data Analysis Assistant", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "container_height" not in st.session_state:
            st.session_state.container_height = 800  # Default height

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
        process_button = st.button("Submit & Process")
        power_bi_url = st.text_input("Enter Power BI Embedded URL", "")
        st.session_state.container_height = st.slider("Adjust Height", min_value=300, max_value=1200, value=st.session_state.container_height, step=50) # add height adjustor


    # --- Main Layout ---
    col1, col2 = st.columns([0.7, 0.3])  # 70/30 split

    with col1:
        st.subheader("Power BI Dashboard")
        if power_bi_url:
            st.components.v1.iframe(power_bi_url, height=st.session_state.container_height, scrolling=True)  # Use session state height
        else:
            st.info("Please enter a Power BI Embedded URL in the sidebar.")

    with col2:
        st.subheader("Insights & QnA")  # Combined header

        # --- Insights & Chat Container (with fixed height and scrolling) ---
        insights_chat_container = st.container()
        with insights_chat_container:
            # Set the height of the container using CSS
            st.markdown(f"""
                <style>
                .stContainer {{
                    max-height: {st.session_state.container_height}px;  /* Match dashboard height */
                    overflow-y: auto;
                }}
                </style>
            """, unsafe_allow_html=True)


            # --- Insights Display ---
            if "insights" in st.session_state:
                insights_text = st.session_state.insights
                insights_text = insights_text.replace("\r\n", "<br><br>").replace("\n", "<br><br>")
                for word in ["Sales", "Customer churn rate", "region", "Implication", "customers", "support"]:
                    insights_text = insights_text.replace(word, f"**{word}**")

                insights_html = f"""
                <div style='
                    border: 1px solid #e0e0e0;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    white-space: pre-wrap;
                '>
                    {insights_text}
                </div>
                """
                st.markdown(insights_html, unsafe_allow_html=True)
            else:
                st.info("Upload data and click 'Submit & Process'.")

            # --- Chat Interface ---
            chat_chain = get_conversational_chain(CHAT_PROMPT)
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask a question:", key="chat_input"):
                if not uploaded_files:
                    st.warning("Please upload files first.")
                else:
                    user_input(prompt, chat_chain, get_dataframe_txt(uploaded_files))
                    st.rerun()

    # --- Process Data ---
    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing..."):
                raw_text = get_dataframe_txt(uploaded_files)
                st.session_state.insights = generate_insights(raw_text)
                st.session_state.messages = []  # Clear chat history

if __name__ == "__main__":
    main()