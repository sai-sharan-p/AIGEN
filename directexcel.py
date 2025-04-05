from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Any, Optional
import streamlit as st
import os
import google.generativeai as genai
import pandas as pd
from functools import lru_cache
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Constants
SUPPORTED_FILE_TYPES = {"csv", "xlsx", "xls"}
DEFAULT_POWERBI_HEIGHT = 600
MIN_POWERBI_HEIGHT = 300
MAX_POWERBI_HEIGHT = 1000
BOLD_KEYWORDS = ["Sales", "Customer churn rate", "region", "Implication", "customers", "support"]

# --- Utility Functions ---
@st.cache_data
def get_dataframe_txt(data_files: List[Any]) -> str:
    """
    Extracts text from pandas DataFrames (CSV/Excel) with caching.
    
    Args:
        data_files: List of uploaded files
        
    Returns:
        str: Concatenated text representation of all dataframes
    """
    all_data_text = []
    for file in data_files:
        try:
            file_extension = Path(file.name).suffix.lower()[1:]
            if file_extension not in SUPPORTED_FILE_TYPES:
                st.warning(f"Unsupported file type: {file.name}. Only CSV/Excel.")
                continue
                
            df = pd.read_csv(file) if file_extension == "csv" else pd.read_excel(file)
            all_data_text.append(df.to_string())
            
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            st.error(f"Error processing file {file.name}: {e}")
            continue
            
    return "\n\n".join(all_data_text)

@st.cache_resource
def get_conversational_chain(prompt_template: str) -> LLMChain:
    """
    Creates a cached Langchain conversational chain.
    
    Args:
        prompt_template: The prompt template to use
        
    Returns:
        LLMChain: The configured chain
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.5, streaming=True)
    return LLMChain(llm=model, prompt=prompt)

@st.cache_resource
def get_insights_chain() -> LLMChain:
    """
    Creates a cached insights generation chain.
    
    Returns:
        LLMChain: The configured chain
    """
    prompt = PromptTemplate(template=INSIGHTS_PROMPT, input_variables=["context"])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3, streaming=False)
    return LLMChain(llm=model, prompt=prompt)

def generate_insights(data: str) -> str:
    """
    Generates insights using the INSIGHTS_PROMPT.
    
    Args:
        data: The data to analyze
        
    Returns:
        str: Generated insights
    """
    try:
        chain = get_insights_chain()
        response = chain({"context": data}, return_only_outputs=True)
        return response["text"]
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        st.error("Failed to generate insights. Please try again.")
        return ""

def format_insights_text(insights_text: str, height: int) -> str:
    """
    Formats insights text with HTML styling.
    
    Args:
        insights_text: The insights text to format
        height: The container height
        
    Returns:
        str: Formatted HTML
    """
    # Add spacing and bolding
    insights_text = insights_text.replace("\r\n", "<br><br>").replace("\n", "<br><br>")
    
    # Bold key terms
    for word in BOLD_KEYWORDS:
        insights_text = insights_text.replace(word, f"**{word}**")
    
    return f"""
    <div style='
        height: {height}px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
        white-space: pre-wrap;
    '>
        {insights_text}
    </div>
    """

def user_input(user_question: str, chain: LLMChain, data_string: str) -> None:
    """
    Handles user input and generates response.
    
    Args:
        user_question: The user's question
        chain: The LLM chain to use
        data_string: The data context
    """
    try:
        with st.spinner("Generating response..."):
            response = chain({"context": data_string, "question": user_question})
            
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": response["text"]})
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        st.error("An error occurred during chat. Please try again.")

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "powerbi_height" not in st.session_state:
        st.session_state.powerbi_height = DEFAULT_POWERBI_HEIGHT

def render_header() -> None:
    """Render the fixed header."""
    st.markdown(
        """
        <style>
        .fixed-header {
            position: sticky;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #f0f2f6;
            padding: 2px 0;
            z-index: 1000;
            text-align: center;
        }
        </style>
        <div class="fixed-header">
            <h2>Data Analytics Assistant</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_sidebar() -> tuple:
    """
    Render the sidebar and return uploaded files and Power BI URL.
    
    Returns:
        tuple: (uploaded_files, power_bi_url)
    """
    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload Excel/CSV",
            type=list(SUPPORTED_FILE_TYPES),
            accept_multiple_files=True
        )
        process_button = st.button("Submit & Process")
        power_bi_url = st.text_input("Enter Power BI Embedded URL", "")
        st.session_state.powerbi_height = st.slider(
            "Dashboard Height",
            MIN_POWERBI_HEIGHT,
            MAX_POWERBI_HEIGHT,
            st.session_state.powerbi_height,
            50
        )
    return uploaded_files, power_bi_url, process_button

def render_dashboard(power_bi_url: str) -> None:
    """
    Render the Power BI dashboard.
    
    Args:
        power_bi_url: The Power BI URL to embed
    """
    st.subheader("Power BI Dashboard")
    if power_bi_url:
        st.components.v1.iframe(
            power_bi_url,
            height=st.session_state.powerbi_height,
            scrolling=True
        )
    else:
        st.info("Please enter a Power BI Embedded URL in the sidebar.")

def render_insights() -> None:
    """Render the insights section."""
    st.subheader("Insights:")
    if "insights" in st.session_state:
        insights_html = format_insights_text(
            st.session_state.insights,
            st.session_state.powerbi_height
        )
        st.markdown(insights_html, unsafe_allow_html=True)
    else:
        st.info("Upload data and click 'Submit & Process'.")

def render_chat(uploaded_files: List[Any]) -> None:
    """
    Render the chat interface.
    
    Args:
        uploaded_files: List of uploaded files
    """
    st.subheader("Chat with your Data")
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

def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="Data Analytics Assistant", layout="wide")
    
    initialize_session_state()
    render_header()
    
    uploaded_files, power_bi_url, process_button = render_sidebar()
    
    # Main content area
    top_container = st.container()
    with top_container:
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:
            render_dashboard(power_bi_url)
            
        with col2:
            render_insights()
    
    # Chat area
    chat_container = st.container()
    with chat_container:
        render_chat(uploaded_files)
    
    # Process button handler
    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing..."):
                raw_text = get_dataframe_txt(uploaded_files)
                st.session_state.insights = generate_insights(raw_text)

if __name__ == "__main__":
    main()
