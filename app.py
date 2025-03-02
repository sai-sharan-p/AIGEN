from dotenv import load_dotenv
load_dotenv() ##loading all the environent varaibles

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

##FUNCTION TO LOAD GEMINI PRO MODEL AND GET RESPONSES

model=genai.GenerativeModel("gemini-pro")

def get_gemini_response(question):
    if not question:  # Check if the input question is empty
        return "Please enter a question." 
    try:
        response = model.generate_content(question)
        if response.text: #Check if the model provides a text output
          return response.text
        else: 
          return "The model did not produce any text output for this question."
    except Exception as e:
        return f"An error occurred: {e}" # Catching any potential error that happens with the model

### Initialize our streamlit app

st.set_page_config(page_title="Ask Me Anything")
st.header("Gemini LLM Application")
input=st.text_input("Question: ", key="input")
submit=st.button("Submit")

## When Submit is clicked

if submit:
    response=get_gemini_response(input)
    st.subheader("The Response is as follows:")
    st.write(response)

