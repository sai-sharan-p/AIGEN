from dotenv import load_dotenv
load_dotenv() ##loading all the environent varaibles

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

### function to load gemini pro model & get responses

model=genai.GenerativeModel("gemini-1.5-pro")
chat=model.start_chat(history=[])

def get_gemini_response(question):
    response=chat.send_message(question,stream=True)
    return response

## initialize streamlit app

st.set_page_config(page_title="QnA Demo")

st.header("Gemimi Application")

if "chat_history" not in st.session_state:
    st.session_state['chat_history']=[]

input=st.text_input("input:",key="Input")
submit=st.button('Ask me the question')

if submit and input:
    response=get_gemini_response(input)
    ## add user query & response to the session chat history
    st.session_state['chat_history'].append(("You", input))
    st.subheader('The response is:')
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(('Bot',chunk.text))
st.subheader("The chat history is")

for role,text in st.session_state['chat_history']:
    st.write(f"{role}:{text}")








