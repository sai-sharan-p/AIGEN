from dotenv import load_dotenv
load_dotenv() ##loading all the environent varaibles

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

### function to load gemini pro model & get responses

model=genai.GenerativeModel("gemini-1.5-pro")

def get_gemini_response(input, image=None):
    if not input and not image:  # Check if both the input question and image are empty
        return "Please enter a question or upload an image." 
    try:
        if image:
            if not input:
               input = "Describe this image" # add default text prompt if there is no text
            response = model.generate_content([input, image]) # Send image and question if an image is present
        else:
            response = model.generate_content(input)
        if response.text: #Check if the model provides a text output
          return response.text
        else: 
          return "The model did not produce any text output for this question."
    except Exception as e:
        return f"An error occurred: {e}" # Catching any potential error that happens with the model

## initialize streamlit app

st.set_page_config(page_title="Gemini Image Demo")

st.header("Gemimi Application")
input=st.text_input("Input Prompt: ", key="input")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"]) #Image Upload Button
image=""
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

submit=st.button("Generate the response")

## If the submit is clicked.
if submit:
    response=get_gemini_response(input, image)
    st.subheader("The response is")
    st.write(response)

