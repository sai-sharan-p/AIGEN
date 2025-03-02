from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## function to load gemini model
model=genai.GenerativeModel("gemini-1.5-pro")

def get_gemini_response(input, image, prompt):
    response=genai.generate_content([input, image[0], prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        #read the file into bytes
        bytes_data=uploaded_file.getvalue()

        image_parts= [
            {
                "mine_type":uploaded_file.type,
                "data":bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

st.set_page_config(page_title="MultiLanguage Invoice Content Extractor")

st.header("MultiLanguage Invoice Content Extractor")

input=st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"]) #Image Upload Button
image=''

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption="The uploaded Invoice", use_container_width=True)

submit=st.button("Tell me about the invoice")


input_prompt=""" 
You are an expert in understanding inovices. We will upload a image as invoice and youill have to answer ny question based on the uploaded invoice image
"""

## If submit button is clicked.

if submit:
    image_data=input_image_details(uploaded_file)
    response=get_gemini_response(input_prompt, image, input)
    st.subheader("The response is")
    st.write(response)





