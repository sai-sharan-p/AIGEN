from dotenv import load_dotenv
load_dotenv() ##loading all the environent varaibles

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image  # Import the Pillow library for image handling
import io

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

##FUNCTION TO LOAD GEMINI PRO MODEL AND GET RESPONSES

model=genai.GenerativeModel("gemini-1.5-pro") # Changed to gemini-pro-vision


def get_gemini_response(question, image=None, input_text=None):
    if not question and not image:  # Check if both the input question and image are empty
        return "Please enter a question or upload an image." 
    try:
        if image:
            if not question:
               question = "Describe this image" # add default text prompt if there is no text
            response = model.generate_content([question, image]) # Send image and question if an image is present
        else:
            response = model.generate_content(question)
        if response.text: #Check if the model provides a text output
          return response.text
        else: 
          return "The model did not produce any text output for this question."
    except Exception as e:
        return f"An error occurred: {e}" # Catching any potential error that happens with the model


### Initialize our streamlit app

st.set_page_config(page_title="Invoice Extractor")
st.header("Gemini LLM Application")
input_prompt = st.text_area("Enter a specific question about the invoice: ", key="input_prompt")
input_text = st.text_input("Enter the invoice name: ", key="input")
uploaded_file = st.file_uploader("Upload an image of the invoice", type=["png", "jpg", "jpeg"]) #Image Upload Button
submit=st.button("Submit")

## When Submit is clicked

if submit:
    image = None
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_data)) # Convert the image file data to a PIL Image object
    response=get_gemini_response(input_prompt, image, input_text) # Pass the image to the function
    st.subheader("The Response is as follows:")
    st.write(response)
    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)