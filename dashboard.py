import streamlit as st
import numpy as np
from ocrimg import OCRImg
from PIL import Image

def main():
    new_title = '<p style="font-size: 42px;">CardR: an OCR Application for Business Cards</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    st.sidebar.title("Navigation Bar")
    choice = st.sidebar.selectbox("Select an Option", ("About", "Upload an Image!", "Take a Photo as Image!"))
    
    if choice == "About":
        #st.subheader("something")
        st.text("Proof-Of-Concept for Optical Character Recognition of business cards \nPowered with the Tesseract OCR Engine wrapped in Pytesseract")
        
    elif choice == "Upload an Image!":
        #read_me_0.empty()
        #read_me.empty()
        try:
            uploaded_img = st.file_uploader("Choose an image to begin OCR", type = ["jpg", "jpeg", "png"])
            
            if uploaded_img is not None:
                image = Image.open(uploaded_img)
                st.header("Image Uploaded for OCR")
                st.image(image)
                
                image2 = OCRImg(image)
                res_text = image2.get_text()
                res_json = image2.get_json()
                
                st.header("Result of the OCR")
                st.subheader("Text detected")
                st.text(res_text)
                st.header("JSON Output")
                st.json(res_json)
                
        except OSError as e:
            st.header("Encountered Error")
            st.text(e)
            
    elif choice == "Take a Photo as Image!":
        picture = st.camera_input("Take a picture to use for OCR!")
        
        if picture:
            st.header("Image Taken for OCR")
            st.image(picture)
            
            pic2 = OCRImg(picture)
            res_text2 = pic2.get_text()
            res_json2 = pic2.get_json()
                
            st.header("Result of the OCR")
            st.subheader("Text detected")
            st.text(res_text2)
            st.subheader("JSON Output")
            st.json(res_json2)

if __name__ == "__main__":
    main()            
