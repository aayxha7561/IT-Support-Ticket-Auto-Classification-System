import streamlit as st
import joblib
import re
import numpy as np
import base64

st.set_page_config(
    page_title="IT Support Ticket Classification",
    page_icon="icon.png"
)

#to set background
with open("banner.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(255,255,255,0.89), rgba(255,255,255,0.89)),
            url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#load trained model
model=joblib.load("model/it_support_ticket_logistic_model.pkl")
tfidf=joblib.load("model/tfidf_vectorizer_new.pkl")

def go_to_classification():
    st.session_state.page = "Classification"

def go_to_home():
    st.session_state.page = "Home"

#text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

#create prediction function
def predict_ticket(ticket):
    cleaned=clean_text(ticket)
    ticket_tfidf= tfidf.transform([cleaned])
    prediction = model.predict(ticket_tfidf)
    return prediction[0]



page = st.radio(
    "Navigation",
    ["Home", "Classification"],
    horizontal=True,
    key="page"
)

if page == "Home":
   st.title("Welcome to IT Support Ticket System")

   st.write("An intelligent system to classify IT support tickets using Machine Learning")
        
   st.button(
       "Get Started",
       on_click=go_to_classification
   )
    
elif page == "Classification":

    st.title("IT Support Ticket Auto Classification System")

    st.write("Enter your IT support ticket below:")

    ticket = st.text_area("Enter Ticket")

    if st.button("Predict Category"):
        if ticket.strip():
            result= predict_ticket(ticket)
            st.success(f"Predicted Category: {result}")
        else:
            st.warning("Please enter a ticket.")

    col1,col2=st.columns([3,1])
    with col2:
    
        st.button(
            "Back to Home",
            on_click=go_to_home
        )
           
       

        

