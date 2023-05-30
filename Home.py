import streamlit as st
import pandas as pd
import nltk
from streamlit_option_menu import option_menu
import webbrowser


nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="STKI",
                   page_icon="📈", layout="wide")

st.title("Welcome to STKI Apps")
st.write("Select one of the menu")


selected = option_menu(None,
    options=["About", "Boolean", "TF-IDF", 'VSM'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")


if selected == "About":
   st.header("STKI Apps")
   st.write("STKI_APPS is an interactive web application designed for information retrieval using various methods.")
   st.write("The app allows users to explore and analyze textual data using three popular methods: ")
   st.write("Boolean, TF-IDF (Term Frequency-Inverse Document Frequency), and Vector Space Model.")
   st.divider()
   st.header("Created by : ")
   st.subheader("Kevin, Krisna, Dek Arya")

elif selected == "Boolean":
    with open("Boolean.py", "r") as file:
      code = file.read()
      exec(code)

elif selected == "TF-IDF":
     with open("TF-IDF.py", "r") as file:
      code = file.read()
      exec(code)
elif selected == "VSM":
     with open("VSM.py", "r") as file:
      code = file.read()
      exec(code)



