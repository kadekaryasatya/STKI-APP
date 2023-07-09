import streamlit as st
import pandas as pd
import nltk
from streamlit_option_menu import option_menu
import webbrowser


nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="STKI",
                   page_icon="📈", layout="wide")

st.title("Retrival System Apps")


selected = option_menu(None,
    options=["About", "Information Retrival"], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="vertical")


if selected == "About":
   st.header("STKI Apps")
   st.write("STKI_APPS is an interactive web application designed for information retrieval using various methods.")
   st.write("The app allows users to explore and analyze textual data using three popular methods: ")
   st.write("Boolean, TF-IDF (Term Frequency-Inverse Document Frequency), and Vector Space Model.")
   st.divider()
   st.header("Created by : ")
   st.subheader("Ketut Ananta Kevin Permana")
   st.subheader("I Kadek Krisna Prayoga")
   st.subheader("I Kadek Arya Satya Dharma")

elif selected == "Information Retrival":
    with open("information_retrival.py", "r") as file:
      code = file.read()
      exec(code)

# elif selected == "TF-IDF":
#      with open("TF-IDF.py", "r") as file:
#       code = file.read()
#       exec(code)
# elif selected == "VSM":
#      with open("VSM.py", "r") as file:
#       code = file.read()
#       exec(code)




