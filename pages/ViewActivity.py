# ---- IMPORT LIBRARIES 


import streamlit as st

import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import base64
import tweepy
import csv
import sys    
    
    
# --------------- BACKGROUND IMAGE 


st.markdown(f'<h1 style="color:#00000;text-align: center;font-size:36px;">{"  Suspicious Activity Recognition "}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('3.avif')   


# --------------- READ CSV


import pandas as pd

data = pd.read_csv("Result.csv")

data = data['Result']

aab = st.button(" View Notification")

if aab:
    st.text(data)





