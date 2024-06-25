# ---- IMPORT LIBRARIES 

import streamlit as st  # type: ignore
import warnings
import pandas as pd # type: ignore
import os
from sklearn import preprocessing # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import base64
import tweepy # type: ignore
import csv
import sys    
    
warnings.filterwarnings('ignore') 
# --------------- BACKGROUND IMAGE 

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"  Suspicious Activity Recognition "}</h1>', unsafe_allow_html=True)


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
add_bg_from_local('2.jpg')   
 
# ==== Title 



typee = st.selectbox( 'CHOOSE HERE ',('HOME', 'REGISTER','LOGIN'))    
    
if typee=="HOME":
    # st.text("Welcome to our page !!!")

    st.markdown(f'<h1 style="color:#000000;font-size:24px;">{" Welcome to our Page !!! "}</h1>', unsafe_allow_html=True)
  
    
if typee=="ADMIN":
    st.success("Wlecome Admin!!!")
    
    
    adname=st.text_input("Enter Name","Name")
    passs=st.text_input("Enter Password","Password",type="password")
    aa=st.button("Submit")
    
    # if aa:
        
    if adname=="Admin" and passs=="12345":
        st.success("Admin Login Successfully !!!")    
    
    
        def hyperlink(url):
                return f'<a target="_blank" href="{url}">{url}</a>'
            
        dff = pd.DataFrame(columns=['page'])
        dff['page'] = ['ViewActivity']
        dff['page'] = dff['page'].apply(hyperlink)
        dff = dff.to_html(escape=False)
        
        st.write(dff, unsafe_allow_html=True)       

    
    
if typee=='LOGIN':
    
        UR1 = st.text_input("Login User Name",key="username")
        psslog = st.text_input("Password",key="password",type="password")
        # tokenn=st.text_input("Enter Access Key",key="Access")
        agree = st.button('LOGIN')
        
        if agree:
            try:
                
                df = pd.read_csv(UR1+'.csv')
                U_P1 = df['User'][0]
                U_P2 = df['Password'][0]
                if str(UR1) == str(U_P1) and str(psslog) == str(U_P2):
                    st.success('Successfully Login !!!')    
                    
                    def hyperlink(url):
                            return f'<a target="_blank" href="{url}">{url}</a>'
                        
                    dff = pd.DataFrame(columns=['page'])
                    dff['page'] = ['Suspicious']
                    dff['page'] = dff['page'].apply(hyperlink)
                    dff = dff.to_html(escape=False)
                    
                    st.write(dff, unsafe_allow_html=True)        
      
                else:
                    st.write('Login Failed!!!')
            except:
                st.write('Login Failed!!!')         

if typee=='REGISTER':        
        UR = st.text_input("Register User Name",key="username1")
        pss1 = st.text_input("First Password",key="password1",type="password")
        pss2 = st.text_input("Confirm Password",key="password2",type="password")
        # temp_user=[]
            
        # temp_user.append(UR)
        
        if pss1 == pss2 and len(str(pss1)) > 2:
            import pandas as pd
            
      
            import csv 
            
            # field names 
            fields = ['User', 'Password'] 
            
    
            
            # st.text(temp_user)
            old_row = [[UR,pss1]]
            
            # writing to csv file 
            with open(UR+'.csv', 'w') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
                    
                # writing the fields 
                csvwriter.writerow(fields) 
                    
                # writing the data rows 
                csvwriter.writerows(old_row)
            st.success('Successfully Registered !!!')
        