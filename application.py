import streamlit as st
import numpy as np
import pandas as pd
import pickle

pipe=pickle.load(open('lr.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://wallpaperaccess.com/full/2503533.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title('car_price_prediction')


df['year']=df['year'].astype(int)
df['kms_driven']=df['kms_driven'].astype(int)

name=st.text_input('give the name of your car')
company=st.text_input('give the name of your company')
year=st.text_input('give the year')
kms_driven=st.text_input('enter the kms_driven')
fuel_type=st.text_input('enter the fuel type')
def make_prediction():
    input = {
        'name':[name],
        'company': [company],
        'year':[year],
        'kms_driven': [kms_driven],
        'fuel_type':[fuel_type]
    }
    query=pd.DataFrame(input)
    prediction=pipe.predict(query)
    st.write(prediction)

if st.button('Make prediction'):
    make_prediction()