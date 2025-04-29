import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title('🏡 House Price Predictor')

area = st.slider('Area (sq ft)', 500, 3500, 1500)
bedrooms = st.selectbox('Number of Bedrooms', [1,2,3,4,5])
age = st.slider('Age of the House (years)', 0, 30, 10)

if st.button('Predict Price'):
    model = LinearRegression()
    df = pd.read_csv('data/raw/house_data.csv')
    X = df[['area', 'bedrooms', 'age']]
    y = df['price']
    model.fit(X, y)
    prediction = model.predict([[area, bedrooms, age]])
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
