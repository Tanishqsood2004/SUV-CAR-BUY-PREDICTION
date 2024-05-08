import os
import threading
import webbrowser

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
suv_car_df = pd.read_csv('suv_data.csv')

# Split data into train and test sets
X = suv_car_df.iloc[:,[2,3]]
y = suv_car_df.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Define Streamlit app
def app():
    # Load image
    image_path = 'suv_car.jpeg'
    # Display image centered with fixed size of 200x200
    with st.container():
        st.image(image_path, width=350)

    st.title("SUV Car Purchasing Prediction")
    st.write("This app predicts whether a customer will purchase an SUV car based on their age and salary.")

    # Collect user input
    age = st.slider("Select age:", min_value=18, max_value=100, step=1, value=30)
    salary = st.slider("Select salary:", min_value=10000, max_value=200000, step=1000, value=50000)

    # Make prediction
    X_new = [[age, salary]]
    X_new_scaled = sc.transform(X_new)
    y_new = model.predict(X_new_scaled)

    # Display prediction
    if y_new == 1:
        st.write("This person has bought the SUV car.")
    else:
        st.write("This person has not bought the SUV car.")

    # Open the live link in the default web browser
    webbrowser.open_new_tab("http://localhost:8501")

# Run the app
if __name__ == '__main__':
    if os.system("tasklist | find /i 'streamlit.exe'") != 1:
        os.system("taskkill /f /im streamlit.exe")
    thread = threading.Thread(target=app)
    thread.start()
