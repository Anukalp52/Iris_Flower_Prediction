import streamlit as st
import pandas as pd
import pickle

model1=pickle.load(open("IrisFlower.pkl","rb"))

def mydeploy():
    st.title("Iris Flower Prediction")
    sepal_length=st.number_input("Enter Sepal Length (cm)")
    sepal_width=st.number_input("Enter Sepal Width (cm)")
    petal_length=st.number_input("Enter Petal Length (cm)")
    petal_width=st.number_input("Enter Petal Width (cm)")
    pred=st.button("Predict Flower Type")
    if pred:
        df=pd.DataFrame({"sepal length (cm)":[sepal_length],"sepal width (cm)":[sepal_width],"petal length (cm)":[petal_length],"petal width (cm)":[petal_width]})
        x=model1.predict(df)
        if x==0:
            st.write("Iris Flower Type: Setosa")
        elif x==1:
            st.write("Iris Flower Type: Versicolor")
        else:
            st.write("Iris Flower Type: Virginica")
mydeploy()    