import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 


data = pd.read_csv("data//ev_sales.csv")
x = np.array(data['Year']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Sales']))


st.title("EV Sales Predictor")
st.image("data//ev.webp",width = 800)
nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
if nav == "Home":
    
    if st.checkbox("Show Table"):
        st.table(data)
    
    graph = st.selectbox("What kind of Graph ? ",["Non-Interactive","Interactive"])

    val = st.slider("Filter data using years",0,20)
    data = data.loc[data["Year"]>= val]
    if graph == "Non-Interactive":
        plt.figure(figsize = (10,5))
        plt.scatter(data["Year"],data["Sales"])
        plt.ylim(0)
        plt.xlabel("Year")
        plt.ylabel("Sales in year")
        plt.tight_layout()
        st.pyplot()
    if graph == "Interactive":
        layout =go.Layout(
            xaxis = dict(range=[2013,2022]),
            yaxis = dict(range =[2000,1000000])
        )
        fig = go.Figure(data=go.Scatter(x=data["Year"], y=data["Sales"], mode='markers'),layout = layout)
        st.plotly_chart(fig)
    
if nav == "Prediction":
    st.header("EV sales")
    val = st.number_input("Enter the Year",2023,2050,step = 1)
    val = np.array(val).reshape(1,-1)
    pred =lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Predicted EV sales for {val} is {round(pred)}")

if nav == "Contribute":
    st.header("What year is it???\nContribute to our dataset to keep us upto date")
    ex = st.number_input("Enter the year",2023,2050)
    sal = st.number_input("Enter EV Sales for {ex} Year",0,10000000,step = 20000)
    if st.button("submit"):
        to_add = {"Year":[ex],"Sales":[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//ev_sales.csv",mode='a',header = False,index= False)
        st.success("Submitted")
