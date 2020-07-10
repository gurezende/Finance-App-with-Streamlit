# Imports
# Basics
import pandas as pd
import numpy as np
from datetime import datetime
# Web App
import streamlit as st
# Load ML model
import pickle
#Graphics
import plotly.offline as pyo
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Header Image
image=Image.open('jpeg2.jpg')
st.image(image,width=600)

# Title of the Project
st.title('''Finance Dashboard App''')

st.write('''### Current Month''')
this_month = datetime.now().month
st.header(this_month)

# Importing the dataset
df_hist = pd.read_csv('finance_train.csv')
df_hist.columns = ['accountID','year', 1,2,3,4,5,6,7,8,9,10,11,12,'budget','target']
df_year = df_hist.iloc[:,1:14]

# Creating the Sidebar
st.sidebar.header('User Input Feature')
st.sidebar.markdown('''
[Example CSV input File](https://github.com/gurezende/Finance-App-with-Streamlit.git)''')

# Getting user's input file
uploaded_file = st.sidebar.file_uploader('Upload your input CSV file here', type=['csv'])
if uploaded_file is not None:
    df_user = pd.read_csv(uploaded_file)
    user_df = df_user.copy()
    # Determining what is Actuals and what is forecast from the user's input
    # Initial values for the actuals and forecast
    actual = [100,100,100,100,100,100]
    forecast = [100,100,100,100,100,100]
    actualp = [100,100,100,100,100,100]
    forecastp = [100,100,100,100,100,100]

    # Prepare user's input for plotting
    user_df.columns = ['accountID', 1,2,3,4,5,6,7,8,9,10,11,12,'budget']
    user_avg = user_df.groupby('accountID').mean().mean()
    user_avg.pop('budget')
    # Actual and Forecast from user's input
    actual = [exp for exp in user_avg[user_avg.index < this_month]]
    forecast = [exp for exp in user_avg[user_avg >= this_month]]

    # Bar graphic for Expenses TOTAL
    avg_exp = df_year.groupby('year').mean().mean()
    trace1 = go.Bar(x=[m for m in range(1,13) if m <this_month] , y=actual, name='Actuals')
    trace2 = go.Bar(x=[m for m in range(1,13) if m >=this_month] , y=forecast, name='Forecast')
    trace3 = go.Scatter(x=[1,2,3,4,5,6,7,8,9,10,11,12] , y=avg_exp , name='Historic Average', line =dict(color='darkorange', dash='dash'))

    data = [trace1, trace2, trace3]
    layout = go.Layout(title='Average Expenses of Your Projects (TOTAL)')
    fig = go.Figure(data=data, layout=layout)
    st.write(fig)


    # Bar graphic for Expenses by Project
    project = st.selectbox('Choose a Project',([i for i in user_df.accountID]))
    # Actual and Forecast for the chosen project
    actualp = user_df[user_df.accountID==project].iloc[:,1:this_month].values.ravel()
    forecastp = user_df[user_df.accountID==project].iloc[:,this_month:13].values.ravel()
    st.write('Approved Budget', user_df[user_df.accountID == project].iloc[:,[0,13]])
    st.write('Monthly Expenses + Forecasts', user_df[user_df.accountID == project].iloc[:,1:13])

    # Graphic
    avg_exp = df_year.groupby('year').mean().mean()
    trace1 = go.Bar(x=[m for m in range(1,13) if m <this_month] , y=actualp, name='Actuals')
    trace2 = go.Bar(x=[m for m in range(1,13) if m >=this_month] , y=forecastp, name='Forecast')
    trace3 = go.Scatter(x=[1,2,3,4,5,6,7,8,9,10,11,12] , y=avg_exp , name='Historic Average', line =dict(color='darkorange', dash='dash'))

    data = [trace1, trace2, trace3]
    layout = go.Layout(title='Average Expenses by Project')
    fig = go.Figure(data=data, layout=layout)
    st.write(fig)


    # Loading the ML model
    filename = 'modelFinApp.sav'
    model = pickle.load(open(filename, 'rb'))

    # Function to format the input file to feed the ML model
    def format_input(df):
        for col in ['feb','apr','jul','oct','dec']:
            df[col] = df[col] * 2
        df = df.iloc[:,1:14]
        return df

    # Format input for ML model
    df_predict = format_input(df_user)
    # Predictions
    preds = model.predict_proba(df_predict)

    # Creating a DataFrame with Project Name and Predictions for visualization
    st.header('Predictions')
    st.markdown('''Here you can see the probability of your project to end the year
    within the budget where **1 means 100%**''')
    predictions = pd.DataFrame({'Project':df_user.accountID, 'Prob. Fail':preds[:,0], 'Prob. Success':preds[:,1]})

    # Function to color probability > 0.5 in green and < 0.5 in red
    def color_proba(val):
        color = 'red' if val <= 0.5 else 'green'
        return 'background-color: %s' % color

    st.dataframe(predictions.style.background_gradient(cmap='BuGn'))
    #st.write(predictions)
    st.sidebar.text('App by Gustavo R Santos')
else:
    st.markdown('# Please load your Forecast File to start')
    st.sidebar.text('App by Gustavo R Santos')
