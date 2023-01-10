import streamlit as st 
import pandas as pd 
import numpy as np 
from prophet import Prophet
from prophet.diagnostics import performance_metrics 
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

st.title('Automated Time Series Forecasting')

"""
This data app uses Facebook open source prophet library to autmaticllly
    """

df =st.file_uploader('Import the time series csv file here. Coumns name : ds and y')

if df is not None:
    data = pd.read_csv(df)
    data['ds']=pd.to_datetime(data['ds'], errors='coerce')
    
    st.write(data)
    
    max_date = data['ds'].max()
    #st.write(max_date)
    
### step 2: select Forecast Horizon
""" 
Keep in mind that forecasts become less accurate with larger forecasting
"""
periods_input = st.number_input('how many periods would you like to forecast into the future?',
min_value=1, max_value=365)

if df is not None:
    