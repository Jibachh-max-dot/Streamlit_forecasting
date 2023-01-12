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
This data app uses Facebook's open-source Prophet library to automate.You'll be able to import your data from a CSV file ,visualize trend
"""

#Step 1 :Import Data
df=st.file_uploader('Import the time series csv file here(2 Column :ds and y)')

if df is not None:
    data=pd.read_csv(df)
    data['ds']=pd.to_datetime(data['ds'],errors='coerce')
    st.write(data)
    max_date=data['ds'].max()

periods_input=st.number_input('How many periods would you like to forecast into the future',
min_value=1,max_value=365)

if df is not None:
    m=Prophet()
    m.fit(data)

if df is not None:
    future=m.make_future_dataframe(periods=periods_input)
    
    forecast = m.predict(future)
    fcst=forecast[['ds','yhat','yhat_lower','yhat_upper']]

    fcst_filtered=fcst[fcst['ds']>max_date]
    st.write(fcst_filtered)

    fig1=m.plot(forecast)
    st.write(fig1)

    fig2=m.plot_components(forecast)
    st.write(fig2)

    # if df is not None:
    #     csv_exp=fcst_filtered.to_csv(index=False)
    #     #When no file name is given, pandas returns the csv as a string,nice.
    #     b64=base64.b64decode(csv_exp.encode()).decode()
    #     href=f'<a href="data:file/csv ;base64,{b64}">Download CSV File </a>(right- click and save as *&lt;forecast_name&gt;.csv*)'
    #     st.markdown(href,unsafe_allow_html=True)