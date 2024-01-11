
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib

# Function to read and preprocess data
def read_and_preprocess_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df.sort_values(by=['Order Date'], inplace=True, ascending=True)
    df = df.set_index('Order Date')

    # Resample the average of daily sales 
    resample = 'MS'
    new_df = df[['Sales']].resample(resample).mean().interpolate(method='linear')
    return new_df

# Function to load pre-trained model
def load_pretrained_model():
    model = joblib.load('my_model.pkl')
    return model

# Function to perform time series analysis
def analyze_time_series(time_series_df):
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(time_series_df, model='additive')
    fig = decomposition.plot()
    st.pyplot(fig)

# Function to display future forecast
def display_future_forecast(model, time_series_df, forecast_steps):
    pred_uc = model.get_forecast(steps=forecast_steps)
    pred_ci = pred_uc.conf_int()
    fig, ax = plt.subplots(figsize=(14, 7))
    time_series_df.plot(ax=ax, label='observed', color='blue')
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color='orange')
    ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.legend(fontsize=10)
    st.pyplot(fig)


def main():
    st.title('Sales Forecasting App')

    # Read and preprocess data
    new_df = read_and_preprocess_data()

    # Load pre-trained model
    model = load_pretrained_model()

    # User interaction for forecast
    start_date_forecast = st.date_input('Select a start date for the forecast:', pd.to_datetime('2017-01-01'))
    forecast_steps = st.slider('Select the number of forecast steps:', min_value=1, max_value=365, value=7)

    # Calculate and display total sales for the selected forecast period
    pred_uc = model.get_forecast(steps=forecast_steps)
    forecast_sales = pred_uc.predicted_mean.sum()
    st.write(f'Total Sales for the forecast period: {round(forecast_sales, 2)}')

    # Display future forecast
    st.header('Forecast Chart')
    display_future_forecast(model, new_df, forecast_steps)

if __name__ == '__main__':
    main()







