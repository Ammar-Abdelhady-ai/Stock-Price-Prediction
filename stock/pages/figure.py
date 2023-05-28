import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import yfinance as yf
import os
from datetime import datetime
import io
import urllib, base64
import streamlit as st
from PIL import Image
import io
import urllib, base64
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.tools as tls
import plotly.graph_objs as go
import numpy as np



yf.pdr_override()

############################################
end = datetime.now()
start = datetime(end.year - 11, end.month, end.day)



valid_apple = joblib.load(r".\pages\save data\apple_validdata.h5")
train_apple = joblib.load(r".\pages\save data\apple_traindata.h5")

valid_google = joblib.load(r".\pages\save data\google_validdata.h5")
train_google = joblib.load(r".\pages\save data\google_traindata.h5")


valid_amazon = joblib.load(r".\pages\save data\amazon_validdata.h5")
train_amazon = joblib.load(r".\pages\save data\amazon_traindata.h5")


valid_meta = joblib.load(r".\pages\save data\meta_validdata.h5")
train_meta = joblib.load(r".\pages\save data\meta_traindata.h5")


valid_microsoft = joblib.load(r".\pages\save data\microsoft_validdata.h5")
train_microsoft = joblib.load(r".\pages\save data\microsoft_traindata.h5")


valid_netflix = joblib.load(r".\pages\save data\netflix_validdata.h5")
train_netflix = joblib.load(r".\pages\save data\netflix_traindata.h5")



valid_tesla = joblib.load(r".\pages\save data\tesla_validdata.h5")
train_tesla = joblib.load(r".\pages\save data\tesla_traindata.h5")



valid_intel = joblib.load(r".\pages\save data\intel_validdata.h5")
train_intel = joblib.load(r".\pages\save data\intel_traindata.h5")



valid_nvidia = joblib.load(r".\pages\save data\nvidia_validdata.h5")
train_nvidia = joblib.load(r".\pages\save data\nvidia_traindata.h5")



apple_df = yf.download("AAPL", start, end)
google_df = yf.download("GOOGL", start, end)
microsoft_df = yf.download("MSFT", start, end)
meta_df = yf.download("meta", start, end)
amazon_df = yf.download("AMZN", start, end)
netflix_df = yf.download("NFLX", start, end)
tesla_df = yf.download('TSLA', start, end)
intel_df = yf.download('INTC', start, end)
nvidia_df = yf.download("NVDA", start, end)





def plot_model_data(train, validate, company):
    # Code for generating the plot
    fig = plt.figure(figsize=(16,6))
    plt.title(f'{company} Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(validate[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

    # Convert the plot to plotly
    plotly_fig = tls.mpl_to_plotly(fig)

    # Update the plot layout
    plotly_fig.layout.update(
        title=f'{company} Model',
        xaxis=dict(title='Date', showgrid=False, showticklabels=True),
        yaxis=dict(title='Close Price USD ($)', showgrid=False, showticklabels=True),
        legend=dict(x=0, y=1, traceorder='normal'),
        margin=dict(l=80, r=50, t=100, b=80),
        width=1000,
        height=500
    )

    # Return the plotly figure
    return  pio.to_json(plotly_fig)


def data_plot(df, company):
    fig = px.line(data_frame=df, y="Close", template = 'plotly_dark', 
                  title=f"Close of {company}")
    line = pio.to_json(fig)

    return line


def change_rate(df, company):
    data = df["Close"]
    # Calculate percentage change
    change_rate = np.diff(data) / data[:-1] * 100

    # Create the figure
    fig = go.Figure(data=go.Scatter(x=df.index[1:], y=change_rate, mode='lines'))

    # Set the layout
    fig.update_layout(title=f'Change Rate in Data of {company}', xaxis_title='Index', yaxis_title='Percentage Change (%)')

    # Show the figure

    # Show plotly figure
    return pio.to_json(fig)

company1 = "Apple"
apple_plot_model_data = plot_model_data(train_apple, valid_apple, company1)
apple_data_plot = data_plot(apple_df, company1)
apple_change_rate = change_rate(apple_df, company1)


company2 = "Google"
google_plot_model_data = plot_model_data(train_google, valid_google, company2)
google_data_plot = data_plot(google_df, company2)
google_change_rate = change_rate(google_df, company2)


company3 = "Microsoft"
microsoft_plot_model_data = plot_model_data(train_microsoft, valid_microsoft, company3)
microsoft_data_plot = data_plot(microsoft_df, company3)
microsoft_change_rate = change_rate(microsoft_df, company3)




company4 = "Amazon"
amazon_plot_model_data = plot_model_data(train_amazon, valid_amazon, company4)
amazon_data_plot = data_plot(amazon_df, company4)
amazon_change_rate = change_rate(amazon_df, company4)



company5 = "Meta"
meta_plot_model_data = plot_model_data(train_meta, valid_meta, company5)
meta_data_plot = data_plot(meta_df, company5)
meta_change_rate = change_rate(meta_df, company5)



company6 = "Netflix"
netflix_plot_model_data = plot_model_data(train_netflix, valid_netflix, company6)
netflix_data_plot = data_plot(netflix_df, company6)
netflix_change_rate = change_rate(netflix_df, company6)



company7 = "Tesla"
tesla_plot_model_data = plot_model_data(train_tesla, valid_tesla, company7)
tesla_data_plot = data_plot(tesla_df, company7)
tesla_change_rate = change_rate(tesla_df, company7)



company8 = "Intel"
intel_plot_model_data = plot_model_data(train_intel, valid_intel, company8)
intel_data_plot = data_plot(intel_df, company8)
intel_change_rate = change_rate(intel_df, company8)



company9 = "Nvidia"
nvidia_plot_model_data = plot_model_data(train_nvidia, valid_nvidia, company9)
nvidia_data_plot = data_plot(nvidia_df, company9)
nvidia_change_rate = change_rate(nvidia_df, company9)