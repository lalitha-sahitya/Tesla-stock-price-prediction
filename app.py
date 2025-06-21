import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


st.title("Tesla Data Forecast Viewer")

@st.cache_resource
def load_model():
    with open('prophet_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_forecast():
    with open('forecast.pkl', 'rb') as f:
        return pickle.load(f)


m = load_model()
forecast = load_forecast()


if st.checkbox("Show raw forecast data"):
    st.write(forecast.head())


st.subheader("Forecast Plot")

fig = m.plot(forecast)
st.pyplot(fig)


if st.checkbox("Show forecast components"):
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
