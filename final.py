import pandas as pd
import streamlit as st
#import altair as alt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objs as go
from prophet import Prophet
import ydata_profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.sidebar.markdown("**Author**: Govind Pande")





page_select = st.sidebar.selectbox("Select page", ["Project Overview","Stock Visualizations","Share Holders Visualization","Compare Stocks","Price Prediction","Bring your own data"])

def mainn():
  if page_select== "Project Overview":

    st.title("Stock Market Data Visualization Project")




def main():
  mainn()

  ##########
  #pr = df.profile_report()

  #st_profile_report(pr)

main()
