import pandas as pd
import streamlit as st
#import altair as alt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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
