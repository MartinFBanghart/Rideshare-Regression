# Dependencies
import pandas as pd
import plotly.express as px
import streamlit as st

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# configuring page (emojis can be found at webfx.com/tools/emoji-cheat-sheet)
st.set_page_config(page_title = 'Ride Share EDA',
                   page_icon = ':oncoming_automobile:',
                   layout = "wide"
                  )

# loading in data
df = pd.read_csv('clean_uber_lyft.csv').drop(columns = 'Unnamed: 0')


# ----- MAINPAGE -----
st.markdown("<h1 style='color: black;font-weight: bold; text-align: center; font-size: 80px'> Ride Share Dataset </h1>", unsafe_allow_html=True)


# ----- DESCRIPTION OF DATASET

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 style='color: black;font-weight: bold;text-align: center'> Overview </h2>", unsafe_allow_html=True)
    st.markdown("""
              ##### The following dataset was collected from user information of rides taken using popular rideshare apps **Uber** and **Lyft** within Boston, MA

              ##### It consists of **19 variables** pertaining to the price, characteristics of the ride, and basic weather conditions at the time of ride.
              ##### There are 637976 data points within the original set and of this selection 50000 are used for the analysis
               """)

    st.markdown("##")

    # images for uber and lyft logos
    with st.container():
        col_1, col_2, col_3 = st.columns([2, 6, 2])

        with col_2:
            st.image("images/Lyft_logo.png", width=340)
            st.markdown("##")
            st.image("images/Uber_logo.png", width=350)

# ----- DATAFRAME OF ATTRIBUTES           

with col2:
    st.markdown("<h2 style='color: black;font-weight: bold; text-align: center'> Dataset Attributes </h2>", unsafe_allow_html=True)

    cols = list(df.columns)
    types = list(df.dtypes)
    descrip = ['cost of ride [USD]',
            'ride cost multiplier based on activity in area',
            'rideshare provider',
            'rideshare tier and or type',
            'pickup location',
            'dropoff location',
            'length of ride [miles]',
            'datetime information at start of ride',
            'month ride occured in',
            'day ride occured on',
            'hour ride occured',
            'outside temperature during ride [F]',
            'low temperature prediction for the day [F]',
            'time of daily low temperature [Hour]',
            'high temperature prediction for the day [F]',
            'time of daily high temperature [Hour]',
            'intensity of precipitation [mm]',
            'probability of Precipitation in given area',
            'short summary of weather for the day']

    attrs = pd.DataFrame({'Feature': cols,
                          'Description': descrip,
                          'Var Type': types})

    st.dataframe(attrs, 850, 700)


# ----- SIDEBAR -----
st.sidebar.success("Select Page")

# ----- HIDE STREAMLIT STYLE -----

hide_st_style = """
                <style>

                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}

                </style>

                """
st.markdown(hide_st_style, unsafe_allow_html=True)