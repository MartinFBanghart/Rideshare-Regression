# loading dependencies
import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as stats
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


# page configuration
st.set_page_config(page_title="General Statistics", page_icon="📈", layout="wide")

st.sidebar.header("Statistic Analysis")

# loading data
data = pd.read_csv('clean_uber_lyft.csv').drop(columns = 'Unnamed: 0')
data = data.iloc[:50000]

# ----- MAINPAGE -----

col1, col2 = st.columns([1,1], gap='small')

# ----- CORRELATION MATRIX

df = data.copy()

# creating encoder object
le = LabelEncoder()

# encoding different columns
df['cab_type_enc'] = le.fit_transform(df['cab_type'])
df['name_enc'] = le.fit_transform(df['name'])

# ensuring encoding is same for source and destination which share same unique elements
combined_col = pd.concat([df['source'], df['destination']])
le.fit(combined_col)

df['source_enc'] = le.transform(df['source'])
df['destination_enc'] = le.transform(df['destination'])

df['short_summary_enc'] = le.fit_transform(df['short_summary'])

# dropping all original columns for categorical variables before creating correlation matrix
df = df.drop(columns=['datetime', 'cab_type', 'name', 'source', 'destination', 'short_summary'])

corr = df.corr()
s = corr.iloc[:,0].sort_values(ascending=False)
s_df = s.to_frame().T


st.markdown("<h2 style='color: #A44F66;font-weight: bold;'> Correlation </h2>", unsafe_allow_html=True)
st.dataframe(s_df.style.background_gradient(cmap='OrRd', vmin=-1, vmax=1))


# ----- HISTOGRAM DISTRIBUTION PLOT

Scatter = st.sidebar.selectbox("Select Variable: ", list(df.columns) )

fig = px.histogram(df[Scatter], x=Scatter, color_discrete_sequence=["#A44F66"])

with col1:
    st.markdown("<h2 style='color: #A44F66;font-weight: bold; text-align: center'> Histogram </h2>", unsafe_allow_html=True)
    st.plotly_chart(fig)


# ----- NORMAL PROBABILITY PLOT

qqplot_data = stats.probplot(df[Scatter], dist="norm", fit=False)

fig = px.scatter(x=qqplot_data[0], y=qqplot_data[1], trendline="ols", color_discrete_sequence=["#A44F66"])
fig.update_yaxes(title_text = Scatter)

with col2:
    st.markdown("<h2 style='color: #A44F66;font-weight: bold; text-align: center'> Normal Probability Plot </h2>", unsafe_allow_html=True)
    st.plotly_chart(fig)



# ----- HIDE STREAMLIT STYLE -----

hide_st_style = """
                <style>

                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}

                </style>

                """
st.markdown(hide_st_style, unsafe_allow_html=True)