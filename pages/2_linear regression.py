# loading dependencies
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

# page configuration
st.set_page_config(page_title="Linear Regression", page_icon="📈", layout="wide")

st.sidebar.header("Regression")

# loading data
data = pd.read_csv('clean_uber_lyft.csv').drop(columns = 'Unnamed: 0')
data = data.iloc[:50000]

# ----- MAINPAGE -----

# ----- ENCODING DATA

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

# ----- LINEAR REGRESSION

# creating first multiselect to tailor explanatory variables to compare
features1 = st.sidebar.multiselect("M1 - Select Explanatory Variables: ", 
                                  options = df.columns[1:],
                                  default = ['distance','surge_multiplier', 'name_enc', 'cab_type_enc'] )

# creating second multiselect with default as all options
features2 = st.sidebar.multiselect("M2 - Select Explanatory Variables: ", 
                                  options = df.columns[1:],
                                  default =  list(df.columns[1:]))

# function getRegression creates model and outputs all metrics for given features
def getRegression(features):
    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # R2 value
    r2 = round(model.score(X_test, y_test),4)

    # Adjusted R2 value
    n = X_test.shape[0]
    k = X_test.shape[1]
    adj_r2 = round(1 - ((1 - r2) * (n - 1) / (n - k - 1)),4)

    # mean squared error
    mse = round(mean_squared_error(y_test, y_pred),4)

    # F-Test
    F_values, p_values = f_regression(X_train, y_train)

    F_value = F_values[0]

    # degrees of freedom for the model
    df_model = X_train.shape[1] - 1

    # degrees of freedom for the residuals
    df_resid = X_train.shape[0] - X_train.shape[1]

    # F-test statistic
    F_test = round((F_value / df_model) / ((np.sum((y - y.mean()) ** 2) / df_resid)),4)

    # Variance Inflation Factors
    vif = pd.DataFrame()
    vif["feature"] = X_train.columns
    vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif["p-val"] = np.round(p_values,4)

    return(r2, adj_r2, mse, vif.sort_values(by='VIF', ascending=False), F_test)

# setting up columns
col1, col2, col3 = st.columns([1,1,1])

# ----- MODEL 1 (OPTIMAL)
with col1:
    st.markdown("<h2 style='color: #A44F66;font-weight: bold;'> Model 1 - Optimal </h2>", unsafe_allow_html=True)

    m1_r2, m1_adj_r2, m1_mse , m1_vif, m1_f = getRegression(features1)

    st.markdown("**R2** : {} | **Adj. R2** : {} ".format(m1_r2, m1_adj_r2))
    st.markdown("**MSE** : {} | **F Statistic** : {} ".format(m1_mse, m1_f))

    st.dataframe(m1_vif, 350, 630)

# ----- MODEL 2 (ALL VARS)

with col2:
    st.markdown("<h2 style='color: #A44F66;font-weight: bold;'> Model 2 - All Vars </h2>", unsafe_allow_html=True)

    m2_r2, m2_adj_r2, m2_mse , m2_vif, m2_f = getRegression(features2)

    st.markdown("**R2** : {} | **Adj. R2** : {} ".format(m2_r2, m2_adj_r2))
    st.markdown("**MSE** : {} | **F Statistic** : {} ".format(m2_mse, m2_f))

    st.dataframe(m2_vif.style.set_properties(**{'text-align': 'center'}), 380, 630)

# ----- Analysis

with col3:
    st.markdown("<h2 style='color: #A44F66;font-weight: bold;text-align : center;'> Analysis </h2>", unsafe_allow_html=True)

    st.markdown("""

                Model 1 is created using the top four variables (distance, surge_mulitplier, name_enc, cab_type_enc) 
                found from observing the correlation matrix. 

                Model 2 is created by using all explanatory variables in the dataset.

                The outcome is that all variables outside of Model 1 add little to no value in improving the performance 
                of the regression. In fact, The adjusted R2 decreases slightly which relates to poorer performance with the
                addition of extra explanatory variables. Moreover, the high VIFs (greater than 15) apparent in Model 2
                indicate there is a high level multicolinearity between explanatory variables.

                However, the R-Squared value of 0.5114 for Model 1 is rather weak and new models should be implemented
                in order to improve the perfomance in prediciting the price of a ride using a rideshare service.
    
                """)