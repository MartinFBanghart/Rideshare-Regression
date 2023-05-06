import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# page configuration
st.set_page_config(page_title="Intermediate Regressor", page_icon="📈", layout="wide")

st.sidebar.header("Variable Selection")

# loading data
data = pd.read_csv('clean_uber_lyft.csv').drop(columns = 'Unnamed: 0')
data = data.iloc[:50000]

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

# ----- MAINPAGE -----

col1, col2, col3 = st.columns([1, 1, 1])

# creating first multiselect to tailor explanatory variables to compare
features = st.sidebar.multiselect("Select Explanatory Variables for SVR/DecisionTree: ", 
                                  options = df.columns[1:],
                                  default = ['distance','surge_multiplier', 'name_enc', 'cab_type_enc'] )

# ----- DECISION TREE REGRESSION

def DTregressor(features, max_dep, min_samp_split, min_samp_leaf):
    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

    DT = DecisionTreeRegressor(max_depth = max_dep, min_samples_split = min_samp_split, min_samples_leaf = min_samp_leaf)

    DT.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = DT.predict(X_test)

    # Evaluate the model's performance
    r2_dt = round(r2_score(y_test, y_pred), 4)
    mse_dt = round(mean_squared_error(y_test, y_pred),4)

    return(r2_dt, mse_dt)


with col1:

    st.markdown("<h2 style='color: #A44F66;font-weight: bold; text-align : center;'> Decision Tree </h2>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<h3 style='color: #A44F66;font-weight: bold;'> Tune Hyperparameters </h3>", unsafe_allow_html=True)

    max_depth = st.number_input("Enter max level of depth :", step=1, value=0, format="%d")

    min_samp_split = st.number_input("Enter min samples per split:", step=1, value=0, format="%d")

    min_samp_leaf = st.number_input("Enter min samples per leaf:", step=1, value=0, format="%d")

    if (max_depth and min_samp_split and min_samp_leaf) != 0:
        r2_dt, mse_dt = DTregressor(features, max_depth, min_samp_split, min_samp_leaf)

        st.markdown("")

        st.markdown("<h4 style='color: #A44F66;font-weight: bold;text-align : center'> R-Squared : {} </h4>".format(r2_dt), unsafe_allow_html=True)
        st.markdown("<h4 style='color: #A44F66;font-weight: bold;text-align : center'> Mean Squared Error : {} </h4>".format(mse_dt), unsafe_allow_html=True)
    else:
        print()

# ----- RANDOM FOREST REGRESSION

def randForest(n_est,  m_dep, m_samp_split, m_samp_leaf):

    X = df.iloc[:,1:]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

    rf = RandomForestRegressor(n_estimators=n_est, max_depth=m_dep, min_samples_split=m_samp_split, min_samples_leaf=m_samp_leaf, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    r2 = round(r2_score(y_test, y_pred), 4)

    mse = round(mean_squared_error(y_test, y_pred), 4)

    return(r2, mse)

with col2:

    st.markdown("<h2 style='color: #A44F66;font-weight: bold; text-align : center;'> Random Forest </h2>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<h3 style='color: #A44F66;font-weight: bold;'> Tune Hyperparameters </h3>", unsafe_allow_html=True)

    l_depth = st.number_input("Enter max level of depth:", step=1, value=0, format="%d")

    m_samp_split = st.number_input("Enter min samples per split :", step=1, value=0, format="%d")

    m_samp_leaf = st.number_input("Enter min samples per leaf :", step=1, value=0, format="%d")

    n_estimators = st.number_input("Enter number of estimators:", step=1, value=0, format="%d")

    if (n_estimators and l_depth and m_samp_split and m_samp_leaf) != 0:
        r2, mse = randForest(n_estimators, l_depth, m_samp_split, m_samp_leaf)


        st.markdown("<h4 style='color: #A44F66;font-weight: bold;text-align : center'> R-Squared : {} </h4>".format(r2), unsafe_allow_html=True)
        st.markdown("<h4 style='color: #A44F66;font-weight: bold;text-align : center'> Mean Squared Error : {} </h4>".format(mse), unsafe_allow_html=True)
    else:
        print()


# ----- SUPPORT VECTOR REGRESSOR

def SupportVectorRegressor(features, kern, c, epsil):

    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

    # Standardizing the data splits
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    regressor = SVR(kernel=kern, C=c, epsilon=epsil)

    # Train the model on the training data
    regressor.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = regressor.predict(X_test)

    # Evaluate the model's performance
    r2_svr = round(r2_score(y_test, y_pred), 4)
    mse_svr = round(mean_squared_error(y_test, y_pred),4)

    return(r2_svr, mse_svr)

with col3:
    st.markdown("<h2 style='color: #A44F66;font-weight: bold; text-align : center;'> Support Vector </h2>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<h3 style='color: #A44F66;font-weight: bold;'> Tune Hyperparameters </h3>", unsafe_allow_html=True)

    C = st.number_input("Enter number for C:", step=1, value=0, format="%d")

    epsilon = st.number_input("Enter number for Epsilon:", step=1, value=0, format="%d")

    kernel = st.radio("Choose Kernel", options=["linear", "poly", "rbf"], horizontal=True)

    if (C and epsilon) != 0:
        r2_svr, mse_svr = SupportVectorRegressor(features, kernel, C, epsilon)
        
        st.markdown("")

        st.markdown("<h4 style='color: #A44F66;font-weight: bold; text-align : center'> R-Squared : {} </h4>".format(r2_svr), unsafe_allow_html=True)
        st.markdown("<h4 style='color: #A44F66;font-weight: bold; text-align : center'> Mean Squared Error : {} </h4>".format(mse_svr), unsafe_allow_html=True)
    else:
        print()

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")

st.markdown("** Random Forest Regressor model incorporates all features as the ensemble method does not require feature selection for optimal performance")