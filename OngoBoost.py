#!/usr/bin/env python
# coding: utf-8
# import libraries
import numpy as np
import pandas as pd
import streamlit as st
import plotly as pt
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
#import pandas_profiling as pf
import plotly.express as px
import plotly.graph_objects as go
sns.set_style("darkgrid")

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

# my app’s title
#st.title('Ongo-Total Run Experience Subscription Enhancement')
#st.markdown("""# **OngoBoost-Total Run Experience Subscription Enhancement**""")

st.markdown("""
<style>
body{
    #color:;
    background-color: #FEF2F2;
}
</style>
""",unsafe_allow_html=True)

#st.markdown("""# ** **""")#ff8c69

st.markdown("<h1 style='text-align: center; color: ;'><b>OngoBoost: Subscribe Today Run Tomorrow!</b></h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left; color: ;'><b></b></h3>", unsafe_allow_html=True)
#st.markdown("<h3 style='text-align: left; color: ;'><b></b></h3>", unsafe_allow_html=True)

#st.title("OngoBoost-Subscribe Today Run Tomorrow")
#st.markdown("<h1 style='text-align: center; color: red;'></h1>", unsafe_allow_html=True)
#st.markdown(<style>h1{color: red}='text-align: center; color: red;'>, unsafe_allow_html=True)
#st.header("Upload New Users")

st.markdown("<h4 style='text-align: left; color: ;'><b>Upload New Users</b></h4>", unsafe_allow_html=True)

upload_flag = st.radio("Upload New User csv", ("Yes, upload new user data", "No, use preloaded data"), index=1)

if upload_flag=="Yes, upload new user data":
    csv_file = st.file_uploader(label="", type=["csv"], encoding="utf-8")#Upload a CSV file
    if csv_file is not None:
        data = pd.read_csv(csv_file)
        #if st.checkbox("Show data"):
        st.dataframe(data)

else:
    def get_data():
        #url = r"test_streamlit.csv"
        #path = '/Users/sayantan/Desktop/test_streamlit.csv'
        path = 'test_streamlit.csv'
        return pd.read_csv(path)
    data = get_data()
    st.dataframe(data.head())
#try:





    num_col = ['metric_started_app_session_week1','metric_started_app_session_week2',
            'metric_started_app_session_week3','metric_started_app_session_week4',
            'converted_to_enrolled_program', 'converted_to_started_session',
            'converted_to_completed_session','converted_to_started_subscription']
    data = data[num_col]
    data_cols = data.columns
    data_index = data.index

    from sklearn.externals import joblib
    knni =joblib.load('knni_imputer.joblib')

    data = knni.transform(data)
    data = pd.DataFrame(data=data,index=data_index,columns=data_cols)

    data['metric_started_app_session_week1'] = np.log1p(data['metric_started_app_session_week1'])
    data['metric_started_app_session_week2'] = np.log1p(data['metric_started_app_session_week2'])
    data['metric_started_app_session_week3'] = np.log1p(data['metric_started_app_session_week3'])
    data['metric_started_app_session_week4'] = np.log1p(data['metric_started_app_session_week4'])



    X = data.drop(columns=['converted_to_started_subscription'])

    from sklearn.externals import joblib
    sc =joblib.load('standard_scaler.joblib')
    X = sc.transform(X)


    randomforest = joblib.load('randomforest_prediction.joblib')


    cb = st.checkbox("""You chose to find non-paying customers that have subscription probability between""")


    if cb:
        MORE_THAN_slider = st.slider(
        "More than",
        min_value=1.0,
        max_value=100.0,
        step=1.0,
        value=20.0,
        )
        LESS_THAN_slider = st.slider(
        "Less than",
        min_value=1.0,
        max_value=100.0,
        step=1.0,
        value=60.0,
        )
        predic_proba_subs = pd.DataFrame(randomforest.predict_proba(X), columns=['Non-subscribed', 'Subscription_Probability (%)'])
        predic_proba_subs = pd.DataFrame(predic_proba_subs.iloc[:,-1],columns=['Subscription_Probability (%)'])*100

        filt1 = predic_proba_subs['Subscription_Probability (%)']<LESS_THAN_slider
        filt2 = predic_proba_subs['Subscription_Probability (%)']>MORE_THAN_slider
        filt = filt1 & filt2

        x = predic_proba_subs[filt]
        if st.checkbox("Show filtered customers"):
            st.dataframe(x)

    else:
        st.write("")

#else:
#    st.write("")
