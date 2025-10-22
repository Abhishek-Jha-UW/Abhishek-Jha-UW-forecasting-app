import streamlit as st
import pandas as pd
from forecasting_models import SimpleMA

st.title("📈 Forecasting Tool")

uploaded_file = st.file_uploader("Upload CSV with Date + numeric column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    target_column = st.selectbox("Select column to forecast", df.select_dtypes(include='number').columns)

    model = SimpleMA(df, target_column)
    result = model.apply()
    st.line_chart(result[[target_column, 'SMA']])
