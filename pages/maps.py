import streamlit as st
import pandas as pd
import numpy as np 
from nav import nav_page



st.set_page_config(
    page_title="Map",
    page_icon="👋",
)

st.write("# Monsters around Me")



df = pd.DataFrame(
    np.random.randn(5,2) / [60, 60] + [36.3721427, 127.360399],
    columns=['lat', 'lon'])


st.map(df)

if st.button("Fight"):
    st.button("Fight", on_click= nav_page("app"))