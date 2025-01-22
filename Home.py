import streamlit as st
import pandas as pd

st.set_page_config(
    page_title = "DataZen",
    page_icon = "🍃"
)

# Initialize session state to persist data
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'cleaned_dataset' not in st.session_state:
    st.session_state['cleaned_dataset'] = None

st.title("DataZen")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    with open("helper_modules/test.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
        print("File Saved")
    st.session_state['dataset'] = pd.read_csv(uploaded_file)
    st.session_state['cleaned_dataset'] = st.session_state['dataset'].copy()
    st.success("Dataset uploaded successfully!")

if st.session_state['dataset'] is not None:
    st.subheader("Dataset Preview")
    st.write(f"{st.session_state['dataset'].shape[0]} records")
    st.dataframe(st.session_state['dataset'].head(100), height=200)