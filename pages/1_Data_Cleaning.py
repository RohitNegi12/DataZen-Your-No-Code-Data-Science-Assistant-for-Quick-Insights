import streamlit as st
import pandas as pd

# Function to remove outliers based on selected columns
def remove_outliers(df, columns):
    if not columns:
        return df
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Function to remove duplicates based on selected columns
def remove_duplicates(df, columns):
    if columns:
        df = df.drop_duplicates(subset=columns)
    else:
        df = df.drop_duplicates()
    return df

st.set_page_config(
    page_title = "DataZen",
    page_icon = "🍃"
)
st.header("Cleaning Options")

if st.session_state['dataset'] is None or st.session_state['cleaned_dataset'] is None:
    st.warning("Please upload a dataset on the Home page first.")
else:
    df = st.session_state['cleaned_dataset']
    outlier_columns = df.select_dtypes(include="number").columns.tolist()
    columns = df.columns.tolist()

    # Option to remove outliers
    st.subheader("Remove Outliers")
    outlier_columns = st.multiselect("Select columns to remove outliers", options=outlier_columns, default=[])
    if st.button("Remove Outliers"):
        if outlier_columns:
            st.session_state['cleaned_dataset'] = remove_outliers(df, outlier_columns)
            st.success("Outliers removed!")
        else:
            st.warning("Please select at least one column")

    # Option to remove duplicates
    st.subheader("Remove Duplicates")
    duplicate_columns = st.multiselect("Select columns to check for duplicates", options=columns, default=[])
    if st.button("Remove Duplicates"):
        if duplicate_columns:
            st.session_state['cleaned_dataset'] = remove_duplicates(df, duplicate_columns)
            st.success("Duplicates removed!")

    # Display cleaned data
    st.subheader("Cleaned Dataset Preview")
    st.write(f"{st.session_state['cleaned_dataset'].shape[0]} records")
    st.dataframe(st.session_state['cleaned_dataset'].head(100), height=200)

    # Option to save changes
    if st.button("Save Changes"):
        st.session_state['dataset'] = st.session_state['cleaned_dataset']
        st.success("Changes saved to the dataset!")
    if st.button("Discard Changes"):
        st.session_state['cleaned_dataset'] = st.session_state['dataset'].copy()
        st.info("Changes discarded.")
        st.rerun()