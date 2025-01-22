import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans


st.set_page_config(
    page_title = "DataZen",
    page_icon = "🍃"
)

st.title("Analysis Page")
data = st.session_state.get('dataset')

if data is None:
    st.warning("Please upload a dataset on the Home page first.")
else:
    # Ensure the necessary session state keys exist
    if 'x_col' not in st.session_state:
        st.session_state['x_col'] = None
    if 'y_col' not in st.session_state:
        st.session_state['y_col'] = None
    if 'num_clusters' not in st.session_state:
        st.session_state['num_clusters'] = 3
    if 'plot_type' not in st.session_state:
        st.session_state['plot_type'] = None
    if 'selected_column' not in st.session_state:
        st.session_state['selected_column'] = None
    # if 'dist_col' not in st.session_state:
    #     st.session_state['dist_col'] = 0
    st.dataframe(data.head(100), height=210)

    # Select the type of plot
    st.selectbox("Select the type of plot", ["Scatter with Clusters", "Distribution Plot"],index=1,
            key="plot_type")

    # Scatter plot with clusters

    if "Scatter with Clusters" == st.session_state.get('plot_type'):
        try:
            # Persist X and Y column selections
            x_col = st.selectbox(
                "Select X-axis column",
                (data.select_dtypes(include='number')).columns,
                index=(data.select_dtypes(include='number')).columns.get_loc(st.session_state['x_col'])
                if st.session_state['x_col'] in data.columns else 0
            )
            st.session_state['x_col'] = x_col

            y_col = st.selectbox(
                "Select Y-axis column",
                (data.select_dtypes(include='number')).columns,
                index=(data.select_dtypes(include='number')).columns.get_loc(st.session_state['y_col'])
                if st.session_state['y_col'] in data.columns else 0
            )
            st.session_state['y_col'] = y_col

            # Persist the number of clusters
            num_clusters = st.slider(
                "Select number of clusters",
                min_value=2,
                max_value=10,
                value=st.session_state['num_clusters']
            )
            st.session_state['num_clusters'] = num_clusters

            # Perform clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            data['Cluster'] = kmeans.fit_predict(data[[x_col, y_col]])

            # Create scatter plot
            fig = px.scatter(
                data, x=x_col, y=y_col, color=data['Cluster'].astype(str),
                title="Scatter Plot with Clusters",
                labels={"color": "Cluster"}
            )
            st.plotly_chart(fig)

        except KeyError as e:
            st.error(f"KeyError: {e}")
        except ValueError as e:
            st.error(f"ValueError: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # Distribution plot
    elif st.session_state["plot_type"] == "Distribution Plot":
        col = st.selectbox(
            "Select column for distribution",
            list(data), # columns in a list
            index=0,
        )

        if data[col].dtype == 'object':
            # Generate a DataFrame for the bar chart
            value_counts_df = data[col].value_counts().reset_index()
            value_counts_df.columns = ['Category', 'Count']  # Rename columns for clarity
            
            # Plot the bar chart
            fig = px.bar(
                value_counts_df, 
                x="Category", y="Count", 
                title=f"Category Distribution of {col}"
            )
        else:
            # Plot histogram for numerical data
            fig = px.histogram(
                data, x=col, nbins=30, 
                title=f"Distribution of {col}"
            )
        
        # Display the chart
        st.plotly_chart(fig)
