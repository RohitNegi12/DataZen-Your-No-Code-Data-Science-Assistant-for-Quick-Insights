import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Then import normally
from helper_modules.dash_insights import categorical_numerical_info, numerical_insights, time_series_info, outliers


st.title("Analysis Page")
data = st.session_state.get('dataset')

if data is None:
    st.warning("Please upload a dataset on the Home page first.")
else:

    # Numerical Insights Section
    st.title("Numerical Insights")

    # Compute the correlation matrix for all numerical columns
    def get_correlation_matrix(df):
        numerical_cols = df.select_dtypes(include=['number']).columns
        return df[numerical_cols].corr()

    correlation_matrix = get_correlation_matrix(data)
    correlation_insights = numerical_insights(data)

    # Plot the correlation matrix
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title="Correlation")
    ))
    fig_corr.update_layout(title="Correlation Matrix", xaxis_title="Columns", yaxis_title="Columns")

    # Display the correlation matrix and insights
    st.plotly_chart(fig_corr)
    st.write("### Correlation Insights:")
    for insight in correlation_insights:
        st.write(f"- {insight}")

    numeric_cols = data.select_dtypes(include=['number']).columns
    distribution_col = st.selectbox("Select columns (Data distribution):", numeric_cols)
    if distribution_col:
        st.subheader(f"Distribution of '{distribution_col}'")

        # Plot the distribution using Plotly
        fig = px.histogram(
            data_frame=data,
            x=distribution_col,
            nbins=30,  # You can adjust the number of bins
            title=f"Distribution of {distribution_col}",
            labels={distribution_col: distribution_col},  # Axis labels
            color_discrete_sequence=['#636EFA']  # Customize the color
        )

        # Add density curve
        fig.update_layout(bargap=0.1)
        fig.update_traces(opacity=0.75)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    col_x = st.selectbox("Select X-axis Numeric Column:", numeric_cols, key="col_x")
    col_y = st.selectbox("Select Y-axis Numeric Column:", numeric_cols, key="col_y")

    # Check if both columns are selected
    if col_x and col_y:
        st.subheader(f"Distribution Between '{col_x}' and '{col_y}'")

        # Plot the scatter plot using Plotly
        fig = px.scatter(
            data_frame=data,
            x=col_x,
            y=col_y,
            title=f"Scatter Plot: {col_x} vs {col_y}",
            labels={col_x: col_x, col_y: col_y},  # Axis labels
            color_discrete_sequence=['#EF553B'],  # Customize the color
            opacity=0.75  # Set transparency for better visualization
        )

        # Update layout for better appearance
        fig.update_layout(
            xaxis_title=col_x,
            yaxis_title=col_y,
            template="plotly_white"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)


    


    # tmp_df = data
    time_series_insights = time_series_info(data)
    print(time_series_insights)
    if(len(time_series_insights) > 0):
        # Time Series Insights Section
        st.title("Time Series Insights")

        
        # Sidebar for time series insights
        time_keys = list(time_series_insights.keys())
        selected_time_key = st.selectbox("Select columns (time series):", time_keys)

        time_column, numerical_column = selected_time_key
        time_insight = time_series_insights[selected_time_key]

        # Plotting the line chart
        avg_df = data.groupby([time_column])[numerical_column].mean().reset_index()
        fig_time = px.line(avg_df, x=time_column, y=numerical_column, 
                        title=f"{time_column} vs {numerical_column}", 
                        labels={time_column: time_column.capitalize(), 
                                numerical_column: numerical_column.capitalize()})

        # Display the graph and insight
        st.plotly_chart(fig_time)
        st.write(f"### Insight:\n{time_insight}")

    else:
        st.title("Insights from Categorical vs Numerical Columns")

        cat_num_insights = categorical_numerical_info(data)
        print(cat_num_insights)
        # Sidebar for selection
        keys = list(cat_num_insights.keys())
        selected_key = st.selectbox("Select columns (categorical vs numerical):", keys)

        categorical_column, numerical_column = selected_key
        insight = cat_num_insights[selected_key]

        # Plotting the graph
        # avg_df = data.groupby([categorical_column])[numerical_column].mean().reset_index()
        fig = px.bar(data, x=categorical_column, y=numerical_column, 
                    title=f"{categorical_column} vs {numerical_column}", 
                    labels={categorical_column: categorical_column.capitalize(), 
                            numerical_column: numerical_column.capitalize()})

        # Display the graph and insight
        st.plotly_chart(fig)
        st.write(f"### Insight:\n{insight}")

    

    # Outlier Insights Section
    st.title("Outliers")
    outlier_insights = outliers(data)

    # Plot all outliers in a grid fashion
    outlier_cols = list(outlier_insights.keys())
    fig_outliers = make_subplots(rows=1, cols=len(outlier_cols), subplot_titles=outlier_cols)

    for i, col in enumerate(outlier_cols, start=1):
        fig_box = px.box(data, y=col, title=col)
        for trace in fig_box.data:
            fig_outliers.add_trace(trace, row=1, col=i)

    fig_outliers.update_layout(title_text="Outliers for Numerical Columns", showlegend=False)

    # Display the outlier plots
    st.plotly_chart(fig_outliers)
    st.write("### Outlier Insights:")
    for col, insight in outlier_insights.items():
        st.write(f"- {insight}")
