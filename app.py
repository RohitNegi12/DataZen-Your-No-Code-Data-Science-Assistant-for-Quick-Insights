import dash
from dash import dcc, html, Input, Output, dash_table, State
import pandas as pd
import io
import base64
import os
import plotly.express as px
from dash_insights import numerical_insights, categorical_numerical_info, time_series_info, outliers
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans
import numpy as np
from agent import query


# Specify the directory where files will be saved
UPLOAD_DIRECTORY = "uploaded_files"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Initialize the Dash app
app = dash.Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.CYBORG])

# Layout of the app
app.layout = html.Div([
    # Sidebar
    html.Div([
        html.H2("Navigation", style={'text-align': 'center'}),
        dcc.Link('Home', href='/home', style={
            'padding': '10px',
            'display': 'block',
            'text-decoration': 'none',
            'color': '#ffffff',
            'background-color': 'black',
            'border-radius': '5px',
            'margin': '5px 0',
            'text-align': 'center'
        }),
        dcc.Link('Analysis', href='/analysis', style={
            'padding': '10px',
            'display': 'block',
            'text-decoration': 'none',
            'color': '#ffffff',
            'background-color': 'black',
            'border-radius': '5px',
            'margin': '5px 0',
            'text-align': 'center'
        }),
        dcc.Link('Settings', href='/settings', style={
            'padding': '10px',
            'display': 'block',
            'text-decoration': 'none',
            'color': '#ffffff',
            'background-color': 'black',
            'border-radius': '5px',
            'margin': '5px 0',
            'text-align': 'center'
        }),
    ], style={
        'width': '20%',
        'height': '100vh',
        'background-color': '#f9f9f9',
        'padding': '10px',
        'position': 'fixed'
    }),
    
    # Main content area
    html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content', style={'margin-left': '25%', 'padding': '20px'})
    ]),
    # Store for data and visualizations
    dcc.Store(id='stored-data', data=None),
    dcc.Store(id='selected-columns', data=None),
])

# Home page layout
home_layout = html.Div([
    html.H1("CSV File Upload"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a CSV File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False  # Restrict to CSV files only
    ),
    html.Div(id='output-data-upload'),
    html.Div(id='dropdown-container', style={'margin-top': '20px'}),
    html.Div(id='column-types', style={'margin-top': '20px'}),
    html.Div(id='graph-output'),
    html.Br(),
    # html.Div(id="column-dropdown"),
    # html.Div(id="dist-column-dropdown"),
    # html.Div(id="numeric-dropdown"),
    # html.Div(id="upload-data"),
    html.Div(
        # [
        #     dcc.Input(
        #         id="input-text",
        #         type="text",
        #         placeholder="Ask Your Query to the LLM ",
        #         className="form-control"
        #     )
        # ]
        # + [html.Button('Submit', id='submit-val', n_clicks=0, 
        #     className="btn btn-light"
        # )]
        # + [html.Div(id="llm-output")]
        # ,
        className="input-group mb-3",
        id="llm-input-container",
    )
    ,
    html.Div(id='dist-dropdown-container', style={'margin-top': '20px'}), #distribution dropdown
    html.Div(id='dist-graph-output'), #ditribution graph
    html.Div(id='clustering-plot-output')
])

# Analysis page layout
analysis_layout = html.Div([
    html.H1("Analysis Page"),
    html.Div(id='analysis-output'),  # Placeholder for displaying analysis results
    html.Div(id='numerical-insights-output'),
    html.Div(id='cat-num-insights-output'),
    html.Div(id='time-series-insights-output'),
    html.Div(id='outliers-insights-output'),
])

# Settings page layout
settings_layout = html.Div([
    html.H1("Settings Page"),
    html.P("This is where you can add your settings options.")
])


@app.callback(
        Output("llm-output","children"),
        Input("submit-val","n_clicks"),
        State("input-text","value"),
        State("stored-data","data"),
        prevent_initial_call=True
)
def answer_query(_,user_query:str, stored_data:dict):
    df=pd.DataFrame(stored_data)
    output=query(user_query,df)
    return output

# @app.callback(
#     Output('numeric-dropdown', 'options'),
#     Input('stored-data', 'data')
# )
# def update_dropdown_options(stored_data):
#     if stored_data is not None:
#         df = pd.DataFrame(stored_data)
#         numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
#         return [{'label': col, 'value': col} for col in numeric_cols]
#     return []




# Callback to display clustering plot based on dropdown selection
@app.callback(
    Output('clustering-plot-output', 'children'),
    Input('stored-data', 'data'),  # Trigger this callback when data is available
    Input('numeric-dropdown', 'value')  # Get selected columns from dropdown
)
def display_clustering_plot(stored_data, selected_columns):
    if stored_data is not None and selected_columns and len(selected_columns) >= 2:
        df = pd.DataFrame(stored_data)
        
        # Selecting only the chosen numeric columns
        numeric_df = df[selected_columns]
        
        # Performing clustering using KMeans
        kmeans = KMeans(n_clusters=3)
        numeric_df['Cluster'] = kmeans.fit_predict(numeric_df)

        # Create scatter plot with clusters
        fig = px.scatter(numeric_df, x=numeric_df.columns[0], y=numeric_df.columns[1], color='Cluster',
                         title="Clustering Plot", hover_data=numeric_df.columns)
        return dcc.Graph(figure=fig)

    return html.Div("Select at least two numeric columns for clustering plot.")



# Callback to render the appropriate layout based on URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/analysis':
        return analysis_layout
    elif pathname == '/settings':
        return settings_layout
    else:  # Default to home page
        return home_layout

# Callback to handle file upload, save the file, and display the content
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('dropdown-container', 'children'),
     Output('dist-dropdown-container','children'),
     Output('stored-data', 'data'),
     Output('llm-input-container','children')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('upload-data', 'last_modified'),
     Input('stored-data', 'data')],
      prevent_initial_call=True 
)
def update_output(content, filename, date, stored_data):
    
    if stored_data is not None and content is None:
        # If data is already stored, display it without re-uploading
        df = pd.DataFrame(stored_data)
        numerical=df.select_dtypes(include=["int64","float64"])
        # Ensure datetime conversion is applied to previously uploaded data
        # if 'Year' in df.columns:
        #     df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')
        # if 'Month' in df.columns:
        #     df['Month'] = pd.to_datetime(df['Month'], format='%m', errors='coerce')
        # if 'Day' in df.columns:
        #     df['Day'] = pd.to_datetime(df['Day'], format='%d', errors='coerce')
        
        # Display the DataFrame using dash_table.DataTable
        table = html.Div([
            html.H5(f"Previously uploaded: {filename}"),
            html.H4("Data Preview:"),
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            )
        ])
        
        # Dropdown for column selection
        dropdown = html.Div([
            html.H4("Select Columns for Visualization:"),
            dcc.Dropdown(
                id='column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                multi=True,
                placeholder="Select columns"
            )
        ])

        # Dropdown for column selection
        dist_dropdown = html.Div([
            html.H4("Select Numerical Columns for plotting Distribution:"),
            dcc.Dropdown(
                id='dist-column-dropdown',
                options=[{'label': col, 'value': col} for col in numerical.columns],
                multi=True,
                placeholder="Select columns"
            )
        ])

        llm_input=html.Div([dcc.Input(
                id="input-text",
                type="text",
                placeholder="Ask Your Query to the LLM ",
                className="form-control"
            )
        ]
        + [html.Button('Submit', id='submit-val', n_clicks=0, 
            className="btn btn-light"
        )]
        + [html.Div(id="llm-output")],className="input-group mb-3",)
        return table, dropdown, dist_dropdown, stored_data,llm_input
    
    if content is not None:
        # Process new file upload
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)
        
        # Save the file and read into a DataFrame
        try:
            with open(file_path, "wb") as f:
                f.write(decoded)
            df = pd.read_csv(file_path)
            numerical=df.select_dtypes(include=["int64","float64"])
            # Data type conversions
            # Ensure datetime conversion is applied to previously uploaded data
            # if 'Year' in df.columns:
            #     df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')
            # if 'Month' in df.columns:
            #     df['Month'] = pd.to_datetime(df['Month'], format='%m', errors='coerce')
            # if 'Day' in df.columns:
            #     df['Day'] = pd.to_datetime(df['Day'], format='%d', errors='coerce')

            print(df.dtypes)
            
            # Display the DataFrame
            table = html.Div([
                html.H5(f"Uploaded and saved as: {filename}"),
                html.H6(f'Last modified: {date}'),
                html.H4("Data Preview:"),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                )
            ])
            
            # Dropdown for column selection
            dropdown = html.Div([
                html.H4("Select Columns for Visualization:"),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    multi=True,
                    placeholder="Select columns"
                )
            ])
            dist_dropdown = html.Div([
                html.H4("Select Numerical Columns for plotting Distribution:"),
                dcc.Dropdown(
                    id='dist-column-dropdown',
                    options=[{'label': col, 'value': col} for col in numerical.columns],
                    multi=True,
                    placeholder="Select columns"
                )
            ])

            llm_input=html.Div([dcc.Input(
                id="input-text",
                type="text",
                placeholder="Ask Your Query to the LLM ",
                className="form-control"
            )
            ]
            + [html.Button('Submit', id='submit-val', n_clicks=0, 
                className="btn btn-light"
            )]
            + [html.Div(id="llm-output")],className="input-group mb-3")

            return table, dropdown, dist_dropdown, df.to_dict('records'), llm_input
        
        except Exception as e:
            return html.Div([f'There was an error processing this file: {str(e)}']), None, None, None, None

    return html.Div(), None, None, None, None

# Callback for visualization based on selected columns and showing column types
@app.callback(
    [Output('graph-output', 'children'),
     Output('column-types', 'children')],
    Input('column-dropdown', 'value'),
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_graph(selected_columns,  stored_data):
    if selected_columns is None or len(selected_columns) != 2 or stored_data is None:
        return html.Div("Please select exactly two columns for visualization."), html.Div()
    
  
    df = pd.DataFrame(stored_data)  # Convert stored JSON back to DataFrame
    print(df.dtypes)
    # if 'Year' in df.columns:
    #     df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')
    # if 'Month' in df.columns:
    #     df['Month'] = pd.to_datetime(df['Month'], format='%m', errors='coerce')
    # if 'Day' in df.columns:
    #     df['Day'] = pd.to_datetime(df['Day'], format='%d', errors='coerce')
    print(df)
    print(df.dtypes)
    x_col = selected_columns[0]
    y_col = selected_columns[1]

    # Determine the type of each column (numeric, categorical, datetime)
    # x_dtype = pd.api.types.infer_dtype(df[x_col])
    # y_dtype = pd.api.types.infer_dtype(df[y_col])
    x_dtype = df[x_col].dtype
    y_dtype = df[y_col].dtype

    print(x_dtype, y_dtype)
    
    # Create a div to show the types of the selected columns
    types_info = html.Div([
        html.H5("Selected Column Types:"),
        html.P(f"Column {x_col} is of type: {x_dtype}"),
        html.P(f"Column {y_col} is of type: {y_dtype}")
    ])
    
    # Visualization logic
    # Scatter plot if both columns are numeric
    if (x_col == 'Month' and y_dtype in ['int64', 'float64']) or (x_dtype in ['int64', 'float64'] and y_col == 'Month'):
        # df['Year'] = df['date'].dt.year
        # df['Month'] = df['date'].dt.month
        

        if x_col == 'Month':
            monthly_avg = df.groupby(['Month'])[y_col].mean().reset_index()
            fig = px.line(monthly_avg, x=x_col, y=y_col, title=f'Line Chart: {y_col} over Time')
        else:
            monthly_avg = df.groupby(['Month'])[x_col].mean().reset_index()
            fig = px.line(monthly_avg, x=y_col, y=x_col, title=f'Line Chart: {x_col} over Time')

    elif (x_col == 'Year' and y_dtype in ['int64', 'float64']) or (x_dtype in ['int64', 'float64'] and y_col == 'Year'):
        
        if x_col == 'Year':
            yearly_avg = df.groupby(['Year'])[y_col].mean().reset_index()
            fig = px.line(yearly_avg, x=x_col, y=y_col, title=f'Line Chart: {y_col} over Time')
        else:
            yearly_avg = df.groupby(['Year'])[x_col].mean().reset_index()
            fig = px.line(yearly_avg, x=y_col, y=x_col, title=f'Line Chart: {x_col} over Time')

    elif x_dtype in ['int64', 'float64'] and y_dtype in ['int64', 'float64']:
        fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter Plot: {x_col} vs {y_col}')
    
    # Bar chart if one column is numeric and the other is categorical
    elif (x_dtype in ['int64', 'float64'] and y_dtype in ['category', 'object']) or (x_dtype in ['category', 'object'] and y_dtype in ['int64', 'float64']):
        if x_dtype in ['category', 'object']:
            fig = px.bar(df, x=x_col, y=y_col, title=f'Bar Chart: {x_col} vs {y_col}')
        else:
            fig = px.bar(df, x=y_col, y=x_col, title=f'Bar Chart: {y_col} vs {x_col}')
    
    # Line chart for time series data (datetime column)
    # elif (x_dtype == 'datetime64[ns]' and y_dtype in ['int64', 'float64']) or (x_dtype in ['inte64', 'float64'] and y_dtype == 'datetime64[ns]'):
    #     if x_dtype == 'datetime64[ns]':
    #         fig = px.line(df, x=x_col, y=y_col, title=f'Line Chart: {y_col} over Time')
    #     else:
    #         fig = px.line(df, x=y_col, y=x_col, title=f'Line Chart: {x_col} over Time')
    

    
    else:
        return html.Div("Unable to generate a visualization with the selected columns."), types_info
    

    return dcc.Graph(figure=fig), types_info

@app.callback(
        Output('dist-graph-output','children'),
        Input('dist-column-dropdown','value'),
        Input('stored-data','data'),
        prevent_initial_call=True )
def update_dist_graph(selected_column, stored_data):

    if selected_column is None or len(selected_column) != 1 or stored_data is None:
        return html.Div("Please select exactly one column for visualization."), html.Div()
  
    df = pd.DataFrame(stored_data)  # Convert stored JSON back to DataFrame
    fig = px.histogram(df, 
                   x=selected_column, 
                   title=f'Histogram of {selected_column}', 
                   labels=selected_column,
                   nbins=10)  # You can adjust the number of bins
    return dcc.Graph(figure=fig)
# Callback for displaying the uploaded data in the Analysis page


@app.callback(
    Output('analysis-output', 'children'),
    Input('stored-data', 'data')
)
def display_analysis_data(stored_data):
    if stored_data is None:
        return html.Div("No data available for analysis.")
    
    df = pd.DataFrame(stored_data)  # Convert stored JSON back to DataFrame
    return html.Div([
        html.H4("Uploaded Data for Analysis:"),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
        )
    ])

# Callback for displaying numerical information from num_info
@app.callback(
    Output('numerical-insights-output', 'children'),
    Input('stored-data', 'data')  # Trigger this callback when data is available
)
def display_num_info(stored_data):
    if stored_data is not None:
        df = pd.DataFrame(stored_data)
        info_list = numerical_insights(df)  # Call the num_info function to get insights
        # Compute the correlation matrix
        num_col = df.select_dtypes(include=['float64', 'int64'])
        corr_matrix = num_col.corr()
        # Generate a heatmap using Plotly for the correlation matrix
        fig = px.imshow(
            corr_matrix,
            text_auto=True,  # Display correlation values on the heatmap
            labels={'color': 'Correlation'},
            title="Correlation Matrix"
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))  # Add some padding around the plot
        return html.Div([
            html.H4("Numerical Information:"),
            # html.H4("Correlation Matrix:"),
            dcc.Graph(figure=fig),
            html.Ul([html.Li(info) for info in info_list]),  # Display the insights as a list
            
        ])
    return html.Div("No numerical data available.")


@app.callback(
    Output('cat-num-insights-output', 'children'),
    Input('stored-data', 'data')  # Trigger this callback when data is available
)
def display_cat_num_info(stored_data):
    if stored_data is not None:
        df = pd.DataFrame(stored_data)
        # html.H5("Categorical vs Numerical info")
        cat_num_dict = categorical_numerical_info(df)  # Your function that returns insights dictionary

        insights_with_viz = [html.H3("Numerical vs Categorical Information")]  # List to store insights with visualizations
        # Iterate over the dictionary where keys are tuples of two columns
        # for column_tuple, insight in cat_num_dict.items():
        #     col1, col2 = column_tuple  # Unpack the tuple to get column names

        #     # Generate a bar plot for the two columns
        #     fig = px.bar(df, x=col1, color=col2, title=f"Bar Plot of {col1} grouped by {col2}")
        for column_tuple, insight in cat_num_dict.items():
            col1, col2 = column_tuple  # Unpack the tuple to get column names
            new_df = df.groupby(col1)[col2].mean().reset_index()
            # Generate a bar plot for the two columns
            fig = px.bar(new_df, x=col1, y=col2, title=f"Bar Plot of {col1} grouped by {col2}")

            # Append the insight and visualization to the list
            insights_with_viz.append(
                html.Div([
                    html.H4(f"Insight for {col1} and {col2}:"),
                    html.P(insight),  # Display the insight text
                    dcc.Graph(figure=fig)  # Display the bar plot
                ])
            )
        
        # Return all insights and visualizations
        return html.Div(insights_with_viz)

    return html.Div("No data available.")

@app.callback(
    Output('time-series-insights-output', 'children'),
    Input('stored-data', 'data')  # Trigger this callback when data is available
)
def display_time_series_info(stored_data):
    if stored_data is not None:
        df = pd.DataFrame(stored_data)
        temp_df = df
        if temp_df.get('Year') is not None:
            temp_df['Year'] = pd.to_datetime(temp_df['Year'], format='%Y')
        if temp_df.get('Month') is not None:
            temp_df['Month'] = pd.to_datetime(temp_df['Month'], format='%m')
        if temp_df.get('Day') is not None:
            temp_df['Day'] = pd.to_datetime(temp_df['Day'], format='%d')
        # html.H5("Categorical vs Numerical info")
        time_series_dict = time_series_info(temp_df)  # Your function that returns insights dictionary

        insights_with_viz = [html.H3("Time Series Information")]  # List to store insights with visualizations

        # Iterate over the dictionary where keys are tuples of two columns
        for column_tuple, insight in time_series_dict.items():
            col1, col2 = column_tuple  # Unpack the tuple to get column names

            avg_df = df.groupby([col1])[col2].mean().reset_index()
            # Generate a bar plot for the two columns
            fig = px.line(avg_df, x=col1, y=col2, title=f'Line Chart: {col2} over Time')

            # Append the insight and visualization to the list
            insights_with_viz.append(
                html.Div([
                    html.H4(f"Insight for {col1} and {col2}:"),
                    html.P(insight),  # Display the insight text
                    dcc.Graph(figure=fig)  # Display the bar plot
                ])
            )
        
        # Return all insights and visualizations
        return html.Div(insights_with_viz)

    return html.Div("No data available.")

@app.callback(
    Output('outliers-insights-output', 'children'),
    Input('stored-data', 'data')  # Trigger this callback when data is available
)
def display_cat_num_info(stored_data):
    if stored_data is not None:
        df = pd.DataFrame(stored_data)
        # html.H5("Categorical vs Numerical info")
        outliers_dict = outliers(df)  # Your function that returns insights dictionary

        insights_with_viz = [html.H3("Outliers Information")]  # List to store insights with visualizations

        # Iterate over the dictionary where keys are tuples of two columns
        for column_tuple, insight in outliers_dict.items():
            col1= column_tuple  # Unpack the tuple to get column names

            # Generate a bar plot for the two columns
            fig = px.box(df, y=col1)

            # Append the insight and visualization to the list
            insights_with_viz.append(
                html.Div([
                    html.H4(f"Insight for {col1}:"),
                    html.P(insight),  # Display the insight text
                    dcc.Graph(figure=fig)  # Display the bar plot
                ])
            )
        
        # Return all insights and visualizations
        return html.Div(insights_with_viz)

    return html.Div("No data available.")




# @app.callback(
#     Output('numerical-insights-output', 'children'),
#     Input('stored-data', 'data')  # Trigger this callback when data is available
# )
# def display_num_info(stored_data):
#     if stored_data is not None:
#         df = pd.DataFrame(stored_data)
        
#         # Debug: Check if the data is loaded correctly
#         print("Dataframe Loaded:", df.head())  # For checking data in the console
        
#         # Ensure dataframe has numerical columns for correlation matrix
#         if df.select_dtypes(include=[float, int]).empty:
#             return html.Div("No numerical data available for correlation matrix.")
#         else:
#             print("num col available.")
        
#         # Get numerical insights
#         print("cal insights...")
#         info_list = numerical_insights(df)  # Call the num_info function to get insights

#         numerical_df = df.select_dtypes(include=['float64', 'int64'])

#         # Compute the correlation matrix (only for numerical columns)
#         print("cal_corr...")
#         corr_matrix = numerical_df.corr()
#         print("success cal_corr")

#         # Debug: Check if the correlation matrix is computed correctly
#         print("Correlation Matrix:", corr_matrix)

#         # Generate a heatmap using Plotly for the correlation matrix
#         fig = px.imshow(
#             corr_matrix,
#             text_auto=True,  # Display correlation values on the heatmap
#             labels={'color': 'Correlation'},
#             title="Correlation Matrix"
#         )
#         fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))  # Add some padding around the plot

#         return html.Div([
#             html.H4("Numerical Information:"),
#             html.Ul([html.Li(info) for info in info_list]),  # Display the insights as a list
#             html.H4("Correlation Matrix:"),
#             dcc.Graph(figure=fig)  # Display the correlation matrix plot
#         ])
    
#     # Return default message if no data is available
#     return html.Div("No numerical data available.")


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,dev_tools_ui=False,dev_tools_props_check=False)
