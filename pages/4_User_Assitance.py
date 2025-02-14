import streamlit as st
from helper_modules.agent import query, query2
from helper_modules.visualAgent import generate_code, get_fig_from_code
import re

st.title("Chat Assistance")

# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'output1' not in st.session_state:
    st.session_state.output1 = ""
if 'visual_query' not in st.session_state:
    st.session_state.visual_query = ""
if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'output2' not in st.session_state:
    st.session_state.output2 = ""
if 'dataset' not in st.session_state:
    st.session_state.dataset = None  # Ensure dataset is initialized

df = st.session_state.get('dataset')

if df is None:
    st.warning("Please upload a dataset on the Home page first.")
else:
    # Query Submission
    user_input = st.text_input("Enter your query")
    if st.button("Submit", key="query_submit"):
        if df is not None:
            with st.spinner("Processing your query..."):
                output = query(user_input, df)
                st.session_state.output1 = output
                st.write(output)
        else:
            st.error("Dataset not found! Please upload or set a dataset.")

    # Visualization Submission
    visual_query = st.text_input("Describe your visualization")
    if st.button("Submit", key="visual_submit"):
        if df is not None:
            with st.spinner("Generating visualization..."):
                response = generate_code(visual_query)
                code_block_match = re.search(r'```(?:[Pp]ython)?(.*?)```', response, re.DOTALL)

                try:
                    if code_block_match:
                        code_block = code_block_match.group(1).strip()
                        cleaned_code = re.sub(r'(?m)^\s*fig\.show\(\)\s*$', '', code_block)
                        fig = get_fig_from_code(cleaned_code)

                        if fig:
                            st.session_state.fig = fig
                            st.plotly_chart(fig, key="visualization_chart")

                            output = query(fig, df)
                            st.session_state.output2 = output
                            st.write(output)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
                # Detailed Explanation
                # if st.button("Detailed Explanation"):
                #     if df is not None and st.session_state.fig is not None:
                #         with st.spinner("Generating detailed explanation..."):
                #             output = query2(st.session_state.fig, df)
                #             st.write(output)
                #     else:
                #         st.error("No dataset or figure found!")
        else:
            st.error("Dataset not found! Please upload or set a dataset.")


