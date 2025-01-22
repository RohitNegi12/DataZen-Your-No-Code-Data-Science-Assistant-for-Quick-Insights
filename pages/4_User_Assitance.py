import streamlit as st
from helper_modules.agent import query
from helper_modules.visualAgent import generate_code,get_fig_from_code
import re
st.title("Chat Assistance")

user_input=st.text_input("Enter your query")

df = st.session_state.get('dataset')
if st.button("Submit", key="query_submit"):
    output=query(user_input,df)
    st.write(output)
    

visual_query=st.text_input("Describe your visualization")
if st.button("Submit", key="visual_submit"):
    response=generate_code(visual_query)
    code_block_match=re.search(r'```(?:[Pp]ython)?(.*?)```',response,re.DOTALL)

    if code_block_match:
        code_block=code_block_match.group(1).strip()
        cleaned_code=re.sub(r'(?m)^\s*fig\.show\(\)\s*$','',code_block)
        print(cleaned_code)
        fig= get_fig_from_code(cleaned_code)
        
    try:
        fig
    except Exception as e:
        st.error("Try again and please describe it clearly")
        # st.exception(e)


