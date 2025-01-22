import pandas as pd
from visualAgent import generate_code,get_fig_from_code
from dotenv import load_dotenv
import re

df=pd.read_csv('test.csv')
response=generate_code("draw a bar graph")
code_block_match=re.search(r'```(?:[Pp]ython)?(.*?)```',response,re.DOTALL)

if code_block_match:
    code_block=code_block_match.group(1).strip()
    cleaned_code=re.sub(r'(?m)^\s*fig\.show\(\)\s*$','',code_block)
    fig= get_fig_from_code(cleaned_code)
    print(fig)