from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import pandas as pd
from matplotlib import pyplot

load_dotenv()

df=pd.read_csv('helper_modules/test.csv')
df_5_rows=df.head()
csv_string=df_5_rows.to_string(index=False)

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
    )


prompt=chat_template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You're a data visualization expert and use your favourite graphing library Plotly only. Suppose, that "
             "the data is provided as as helper_modules/test.csv file. Here are the first five rows of the data set: {data}"
             "Follow the user's indications when creating graph"
             ),
             MessagesPlaceholder(variable_name="messages"),
        ]
    )

chain= prompt | llm

def get_fig_from_code(code):
    local_variables={}
    exec(code,{},local_variables)
    return local_variables['fig']

def generate_code(user_input):
    # Create a HumanMessagePromptTemplate object
    user_message = HumanMessage(content=user_input)
    
    # Invoke chain
    response = chain.invoke(
        {
            "messages": [user_message],
            "data": csv_string,
        }
    )
    return response.content

