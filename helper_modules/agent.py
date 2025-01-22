from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
    )

    


def query(query:str, df:pd.DataFrame):
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        verbose=True
    )

    

    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    """You are a helpful data analyst and statistician.
                    Follow these rules:
                        1) Answer the user query and support your answer with facts
                        2) Also include observable trends between variables, necessary relations and patterns 
                        3) Dont give any code in output
                        4) if the user asks for modelling results then use this helper_modules/modeling_results.json.
                    """
                    
                )
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    messages = chat_template.format_messages(text=query)


    ans=agent_executor.invoke(messages)
    return ans['output']