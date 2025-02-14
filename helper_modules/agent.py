from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
import re

load_dotenv()

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    verbose=False
    # other params...
    )

    


def query2(query:str, df:pd.DataFrame):
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
                        0) Answer in two lines only dont give long explainations just give the key insights with small description
                        1)Answer the user query and please prefer tables to show the facts,
                        2)Also include observable trends between variables, necessary relations and patterns 
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
                        0) Answer in two lines only dont give long explainations just give the key insights with small description
                        {instructions}
                        1) Dont give any code in output
                        2) if the user asks for modelling results then use this helper_modules/modeling_results.json.
                    """
                    
                )
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    messages = chat_template.format_messages(text=query)


    ans=agent_executor.invoke(messages)
    return ans['output']

def clean_summary(response: str) -> str:
    # Remove the <think>...</think> block
    cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    return cleaned_response

def summarize_text(review: str) -> str:
    """
    Summarizes the given review in 2-3 sentences using a language model.

    Args:
        review (str): The review text to summarize.

    Returns:
        str: The summarized text.
    """
    # Define the chat template with a system message and human message
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    """You are an expert summarization assistant.
                    Your task is to summarize the given review in **2-3 sentences**.
                    Keep it **concise, clear, and informative**.
                    Do not generate code, and do not provide explanations.
                    Just give the summary directly without any additional text or reasoning.
                    """
                )
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )

    # Format the messages with the review text
    messages = chat_template.format_messages(text=review)

    try:
        # Invoke the language model to generate the summary
        ans = llm.invoke(messages)
        # Extract the content from the ChatResult object
        summary = ans.content  # Access the generated text directly
        summary=clean_summary(summary)
        return summary
    except Exception as e:
        # Handle any errors that occur during LLM invocation
        print(f"An error occurred while summarizing the review: {e}")
        return "Unable to generate a summary due to an error."