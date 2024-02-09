from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from tools.sql import run_query_tool, list_tables

load_dotenv()

chat = ChatOpenAI()

tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=f"You are an AI that has access to a SQLite datbase.\n{tables}"),
        HumanMessagePromptTemplate.from_template("{input}"),
        # Will look for an input with the name "agent_scratchpad".
        # the agent scratchpad is basically just memory that remembers the initial human message,
        # any AI responses, function call results, etc... that it can pass during subsequant requests
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

tools = [run_query_tool]

# Basically a chain with all the normal things like input and prompts but adds available
# tools to the request that goes to OpenAI
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

# Takes a chain (in the form of an agent) and calls the OpenAI API over and over
# until it gets a response that is not a request to call a tool.
agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools
)

agent_executor("How many uses have added addresses?")