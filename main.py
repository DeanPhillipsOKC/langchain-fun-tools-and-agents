from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

handler = ChatModelStartHandler()
chat = ChatOpenAI(
    callbacks=[handler]
)

tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            f"You are an AI that has access to a SQLite datbase.\n"
            f"The database has tables of: {tables}\n"
            "Do not make any assumptions about what tables exist "
            "or what columns exist.  Instead, use the 'describe_tables'"
            "function"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        # Will look for an input with the name "agent_scratchpad".
        # the agent scratchpad is basically just memory that remembers the initial human message,
        # any AI responses, function call results, etc... that it can pass during subsequant requests.
        # The problem here is that once we get a response from ChatGPT that is not a function call, it
        # considers the job done and the agent_scratchpad intermediate steps get deleted.
        # This means that if you ask another question, ChatGPT will not have any context of the previous
        # stuff.
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

# return_messages tells it to return the results as message objects instead of stringss
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [run_query_tool, describe_tables_tool, write_report_tool]

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
    #verbose=True,
    tools=tools,
    memory=memory
)

agent_executor(
    "How many orders are there?")

#agent_executor(
#    "Repeat the exact same process for users"
#)