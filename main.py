from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

# Set project environment variables
load_dotenv()

# Get the arguments
parser = argparse.ArgumentParser(
    prog="PyCode",
    description="Playing around with LangChain and the OpenAI API",
    epilog="Add a task and language argument, or just let it pick the default."
)

parser.add_argument("--task", default="return a list of numbers starting with 1 and ending with 10")
parser.add_argument("--language", default="python")

args = parser.parse_args()

llm = OpenAI()

code_generation_chain = LLMChain(
    prompt=PromptTemplate(
        template="Create a {language} function that accomplishes the following task: {task}",
        input_variables=["language", "task"]),
    llm=llm,
    output_key="code"
)

test_generation_chain = LLMChain(
    prompt=PromptTemplate(
        template="Create a unit test for the following {language} code:\n{code}",
        input_variables=["language", "code"]),
    llm=llm,
    output_key="test"
)

chain = SequentialChain(
    chains=[code_generation_chain, test_generation_chain],
    input_variables=["task", "language"],
    output_variables=["code", "test"]
)

result = chain({
    "language": args.language,
    "task": args.task
})

print(">>>>>> GENERATED CODE:")
print(result["code"])

print(">>>>>> GENERATED TEST:")
print(result["test"])