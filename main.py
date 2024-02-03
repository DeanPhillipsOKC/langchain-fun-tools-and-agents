from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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

prompt_template = PromptTemplate(
    template="Create a {language} function that accomplishes the following task: {task}",
    input_variables=["language", "task"]
)

code_generation_chain = LLMChain(
    prompt=prompt_template,
    llm=llm
)

result = code_generation_chain({
    "language": args.language,
    "task": args.task
})

print(result["text"])