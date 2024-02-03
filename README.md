# Introduction 
Generates a snippet in any language along with a unit test in that same language.  This is just a hands on exercise using langchain to communicate
with LLMs for simple code generation.

# Prerequisites
- Python 3.x
- An Open AI API key

# Setup
## Unix / Linux / Mac
From the root of this repository run the following commands...
1. `python -m venv .venv` to create your virtual environment.
2. `source ./.venv/bin/activate` to start a Python virtual environment.
3. `pip intall -r requirements.txt` to install the required dependencies into the virtual environment.

### Required environment variables
You will need to create a `.env` file in the root of this repository and add your OpenAI API key.  This file is ignored by Git and will not get
added to source control where your API key might get leaked.

```
OPENAI_API_KEY=MY-API-KEY
```

## Windows
TBD

