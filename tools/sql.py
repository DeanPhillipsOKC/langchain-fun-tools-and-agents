import sqlite3
from langchain.tools import Tool
from pydantic.v1 import BaseModel
from typing import List

conn = sqlite3.connect("db.sqlite")

def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occured: {str(err)}"

# We use Pydantic here to add annotiations to our tool description so that ChatGPT
# will understand how to use it better.  If we do not do this, the tool description
# that ends up getting passed will use a default name of __arg1 which isn't very descriptive
# or helpful.
class RunQueryArgsSchema(BaseModel):
    query: str

run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query.",
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema
)

def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return '\n'.join(row[0] for row in rows if row[0] is not None)

class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]

describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Give a list of table names, returns the schema of those tables",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema
)