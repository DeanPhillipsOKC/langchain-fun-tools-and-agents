from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel

class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str

def write_report(filename, html):
    with open(filename, 'w') as f:
        f.write(html)

# Whe are using a structured tool here because we need to pass multiple args
write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write an HTML file to disk.  Ise this tool whenever someone asks for a report.",
    func=write_report,
    args_schema=WriteReportArgsSchema
)