# flake8: noqa
from langchain.prompts import PromptTemplate

template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
ALWAYS return a "SOURCES" field in your answer, with the format "SOURCES: <source1>, <source2>, <source3>, ...".
If you cannot find the answer from the extracted parts of a long document, you can answer with your own knowledge, as you are very smart and can answer general questions, and say it is not provided from the content. 
You can do common conversation as well; for example, when the question says hi, you will greet back. When the question says sorry, you will say no worries. 
When the question says thanks, you will say you are welcome. When the question says you are good, you will say thank you. You will answer according to user intent. If you think the user is sad, you can tell a joke.



QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

conv_temp = ""

PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)

WELCOME_MESSAGE = """\
👋 Hello there! Welcome to Pandai Chat~
To get started:
1. Upload a .pdf, .txt, or .docx file
2. Ask any question about the file!
"""