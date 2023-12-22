# flake8: noqa
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES") below the final answer.
ALWAYS return a "SOURCES" field in your answer, with the format "SOURCES: <source1>, <source2>, <source3>, ...".
If you cannot find the answer from the extracted parts of a long document, just say you don't know, never try to make up an answer. 
You can do common chit chat for example when the question is about greeting, you will greet back, if the question is saying sorry, you will say no worries,
if the question about thanks, you will say welcome.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

BASIC_PROMPT = ChatPromptTemplate(
        messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),]
    ) 

PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)


FILE_QUERY = """\
1. Upload a .pdf, .txt, or .docx file
2. Ask any question about the file!
"""

BASIC_QUERY = """\
Hi! Feel free to ask any questions~
"""

WELCOMINGS = """\
👋 Hello there! Welcome to Pandai Chat~
We are delighted to have you here. Pandai Chat is your intelligent companion designed to assist you effortlessly. Whether you seek engaging conversation or require assistance with file-related tasks, our chatbot is here to cater to your needs.
\nTo get started, kindly choose from the following options: 
"""

