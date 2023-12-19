import chainlit as cl 
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
import PyPDF2
from docx import Document
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
#from gpt4all import GPT4All
import openai
from openai import OpenAI
import spacy
import os
import sys
from typing import Optional
import logging   
import time
import random
import pyttsx3

#set_api_key("fda3cd815581712c396688db1b9ee067")
#available_voices = voices()

llm = ChatOpenAI()
backoff_in_seconds = float(os.getenv("BACKOFF_IN_SECONDS", 3))
max_retries = int(os.getenv("MAX_RETRIES", 10))

logging.basicConfig(stream = sys.stdout,
                    format = '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def backoff(attempt : int) -> float:
    return backoff_in_seconds * 2**attempt + random.uniform(0, 1)

# local LLM but my computer sucks 
#gpt = GPT4All("gpt4all-falcon-q4_0.gguf")

def set_templates():
    system_template = """Use the following pieces of context to answer the users question.
    If you don't know the answer, just say you don't know, never try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    The "SOURCES" part should be a reference to the source of the document from which you got your answer.

    Example of your response should be:

    ```
    The answer is foo
    SOURCES: xyz
    ```

    Begin!
    ----------------
    {summaries}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}
    return chain_type_kwargs

def get_completion(question):
    
    client = OpenAI()
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role":"assistant",
                "content":question
            },
            {
                "role":"system",
                "content":"You are a helpful assistant, answer the questions given within a maximum of 1000 words"
            }
        ],
        temperature = 0.5,
    )
    return response.choices[0].message.content

def get_pdf_text(file):
    
    pdf_stream = BytesIO(file.content)
    extension = file.name.split('.')[-1]

    text = ''

    if extension == "pdf":
        reader = PyPDF2.PdfReader(pdf_stream)
        for i in range(len(reader.pages)):
            text +=  reader.pages[i].extract_text()
    elif extension  ==  "docx":
        doc = Document(pdf_stream)
        paragraph_list = []
        for paragraph in doc.paragraphs:
            paragraph_list.append(paragraph.text)
        text = '\n'.join(paragraph_list)
    
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200, #ensure full sentence
        length_function=len 
    )

    chunks = text_splitter.split_text(text) #list of chunks
    return chunks 


async def get_vectorstore(text_chunks, metadatas):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vstore = await cl.make_async(Chroma.from_texts)(
        text_chunks, embeddings, metadatas=metadatas
    )
    return vstore

def get_conversation_chain(vectorstore):
    #llm = huggingface_hub.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        #chain_type_kwargs = set_templates(),
    )
    return chain

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

#continuously on a loop
@cl.on_message
async def main(message: cl.Message):
    # get previous chain
    chain = cl.user_session.get("chain")  
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])  

    for attempt in range(max_retries):
        try:
            res = await chain.acall(message.content, 
                                    callbacks = [cl.AsyncLangchainCallbackHandler()])
            break
        # except openai.Timeout:
        #     # Implement exponential backoff
        #     wait_time = backoff(attempt)
        #     logger.exception(f"OpenAI API timeout occurred. Waiting {wait_time} seconds and trying again.")
        #     time.sleep(wait_time)
        # except openai.APIError:
        #     # Implement exponential backoff
        #     wait_time = backoff(attempt)
        #     logger.exception(f"OpenAI API error occurred. Waiting {wait_time} seconds and trying again.")
        #     time.sleep(wait_time)
        # except openai.APIConnectionError:
        #     # Implement exponential backoff
        #     wait_time = backoff(attempt)
        #     logger.exception(f"OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting {wait_time} seconds and trying again.")
        #     time.sleep(wait_time)
        # except openai.InvalidRequestError:
        #     # Implement exponential backoff
        #     wait_time = backoff(attempt)
        #     logger.exception(f"OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting {wait_time} seconds and trying again.")
        #     time.sleep(wait_time)
        # except openai.ServiceUnavailableError:
        #     # Implement exponential backoff
        #     wait_time = backoff(attempt)
        #     logger.exception(f"OpenAI API service unavailable. Waiting {wait_time} seconds and trying again.")
        #     time.sleep(wait_time)
        except Exception:
            wait_time = backoff(attempt)
            logger.exception(f"Rate limit reached. Waiting {wait_time} seconds and trying again")
            time.sleep(wait_time)
            break
    
    answer = res["answer"]

    ### cannot works every time diff
    # if "no mention" or "not mentioned"

    # if answer is confident: continue
    # if not, responsible of text completion
    
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources != "":
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo source is found. "
    
    else:
        answer = "The information is not provided in the given context, but I will try my best to answer: \n\n" + get_completion(message.content)
    

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, 
                         elements=source_elements, 
                         author="Chatbot").send()
    
    speak_text(answer)

@cl.on_chat_start
async def start():
    cl.user_session.get("user")

    await cl.Avatar(
        name="Chatbot",
        url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    ).send()

    await cl.Avatar(
        name="admin",
        path="user.jpg"
    ).send()

    files = None  
    while files == None:
        files = await cl.AskFileMessage(
            content = """
                    👋 Hello there!\nFeel free to share a PDF or DOCX file containing the details or information you'd like assistance with!
                    """, 
            accept=["application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            author = "Chatbot",
            max_size_mb = 20,
            timeout = 86400,
            raise_on_timeout = False
        ).send()

    file = files[0]

    msg = cl.Message(
            content=f"Processing '{file.name}' ...",
            author = "Chatbot",
            )
    await msg.send()

    text = get_pdf_text(file)
    text_chunks = get_text_chunks(text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(text_chunks))]

    vectorstore = await get_vectorstore(text_chunks, metadatas)
    chain = get_conversation_chain(vectorstore)

    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", text_chunks)

    msg.content = f"Processing '{file.name}' done.\nFeel free to ask any question!"
    await msg.update()

    cl.user_session.set("chain",chain)

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.AppUser]:
  if (username, password) == ("admin", "admin"):
    return cl.AppUser(username="admin", role="ADMIN", provider="credentials")
  else:
    return None
    

