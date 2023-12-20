import chainlit as cl 

import PyPDF2
from docx import Document
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain.schema import Document
from chromadb.config import Settings
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
from typing import List
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.base import VectorStore
from prompt import EXAMPLE_PROMPT, PROMPT, WELCOME_MESSAGE
#from langchain.llms import Cohere
#from langchain.llms import Together

llm = ChatOpenAI()
# llm = Together(
#     model="togethercomputer/llama-2-70b-chat",
#     temperature = 0,
#     max_tokens = 1024,
#     top_k=1,
#     together_api_key="3a72b2ff879a8c06e7198b6fc5515957a5f3515bcddbe8138d442b221c8aee61"
# )
#llm = Cohere(cohere_api_key='5Yw91akglQ0ERsH0NxmiyG31C4w37UFC5oVozKAU')
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
                "content":"You are a helpful assistant. You will do some jokes if the user is sad."
            }
        ],
        temperature = 0.5,
    )
    return response.choices[0].message.content

def get_pdf_text(file):
    
    file_stream = BytesIO(file.content)
    extension = file.name.split('.')[-1]

    text = ''

    if extension == "pdf":
        reader = PyPDF2.PdfReader(file_stream)
        for i in range(len(reader.pages)):
            text +=  reader.pages[i].extract_text()
    elif extension  ==  "docx":
        doc = Document(file_stream)
        paragraph_list = []
        for paragraph in doc.paragraphs:
            paragraph_list.append(paragraph.text)
        text = '\n'.join(paragraph_list)
    elif extension == "txt":
        text = file_stream.read().decode('utf-8')
    
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

def create_search_engine(
    *, docs: List[Document], embeddings: Embeddings, metadatas
) -> VectorStore:
# Initialize Chromadb client to enable resetting and disable telemtry
    client = chromadb.EphemeralClient()
    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        anonymized_telemetry=False,
        persist_directory=".chromadb",
        allow_reset=True,
    )

    # Reset the search engine to ensure we don't use old copies.
    # NOTE: we do not need this for production
    search_engine = Chroma(client=client, client_settings=client_settings)
    search_engine._client.reset()

    search_engine = Chroma.from_texts(
        client=client,
        texts = docs,
        embedding=embeddings,
        client_settings=client_settings,
        metadatas = metadatas
    )

    return search_engine

async def get_vectorstore(text_chunks, metadatas):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vstore = await cl.make_async(create_search_engine)(
        docs = text_chunks, embeddings = embeddings, metadatas = metadatas
    )
    return vstore

def get_conversation_chain(vectorstore):
    #llm = huggingface_hub.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs = {"prompt": PROMPT, "document_prompt": EXAMPLE_PROMPT},
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
        except Exception:
            wait_time = backoff(attempt)
            logger.exception(f"Rate limit reached. Waiting {wait_time} seconds and trying again")
            time.sleep(wait_time)
            break
    
    answer = res["answer"]
    
    sources = res["sources"].strip()
    source_elements = []

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
    

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, 
                   elements=source_elements, 
                   author="Chatbot").send()
    #speak_text(answer)

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
            content = WELCOME_MESSAGE, 
            accept=["application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "text/plain"],
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
    

