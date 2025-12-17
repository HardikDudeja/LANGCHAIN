from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = ChatPromptTemplate.from_messages([
    ("user", "Generate a detailed report on {topic}.")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("user", "Generate a 5 pointer summary from the following text \n {text}")
])

model = ChatOpenAI()

parser = StrOutputParser()