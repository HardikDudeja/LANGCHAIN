from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

class Mood():
    feeling: str
    reason: str

prompt = ChatPromptTemplate.from_messages([
    ("user", "What is your mood today?")
])

model = ChatOpenAI(model="gpt-4o-mini")

parser = PydanticOutputParser(pydantic_object=Mood)

chain = prompt | model | parser

print(chain.invoke({"input": "I am feeling happy today."}))