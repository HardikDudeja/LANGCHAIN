from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("user", "Write a poem about {topic}.")
])

model = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({"topic": "courage"}))

chain.get_graph().print_ascii()
