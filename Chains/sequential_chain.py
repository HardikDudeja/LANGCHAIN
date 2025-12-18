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

# Chain flow explanation:
# prompt1 | model | parser | prompt2 | model | parser
#
# Step-by-step:
# 1. prompt1 receives {"topic": "The future of AI"} -> formats to message
# 2. model processes prompt1 output -> returns AIMessage object
# 3. parser extracts .content from AIMessage -> returns STRING (the report text)
# 4. prompt2 receives the STRING from parser
#    ** KEY POINT: When a ChatPromptTemplate receives a STRING (not a dict),
#       LangChain automatically uses that string to fill ALL template variables.
#       Since prompt2 only has one variable {text}, the string fills {text}
# 5. model processes prompt2 output -> returns AIMessage object
# 6. parser extracts final text -> returns STRING (the summary)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "The future of AI"})
print(result)
