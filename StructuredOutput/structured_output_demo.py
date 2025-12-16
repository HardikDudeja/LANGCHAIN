from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load .env file from parent directory (LANGCHAIN folder)

class MovieInfo(TypedDict):
    title: str
    year: int
    genre: str

model = ChatOpenAI(model="gpt-4o-mini")

structured_model = model.with_structured_output(MovieInfo)

result = structured_model.invoke("Tell me about the movie Inception")
print(result)
