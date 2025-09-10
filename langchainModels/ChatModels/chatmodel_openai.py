from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano", temperature=0.7) #this temperature param is used to control the randomness of the output

result = model.invoke("Suggest me girl names starting from A?") #in invoke we pass the input prompt to the model

print(result)



