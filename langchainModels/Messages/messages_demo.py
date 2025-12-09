from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the chat model
# Temperature parameter (0.0 to 2.0):
# - Controls the randomness/creativity of the model's output
# - Lower values (0.0-0.3): More deterministic, focused, and consistent responses
# - Medium values (0.5-0.7): Balanced between creativity and consistency
# - Higher values (0.8-2.0): More creative, diverse, and unpredictable responses
# - Default is usually around 0.7-1.0
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Different types of messages in LangChain

# 1. HumanMessage - represents a message from the user
human_msg = HumanMessage(content="What is the capital of France?")

# 2. AIMessage - represents a message from the AI assistant
ai_msg = AIMessage(content="The capital of France is Paris.")

# 3. SystemMessage - represents a system message (instructions for the AI)
system_msg = SystemMessage(content="You are a helpful assistant that provides concise answers.")

# Example 1: Using a single HumanMessage
print("=" * 50)
print("Example 1: Single HumanMessage")
print("=" * 50)
response = model.invoke([human_msg])
print(f"Human: {human_msg.content}")
print(f"AI: {response.content}\n")

# Example 2: Using SystemMessage + HumanMessage (conversation with context)
print("=" * 50)
print("Example 2: SystemMessage + HumanMessage")
print("=" * 50)
messages = [
    system_msg,
    HumanMessage(content="Explain quantum computing in simple terms.")
]
response = model.invoke(messages)
print(f"System: {system_msg.content}")
print(f"Human: {messages[1].content}")
print(f"AI: {response.content}\n")

# Example 3: Conversation history (multiple messages)
print("=" * 50)
print("Example 3: Conversation with history")
print("=" * 50)
conversation = [
    SystemMessage(content="You are a friendly math tutor."),
    HumanMessage(content="What is 2 + 2?"),
    AIMessage(content="2 + 2 equals 4."),
    HumanMessage(content="What about 3 + 3?")
]
response = model.invoke(conversation)
print("Conversation:")
for msg in conversation:
    print(f"{type(msg).__name__}: {msg.content}")
print(f"AI Response: {response.content}\n")

# Example 4: Accessing message properties
print("=" * 50)
print("Example 4: Message properties")
print("=" * 50)
msg = HumanMessage(content="Hello, how are you?")
print(f"Content: {msg.content}")
print(f"Type: {type(msg).__name__}")
print(f"Message object: {msg}")


