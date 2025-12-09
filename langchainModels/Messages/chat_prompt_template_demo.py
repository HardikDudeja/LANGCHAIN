from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the chat model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# ============================================================================
# Example 1: Basic ChatPromptTemplate with system and user messages
# ============================================================================
print("=" * 70)
print("Example 1: Basic ChatPromptTemplate")
print("=" * 70)

# Create a prompt template with system and user messages
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains concepts clearly."),
    ("human", "Explain {topic} in simple terms.")
])

# Format the prompt with a variable
formatted_prompt = prompt_template.format_messages(topic="quantum computing")
print("Formatted Messages:")
for msg in formatted_prompt:
    print(f"  {type(msg).__name__}: {msg.content}")

# Invoke the model with the formatted prompt
response = model.invoke(formatted_prompt)
print(f"\nResponse: {response.content}\n")

# ============================================================================
# Example 2: ChatPromptTemplate with multiple variables
# ============================================================================
print("=" * 70)
print("Example 2: Template with Multiple Variables")
print("=" * 70)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Provide {style} answers."),
    ("human", "Question: {question}")
])

formatted_prompt = prompt_template.format_messages(
    role="math tutor",
    style="step-by-step",
    question="How do I solve 2x + 5 = 15?"
)

response = model.invoke(formatted_prompt)
print("Prompt Variables:")
print(f"  Role: math tutor")
print(f"  Style: step-by-step")
print(f"  Question: How do I solve 2x + 5 = 15?")
print(f"\nResponse: {response.content}\n")

# ============================================================================
# Example 3: Using MessagesPlaceholder for conversation history
# ============================================================================
print("=" * 70)
print("Example 3: ChatPromptTemplate with Conversation History")
print("=" * 70)

# Template that includes conversation history
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly chatbot. Remember the conversation context."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Simulate a conversation
conversation_history = [
    HumanMessage(content="My name is Alice."),
    AIMessage(content="Nice to meet you, Alice! How can I help you today?")
]

formatted_prompt = prompt_template.format_messages(
    history=conversation_history,
    input="What's my name?"
)

response = model.invoke(formatted_prompt)
print("Conversation History:")
for msg in conversation_history:
    print(f"  {type(msg).__name__}: {msg.content}")
print(f"\nUser: What's my name?")
print(f"Bot: {response.content}\n")

# ============================================================================
# Example 4: Simple Chatbot Simulation
# ============================================================================
print("=" * 70)
print("Example 4: Chatbot Simulation")
print("=" * 70)

class SimpleChatbot:
    def __init__(self, system_prompt="You are a helpful assistant."):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.conversation_history = []
        
        # Create prompt template with system message and conversation history
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}")
        ])
    
    def chat(self, user_input):
        # Format the prompt with current history and user input
        formatted_prompt = self.prompt_template.format_messages(
            history=self.conversation_history,
            user_input=user_input
        )
        
        # Get response from model
        response = self.model.invoke(formatted_prompt)
        
        # Update conversation history
        self.conversation_history.append(HumanMessage(content=user_input))
        self.conversation_history.append(AIMessage(content=response.content))
        
        return response.content
    
    def reset(self):
        """Reset the conversation history"""
        self.conversation_history = []

# Create a chatbot instance
bot = SimpleChatbot(system_prompt="You are a friendly travel assistant. Help users plan their trips.")

# Simulate a conversation
print("Chatbot: Hello! I'm your travel assistant. How can I help you today?\n")

conversation = [
    "I want to visit Paris. What are some must-see places?",
    "What's the best time of year to visit?",
    "Can you suggest a 3-day itinerary?"
]

for user_msg in conversation:
    print(f"User: {user_msg}")
    bot_response = bot.chat(user_msg)
    print(f"Bot: {bot_response}\n")

# ============================================================================
# Example 5: Interactive Chatbot (commented out - uncomment to use)
# ============================================================================
print("=" * 70)
print("Example 5: Interactive Chatbot Template")
print("=" * 70)
print("""
# Uncomment the code below to run an interactive chatbot:

interactive_bot = SimpleChatbot(
    system_prompt="You are a helpful coding assistant. Provide clear code examples."
)

print("Chatbot started! Type 'quit' to exit.")
while True:
    user_input = input("\\nYou: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Bot: Goodbye! Happy coding!")
        break
    response = interactive_bot.chat(user_input)
    print(f"Bot: {response}")
""")

# ============================================================================
# Example 6: Different Template Formats
# ============================================================================
print("=" * 70)
print("Example 6: Different Template Creation Methods")
print("=" * 70)

# Method 1: Using from_messages with tuples
template1 = ChatPromptTemplate.from_messages([
    ("system", "You are {assistant_type}"),
    ("human", "{question}")
])

# Method 2: Using from_template (single message)
template2 = ChatPromptTemplate.from_template("Explain {concept} like I'm 5 years old.")

# Method 3: Using from_messages with Message objects
template3 = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="{user_query}")
])

print("Template 1 (tuples):")
messages1 = template1.format_messages(assistant_type="a chef", question="How do I make pasta?")
print(f"  System: {messages1[0].content}")
print(f"  Human: {messages1[1].content}\n")

print("Template 2 (from_template):")
messages2 = template2.format_messages(concept="gravity")
print(f"  Human: {messages2[0].content}\n")

print("Template 3 (Message objects):")
messages3 = template3.format_messages(user_query="What is Python?")
print(f"  System: {messages3[0].content}")
print(f"  Human: {messages3[1].content}\n")

