from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# ============================================================================
# Example 1: Basic Conditional Chain - Route Based on Input Type
# ============================================================================
print("=" * 70)
print("Example 1: Basic Conditional Chain - Route by Input Type")
print("=" * 70)

# Define different chains for different types of queries
technical_prompt = ChatPromptTemplate.from_messages([
    ("user", "Provide a technical explanation of {query}")
])

simple_prompt = ChatPromptTemplate.from_messages([
    ("user", "Explain {query} in simple, easy-to-understand terms")
])

# Function to determine which chain to use
def route_by_complexity(input_dict):
    query = input_dict.get("query", "").lower()
    # Simple heuristic: if query contains technical terms, use technical chain
    technical_terms = ["algorithm", "api", "database", "protocol", "architecture", "framework"]
    if any(term in query for term in technical_terms):
        return "technical"
    return "simple"

# Create chains
technical_chain = technical_prompt | model | parser
simple_chain = simple_prompt | model | parser

# Create conditional chain using RunnableLambda for routing
def route_chain(input_dict):
    if route_by_complexity(input_dict) == "technical":
        return technical_chain.invoke(input_dict)
    else:
        return simple_chain.invoke(input_dict)

conditional_chain = RunnableLambda(route_chain)

# Test with different inputs
result1 = conditional_chain.invoke({"query": "What is an API?"})
print(f"Query: 'What is an API?'")
print(f"Route: Technical")
print(f"Response: {result1}\n")

result2 = conditional_chain.invoke({"query": "What is a cat?"})
print(f"Query: 'What is a cat?'")
print(f"Route: Simple")
print(f"Response: {result2}\n")


# ============================================================================
# Example 2: Conditional Chain Based on Sentiment Analysis
# ============================================================================
print("=" * 70)
print("Example 2: Conditional Chain - Route by Sentiment")
print("=" * 70)

# Chains for different sentiments
positive_prompt = ChatPromptTemplate.from_messages([
    ("user", "The user said: '{text}'. Respond enthusiastically and positively.")
])

negative_prompt = ChatPromptTemplate.from_messages([
    ("user", "The user said: '{text}'. Respond empathetically and helpfully.")
])

neutral_prompt = ChatPromptTemplate.from_messages([
    ("user", "The user said: '{text}'. Respond neutrally and informatively.")
])

# Sentiment analysis chain
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("user", "Analyze the sentiment of this text and respond with ONLY one word: 'positive', 'negative', or 'neutral'. Text: {text}")
])

sentiment_chain = sentiment_prompt | model | parser

# Function to route based on sentiment
def route_by_sentiment(input_dict):
    text = input_dict.get("text", "")
    sentiment = sentiment_chain.invoke({"text": text}).strip().lower()
    
    if "positive" in sentiment:
        return "positive"
    elif "negative" in sentiment:
        return "negative"
    else:
        return "neutral"

# Create response chains
positive_chain = positive_prompt | model | parser
negative_chain = negative_prompt | model | parser
neutral_chain = neutral_prompt | model | parser

# Conditional routing
conditional_chain = RunnableBranch(
    (lambda x: route_by_sentiment(x) == "positive", positive_chain),
    (lambda x: route_by_sentiment(x) == "negative", negative_chain),
    (lambda x: True, neutral_chain)  # Default to neutral
)

result1 = conditional_chain.invoke({"text": "I love this product! It's amazing!"})
print(f"Input: 'I love this product! It's amazing!'")
print(f"Response: {result1}\n")

result2 = conditional_chain.invoke({"text": "This is terrible, I'm very disappointed."})
print(f"Input: 'This is terrible, I'm very disappointed.'")
print(f"Response: {result2}\n")

# ============================================================================
# Example 3: Multi-Conditional Chain - Route by Topic Category
# ============================================================================
print("=" * 70)
print("Example 3: Multi-Conditional Chain - Route by Topic Category")
print("=" * 70)

# Different chains for different topics
science_prompt = ChatPromptTemplate.from_messages([
    ("user", "As a science expert, explain: {topic}")
])

history_prompt = ChatPromptTemplate.from_messages([
    ("user", "As a history expert, explain: {topic}")
])

technology_prompt = ChatPromptTemplate.from_messages([
    ("user", "As a technology expert, explain: {topic}")
])

general_prompt = ChatPromptTemplate.from_messages([
    ("user", "Explain: {topic}")
])

# Category detection chain
category_prompt = ChatPromptTemplate.from_messages([
    ("user", """Categorize this topic into ONE of these categories: 'science', 'history', 'technology', or 'general'.
Topic: {topic}
Respond with ONLY the category name.""")
])

category_chain = category_prompt | model | parser

# Function to route by category
def route_by_category(input_dict):
    topic = input_dict.get("topic", "")
    category = category_chain.invoke({"topic": topic}).strip().lower()
    
    if "science" in category:
        return "science"
    elif "history" in category:
        return "history"
    elif "technology" in category or "tech" in category:
        return "technology"
    else:
        return "general"

# Create chains
science_chain = science_prompt | model | parser
history_chain = history_prompt | model | parser
technology_chain = technology_prompt | model | parser
general_chain = general_prompt | model | parser

# Multi-conditional routing
conditional_chain = RunnableBranch(
    (lambda x: route_by_category(x) == "science", science_chain),
    (lambda x: route_by_category(x) == "history", history_chain),
    (lambda x: route_by_category(x) == "technology", technology_chain),
    (lambda x: True, general_chain)  # Default fallback
)

result1 = conditional_chain.invoke({"topic": "photosynthesis"})
print(f"Topic: 'photosynthesis'")
print(f"Category: Science")
print(f"Response: {result1}\n")

result2 = conditional_chain.invoke({"topic": "World War II"})
print(f"Topic: 'World War II'")
print(f"Category: History")
print(f"Response: {result2}\n")

# ============================================================================
# Example 4: Conditional Chain with Direct Condition Check
# ============================================================================
print("=" * 70)
print("Example 4: Conditional Chain - Direct Condition Check")
print("=" * 70)

# Chains for different user roles
admin_prompt = ChatPromptTemplate.from_messages([
    ("user", "As an admin, provide detailed technical information about: {query}")
])

user_prompt = ChatPromptTemplate.from_messages([
    ("user", "Provide user-friendly information about: {query}")
])

# Simple conditional routing based on input field
def route_by_role(input_dict):
    role = input_dict.get("role", "user").lower()
    return role == "admin"

admin_chain = admin_prompt | model | parser
user_chain = user_prompt | model | parser

conditional_chain = RunnableBranch(
    (lambda x: route_by_role(x), admin_chain),
    (lambda x: True, user_chain)
)

result1 = conditional_chain.invoke({"role": "admin", "query": "database optimization"})
print(f"Role: admin, Query: 'database optimization'")
print(f"Response: {result1}\n")

result2 = conditional_chain.invoke({"role": "user", "query": "database optimization"})
print(f"Role: user, Query: 'database optimization'")
print(f"Response: {result2}\n")

# ============================================================================
# Example 5: Complex Conditional Chain - Language Detection and Routing
# ============================================================================
print("=" * 70)
print("Example 5: Conditional Chain - Language Detection")
print("=" * 70)

# Chains for different languages (simplified - using English prompts)
english_prompt = ChatPromptTemplate.from_messages([
    ("user", "Respond in English: {text}")
])

spanish_prompt = ChatPromptTemplate.from_messages([
    ("user", "Responde en español: {text}")
])

# Language detection
lang_detect_prompt = ChatPromptTemplate.from_messages([
    ("user", """Detect the language of this text and respond with ONLY 'english' or 'spanish'.
Text: {text}""")
])

lang_detect_chain = lang_detect_prompt | model | parser

def route_by_language(input_dict):
    text = input_dict.get("text", "")
    detected_lang = lang_detect_chain.invoke({"text": text}).strip().lower()
    return "spanish" in detected_lang

english_chain = english_prompt | model | parser
spanish_chain = spanish_prompt | model | parser

conditional_chain = RunnableBranch(
    (lambda x: route_by_language(x), spanish_chain),
    (lambda x: True, english_chain)
)

result1 = conditional_chain.invoke({"text": "Hello, how are you?"})
print(f"Input: 'Hello, how are you?'")
print(f"Response: {result1}\n")

result2 = conditional_chain.invoke({"text": "Hola, ¿cómo estás?"})
print(f"Input: 'Hola, ¿cómo estás?'")
print(f"Response: {result2}\n")
