from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# ============================================================================
# Example 1: Basic Parallel Chain - Multiple Analyses of Same Input
# ============================================================================
print("=" * 70)
print("Example 1: Basic Parallel Chain - Multiple Analyses")
print("=" * 70)

# Define different prompts for different analyses
summary_prompt = ChatPromptTemplate.from_messages([
    ("user", "Provide a brief summary of: {topic}")
])

pros_prompt = ChatPromptTemplate.from_messages([
    ("user", "List 3 pros of: {topic}")
])

cons_prompt = ChatPromptTemplate.from_messages([
    ("user", "List 3 cons of: {topic}")
])

# Create individual chains
summary_chain = summary_prompt | model | parser
pros_chain = pros_prompt | model | parser
cons_chain = cons_prompt | model | parser

# Create parallel chain - runs all chains simultaneously
parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "pros": pros_chain,
    "cons": cons_chain
})

# Invoke with single input - all chains run in parallel
result = parallel_chain.invoke({"topic": "artificial intelligence"})

print(f"Topic: artificial intelligence\n")
print(f"Summary: {result['summary']}\n")
print(f"Pros: {result['pros']}\n")
print(f"Cons: {result['cons']}\n")

# ============================================================================
# Example 2: Parallel Chain with Different Output Formats
# ============================================================================
print("=" * 70)
print("Example 2: Parallel Chain - Different Output Formats")
print("=" * 70)

# Different types of analysis
explanation_prompt = ChatPromptTemplate.from_messages([
    ("user", "Explain {topic} in simple terms.")
])

examples_prompt = ChatPromptTemplate.from_messages([
    ("user", "Give 3 real-world examples of {topic}")
])

use_cases_prompt = ChatPromptTemplate.from_messages([
    ("user", "What are the main use cases of {topic}?")
])

parallel_chain = RunnableParallel({
    "explanation": explanation_prompt | model | parser,
    "examples": examples_prompt | model | parser,
    "use_cases": use_cases_prompt | model | parser
})

result = parallel_chain.invoke({"topic": "machine learning"})

print(f"Topic: machine learning\n")
print(f"Explanation:\n{result['explanation']}\n")
print(f"Examples:\n{result['examples']}\n")
print(f"Use Cases:\n{result['use_cases']}\n")

# ============================================================================
# Example 3: Parallel Chain with Sequential Processing After
# ============================================================================
print("=" * 70)
print("Example 3: Parallel Chain Followed by Sequential Processing")
print("=" * 70)

# Parallel chains for different analyses
analysis_chain = RunnableParallel({
    "sentiment": ChatPromptTemplate.from_messages([
        ("user", "Analyze the sentiment of: {text}")
    ]) | model | parser,
    
    "keywords": ChatPromptTemplate.from_messages([
        ("user", "Extract 5 key keywords from: {text}")
    ]) | model | parser,
    
    "summary": ChatPromptTemplate.from_messages([
        ("user", "Summarize: {text}")
    ]) | model | parser
})

# Sequential chain that uses parallel chain output
final_prompt = ChatPromptTemplate.from_messages([
    ("user", """Based on the following analysis:
    
Sentiment: {sentiment}
Keywords: {keywords}
Summary: {summary}

Provide a comprehensive report combining all this information.""")
])

# Combine parallel and sequential
combined_chain = analysis_chain | final_prompt | model | parser

result = combined_chain.invoke({
    "text": "Artificial intelligence is transforming industries worldwide. While it offers incredible opportunities for innovation and efficiency, there are concerns about job displacement and ethical implications."
})

print(f"Input Text: Artificial intelligence is transforming industries...\n")
print(f"Final Report:\n{result}\n")

# ============================================================================
# Example 4: Parallel Chain with Different Models (if needed)
# ============================================================================
print("=" * 70)
print("Example 4: Parallel Chain - Multiple Perspectives")
print("=" * 70)

# Same model, different perspectives
technical_prompt = ChatPromptTemplate.from_messages([
    ("user", "Explain {topic} from a technical perspective.")
])

business_prompt = ChatPromptTemplate.from_messages([
    ("user", "Explain {topic} from a business perspective.")
])

educational_prompt = ChatPromptTemplate.from_messages([
    ("user", "Explain {topic} from an educational perspective.")
])

parallel_chain = RunnableParallel({
    "technical": technical_prompt | model | parser,
    "business": business_prompt | model | parser,
    "educational": educational_prompt | model | parser
})

result = parallel_chain.invoke({"topic": "blockchain technology"})

print(f"Topic: blockchain technology\n")
print(f"Technical Perspective:\n{result['technical']}\n")
print(f"Business Perspective:\n{result['business']}\n")
print(f"Educational Perspective:\n{result['educational']}\n")
