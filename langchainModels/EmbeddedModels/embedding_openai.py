from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

sentences = [
    "Delhi is the capital of India",
    "Paris is the capital of France",
    "Tokyo is the capital of Japan",
    "Bananas are yellow",
    "Apples are sweet and crunchy",
    "Cricket is a popular sport in India",
    "Football is loved in Brazil",
    "Python is a popular programming language",
    "JavaScript powers the web"
]

# Generate embeddings
vectors = [embedding.embed_query(s) for s in sentences]

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compare similarities across themes
pairs = [
    (0, 1),  # Delhi vs Paris
    (0, 2),  # Delhi vs Tokyo
    (0, 3),  # Delhi vs Bananas
    (3, 4),  # Bananas vs Apples
    (5, 6),  # Cricket vs Football
    (7, 8),  # Python vs JavaScript
    (5, 7),  # Cricket vs Python
]

for i, j in pairs:
    sim = cosine_similarity(vectors[i], vectors[j])
    print(f"Similarity ({sentences[i]}  <->  {sentences[j]}): {sim:.3f}")