# =========================================
# part 1 imports and environments
# =========================================

import os
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================================
# part 2
# =========================================

client = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)

print("connected to Qdrant Cloud!")

# =========================================
# part 3
# =========================================

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

# delete collection if it already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

# =========================================
# create collection
# =========================================
client.create_collection(
    collection_name = COLLECTION_NAME,
    vectors_config = VectorParams(
        size = EMBEDDING_SIZE,
        distance = Distance.COSINE
    )
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")

# =========================================
# part 4 load our knowledge
# =========================================

with open("knowledge.txt", "r", encoding = 'utf-8') as f:
    documents = [
        line.strip() for line in f if line.strip()
    ]

print(f"Loaded{len(documents)} documents")

# =========================================
# part 5 - Create Embedding
# =========================================
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")

embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")

# =========================================
# part 6 create qdrant points
# =========================================

points = []
for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id = i+1,
        vector = embedding.tolist(),
        payload = {
            "text": documents[i]
        }
    )
    
    points.append(point)
    

# =========================================
# part 7 upload to qdrant
# =========================================

client.upsert(  # upload + insert
    collection_name = COLLECTION_NAME,
    points = points
)
print(f"Uploaded {len(points)} documents to Qdrant!")

# =========================================
# part 8 search qdrant
# =========================================

def search(query, top_k = 3):
    # convert the question into an embedding
    query_vector = model.encode(query).tolist()

    # search Qdrant for similar vectors
    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k,
        with_payload = True,
    ).points
    
    return results


# =========================================
# part 9 test search
# =========================================

query = "How many vacation days do i get?"

results = search(query, top_k = 3)

print("\nSearch results: ")
for result in results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()
    
# =========================================
# part 10 connected to groq
# =========================================

groq_client = Groq(
    api_key = GROQ_API_KEY
)

# =========================================
# part 11 ask the llm
# =========================================

def ask_llm(question, context):
    prompt = f"""
    Answer the question using only the information provided below.
    context: {context}

    Question:
    {question}
    
    If the answer is not present in the context, say: "I don't know based on provided information."
    """
    
    response = groq_client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# =========================================
# part 12 connected rag pipeline
# =========================================

question = "how many vacation days do i get?"

result = search(question, top_k=3)

# Extract text from the search results
context = "\n".join(
    result.payload["text"] for result in results
)

answer = ask_llm(question, context)

print("\nFinal Answer: ")
print(answer)