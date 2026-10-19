
# RAG Using Qdrant Cloud, Sentence Transformers, and Groq

## 1. Project Overview

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline using:

- **Sentence Transformers** to convert text into numerical embeddings.
- **Qdrant Cloud** to store and retrieve document embeddings.
- **Cosine similarity** to identify semantically relevant documents.
- **Groq API** to access a language model that generates answers from retrieved context.
- **Python dotenv** to load API credentials securely from environment variables.

The application reads employee policies from `knowledge.txt`, stores their embeddings in Qdrant, retrieves the most relevant policies for a user's question, and passes the retrieved text to an LLM to generate an answer.

### Example

**Question:**

How many vacation days do I get?

**Relevant document:**

Employees receive 24 days of paid leave per year.

**Expected answer:**

Employees receive 24 days of paid leave per year.

## 2. Project Structure

```text
rag-qdrant-project/
│
├── main.py
├── knowledge.txt
├── .env
└── .gitignore
```

| File              | Purpose                                       |
| ----------------- | --------------------------------------------- |
| `main.py`       | Contains the complete RAG pipeline            |
| `knowledge.txt` | Stores employee policy documents              |
| `.env`          | Stores API credentials                        |
| `.gitignore`    | Prevents sensitive files from being committed |

### Example `knowledge.txt`

```text
Employees receive 24 days of paid leave per year
Employees work from the office on Tuesday, Wednesday and Thursday.
Monday and Friday are optional work-from-home days.
Employees receive Rs 3000 per month for gym reimbursement.
Employees can claim Rs 2000 per month for home internet.
Employees have a 90 day notice period.
```

Each non-empty line is treated as a separate document by the current code.

**Important:** The two office-attendance lines become two separate documents. This is acceptable for this demonstration, but related lines might be better combined into one document in a larger application.

## 3. Installation and Configuration

### Step 1: Install dependencies

```bash
pip install python-dotenv groq qdrant-client sentence-transformers
```

### Step 2: Configure environment variables

Create a `.env` file:

```env
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholders with your actual credentials.

### Step 3: Protect credentials

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
```

Never push your API keys to a public GitHub repository.

### Step 4: Run the program

```bash
python main.py
```

The program requires a working Qdrant Cloud connection and valid credentials. The embedding model may also need to download its files on the first run.

---

## 4. Complete RAG Architecture

Understand this workflow before memorizing individual lines.

```text
                KNOWLEDGE.TXT
                     |
                     v
             Read the documents
                     |
                     v
          Sentence Transformer Model
                     |
                     v
             Document Embeddings
                     |
                     v
               Create Points
           (ID + Vector + Payload)
                     |
                     v
              QDRANT CLOUD
              Stores vectors
                     |
                     |
USER QUESTION        |
      |              |
      v              |
 Query Embedding     |
      |              |
      v              |
 Search Qdrant ------+
      |
      v
 Retrieve Top-K Documents
      |
      v
 Build Context
      |
      v
 Question + Context
      |
      v
   Groq API / LLM
      |
      v
 Generated Answer
```

There are two important phases:

1. **Indexing phase:** Read documents, generate embeddings, and store them in Qdrant.
2. **Query phase:** Embed a question, retrieve relevant documents, and generate an answer using the LLM.

Your current script performs both phases every time it runs.

---

## 5. Part 1 — Imports and Environment Variables

### Code

```python
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
```

### Line-by-line explanation

**1. `import os`**

Imports Python's operating-system utilities. Here, it is used to read environment variables.

**2. `from dotenv import load_dotenv`**

Imports the function that loads variables from a `.env` file into the process environment.

**3. `from groq import Groq`**

Imports the Groq client used to send requests to a language model.

**4. `from qdrant_client import QdrantClient`**

Imports the client used to communicate with Qdrant Cloud.

**5. Import Qdrant models**

```python
from qdrant_client.models import Distance, VectorParams, PointStruct
```

- `Distance`: Specifies the vector similarity or distance metric.
- `VectorParams`: Defines the configuration of vectors in a collection.
- `PointStruct`: Represents a Qdrant point containing an ID, vector, and optional payload.

**6. `SentenceTransformer`**

Loads a pre-trained model that converts text into embeddings.

**7. `load_dotenv()`**

Loads environment variables from the `.env` file.

**8. `os.getenv()`**

Retrieves each credential from the process environment.

For example:

```python
QDRANT_URL = os.getenv("QDRANT_URL")
```

If `QDRANT_URL` is not defined, the result is normally `None`.

### Interview question: Why use environment variables?

Environment variables help separate sensitive credentials from application source code. They reduce the risk of accidentally committing API keys to version control.

**Improvement:** Validate the credentials before attempting to connect:

```python
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Qdrant credentials are missing")

if not GROQ_API_KEY:
    raise ValueError("Groq API key is missing")
```

This detects missing configuration early. It does not guarantee that the supplied credentials are valid.

---

## 6. Part 2 — Connect to Qdrant Cloud

### Code

```python
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("connected to Qdrant Cloud!")
```

### Explanation

**1. `QdrantClient(...)`**

Creates a client configured to communicate with your Qdrant Cloud instance.

**2. `url=QDRANT_URL`**

Specifies the address of your Qdrant service.

**3. `api_key=QDRANT_API_KEY`**

Supplies the credential used to authenticate with Qdrant Cloud.

**4. `print(...)`**

Displays a message indicating that the client has been initialized.

Strictly speaking, printing the message does not independently prove that the remote service is reachable. An actual successful request, such as creating a collection, provides stronger evidence that the connection works.

### Interview question: What is Qdrant?

Qdrant is a vector database designed to store vectors and retrieve similar vectors efficiently. It can also store associated metadata, allowing applications to retrieve the original text alongside search results.

### Why not store everything in a normal Python list?

A Python list works for a small experiment. A vector database provides persistent storage, vector-search capabilities, and options for scaling to larger collections.

---

## 7. Part 3 — Configure and Create a Collection

### Code

```python
COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    )
)
```

### Line-by-line explanation

**1. `COLLECTION_NAME = "knowledge"`**

Defines the name of the Qdrant collection.

A collection is a logical container that holds points, their vectors, and their associated payloads.

**2. `EMBEDDING_SIZE = 384`**

Defines the number of dimensions in each embedding.

The `all-MiniLM-L6-v2` model produces 384-dimensional sentence embeddings, so the collection must be configured to accept vectors of that dimension.

If the embedding model produces vectors of a different dimension, the vector configuration must be changed accordingly.

**3. `client.collection_exists(...)`**

Checks whether the named collection already exists.

**4. `client.delete_collection(...)`**

Deletes the existing collection if present.

This is useful for rebuilding a clean dataset during a learning experiment. However, it also deletes the existing collection's data.

**5. `client.create_collection(...)`**

Creates a new collection.

**6. `VectorParams(...)`**

Configures the vector properties:

```python
VectorParams(
    size=384,
    distance=Distance.COSINE
)
```

- `size=384`: Each vector has 384 dimensions.
- `distance=Distance.COSINE`: Qdrant uses cosine distance for vector comparisons.

### What is cosine similarity?

Cosine similarity measures the directional alignment between two nonzero vectors.

\[
\operatorname{cosine}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Qdrant uses the configured cosine metric to rank vectors. For cosine-based search, higher returned scores generally indicate greater similarity.

### Important warning

Your script deletes and recreates the collection every time it runs. For production applications, you should usually preserve the collection and update only the documents that need to change.

### Interview question: What happens if the vector size is wrong?

Qdrant will reject vectors whose dimensionality does not match the collection's configuration.

---

## 8. Part 4 — Load the Knowledge Base

### Code

```python
with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [
        line.strip() for line in f if line.strip()
    ]

print(f"Loaded{len(documents)} documents")
```

### Explanation

**1. `open(...)`**

Opens `knowledge.txt` in read mode.

**2. `encoding="utf-8"`**

Specifies the character encoding used to read the file.

**3. `with`**

Ensures that the file is properly closed when the block finishes.

**4. `for line in f`**

Iterates through the file one line at a time.

**5. `line.strip()`**

Removes leading and trailing whitespace, including newline characters.

**6. `if line.strip()`**

Skips lines that contain only whitespace.

**7. `documents = [...]`**

Creates a Python list containing the cleaned, non-empty lines.

For the supplied file, the list should contain six documents because the office-attendance policy occupies two lines.

**8. `len(documents)`**

Returns the number of documents loaded.

### Small improvement

Your print statement omits a space between `Loaded` and the count.

Use:

```python
print(f"Loaded {len(documents)} documents")
```

### Interview question: What is a document in this project?

A document is one text unit used for retrieval. In this implementation, each non-empty line in `knowledge.txt` is treated as one document.

In a larger RAG application, a document might instead be a paragraph, a chunk of a long PDF, or a section of a manual.

---

## 9. Part 5 — Generate Document Embeddings

### Code

```python
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")

embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")
```

### Line-by-line explanation

**1. Load the model**

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

Loads a pre-trained sentence embedding model.

**2. Encode the documents**

```python
embeddings = model.encode(documents)
```

Converts all the documents into numerical vectors.

If six documents are loaded, the expected array shape is:

```text
(6, 384)
```

This means there are six vectors, each containing 384 values.

**3. Count the embeddings**

```python
len(embeddings)
```

Returns the number of document embeddings generated.

**4. Check the embedding dimension**

```python
len(embeddings[0])
```

Returns the number of values in the first embedding, which should be 384.

### Why generate embeddings before uploading?

Qdrant stores vectors, not the semantic meaning of raw text by itself. The embedding model transforms the text into a representation that the database can search.

### Why use `all-MiniLM-L6-v2`?

It is a relatively lightweight sentence embedding model that produces 384-dimensional vectors suitable for semantic similarity and retrieval tasks.

### Interview question: Does this code train the model?

No. It loads a pre-trained model and uses it for inference. The application does not train or fine-tune the model.

### Important limitation

The embeddings exist only in the Python process until they are uploaded. Since the collection is recreated each run, the script also recomputes the embeddings each time.

For a larger application, you would generally persist the vectors and avoid recomputing embeddings for unchanged documents.

---

## 10. Part 6 — Create Qdrant Points

### Code

```python
points = []

for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id=i + 1,
        vector=embedding.tolist(),
        payload={
            "text": documents[i]
        }
    )

    points.append(point)
```

This section transforms each embedding and its corresponding document into a Qdrant point.

### Line-by-line explanation

**1. `points = []`**

Creates an empty list that will contain the points to upload.

**2. `enumerate(embeddings)`**

Iterates through the embedding collection while providing both the index and the vector.

For example:

```text
i = 0 -> first document embedding
i = 1 -> second document embedding
i = 2 -> third document embedding
```

**3. `PointStruct(...)`**

Constructs a point containing three important fields.

#### A. Point ID

```python
id=i + 1
```

Assigns a unique integer ID to each point: 1, 2, 3, and so on.

Unique IDs are essential because Qdrant uses them to identify points.

#### B. Vector

```python
vector=embedding.tolist()
```

Converts the NumPy embedding array into a regular Python list of numerical values.

The vector has 384 elements.

#### C. Payload

```python
payload={
    "text": documents[i]
}
```

Stores the original document text as metadata associated with the vector.

This is important because the vector helps identify relevant content, while the payload gives the application the original text to pass to the LLM.

**4. `points.append(point)`**

Adds each constructed point to the list.

### Conceptual representation of a Qdrant point

```text
Point
├── ID: 1
├── Vector: [384 numerical values]
└── Payload
    └── text: "Employees receive 24 days of paid leave per year"
```

### Interview question: What is the difference between a vector and a payload?

A vector is the numerical representation used for similarity search. A payload stores associated data, such as the original text, source filename, or document category.

---

## 11. Part 7 — Upload Points to Qdrant

### Code

```python
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"Uploaded {len(points)} documents to Qdrant!")
```

### Explanation

**1. `client.upsert(...)`**

Uploads points to the specified collection.

Upsert means that a point is inserted if its ID does not already exist, or its existing contents are updated if that ID already exists.

**2. `collection_name=COLLECTION_NAME`**

Specifies the collection that will receive the points.

**3. `points=points`**

Provides the list of points containing IDs, vectors, and payloads.

**4. `print(...)`**

Displays the number of points submitted for upload.

### Why use upsert?

Upsert makes it possible to insert new records and update existing records using the same operation.

In this project, the collection is deleted and recreated first, so the points are effectively inserted into a fresh collection.

### Interview question: Is Qdrant storing the original text as a vector?

No. The embedding is stored in the vector field. The original text is stored separately in the payload.

---

## 12. Part 8 — Search Qdrant

### Code

```python
def search(query, top_k=3):
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results
```

This function performs semantic retrieval.

### Line-by-line explanation

**1. Define the function**

```python
def search(query, top_k=3):
```

- `query`: The user's question.
- `top_k=3`: The default number of search results to retrieve.

The caller can override the default by passing another value.

**2. Encode the query**

```python
query_vector = model.encode(query).tolist()
```

Converts the user's question into a 384-dimensional vector.

For example:

```text
"How many vacation days do I get?"
                 |
                 v
         Sentence Transformer
                 |
                 v
         Query embedding
```

The query embedding must be compatible with the document embeddings stored in the collection.

**3. Call `query_points()`**

```python
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=top_k,
    with_payload=True,
).points
```

This sends the query vector to Qdrant.

- `collection_name`: Identifies the collection to search.
- `query`: Supplies the query vector.
- `limit`: Sets the maximum number of results to return.
- `with_payload=True`: Requests the associated payload, including the original text.
- `.points`: Extracts the returned point results from the response.

**4. Return the results**

```python
return results
```

Returns the retrieved points to the caller.

Each result contains information such as its ID, similarity score, and payload.

### What does `top_k=3` mean?

It means the search returns up to three highest-ranked matches, assuming enough points exist.

It does not mean that Qdrant retrieves exactly three documents in every situation. Fewer results may be returned if fewer matching points are available.

### Interview question: Why encode the query with the same model?

Both query and document vectors must be in a compatible vector space so that their similarity scores are meaningful.

Using the same embedding model for both is the standard approach in this implementation.

### Important distinction

Qdrant performs the vector search. Your Python code does not loop over every document or manually calculate cosine similarity during retrieval.

---

## 13. Part 9 — Test the Search Function

### Code

```python
query = "How many vacation days do i get?"

results = search(query, top_k=3)

print("\nSearch results:")

for result in results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()
```

### Explanation

**1. Define a test query**

```python
query = "How many vacation days do i get?"
```

This is the question used to test retrieval.

**2. Search for the top three results**

```python
results = search(query, top_k=3)
```

Calls the search function and stores the returned points.

**3. Print a heading**

```python
print("\nSearch results:")
```

The `\n` adds a blank line before the heading.

**4. Iterate over the results**

```python
for result in results:
```

Processes each retrieved point.

**5. Display the similarity score**

```python
print(f"Score: {result.score:.3f}")
```

Displays the score rounded to three decimal places.

For example, `0.823` is a possible display format, not a guaranteed score for this question.

**6. Display the original document**

```python
print(result.payload["text"])
```

Reads the `text` field from the point's payload.

### What should you observe?

The paid-leave document should be relevant to the vacation question and would generally be expected to rank highly.

However, do not assume that the top result is always correct. Retrieval quality must be checked against representative test questions.

### Interview question: Is the score a probability?

No. It is a vector similarity score, not a calibrated probability that the document contains the correct answer.

---

## 14. Part 10 — Connect to Groq

### Code

```python
groq_client = Groq(
    api_key=GROQ_API_KEY
)
```

### Explanation

This creates a Groq client using the API key from the environment.

The client is used to send requests to the configured language model.

Notice that the application uses two different clients:

| Client          | Responsibility                                           |
| --------------- | -------------------------------------------------------- |
| `client`      | Stores and searches vectors in Qdrant                    |
| `groq_client` | Sends prompts to the LLM and retrieves generated answers |

### Interview question: Why are there two clients?

Because the application uses two separate services for two separate responsibilities: Qdrant handles vector storage and retrieval, while Groq provides access to the language model used for generation.

---

## 15. Part 11 — Ask the LLM

### Code

```python
def ask_llm(question, context):
    prompt = f"""
    Answer the question using only the information provided below.
    context: {context}

    Question:
    {question}

    If the answer is not present in the context, say: "I don't know based on provided information."
    """

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
```

This function generates the answer using the user's question and the retrieved documents.

### Line-by-line explanation

**1. Define the function**

```python
def ask_llm(question, context):
```

Accepts two inputs:

- `question`: What the user wants to know.
- `context`: The retrieved document text.

**2. Construct the prompt**

```python
prompt = f""" ... """
```

The f-string inserts the actual context and question into the prompt.

For example, the context might contain:

```text
Employees receive 24 days of paid leave per year.
```

The question is inserted into the `Question` section.

**3. Give the LLM an instruction**

```text
Answer the question using only the information provided below.
```

This guides the model to rely on the retrieved information rather than introduce unrelated claims.

**4. Define the fallback instruction**

```text
If the answer is not present in the context, say:
"I don't know based on provided information."
```

This instructs the model to acknowledge missing evidence.

However, a prompt is not a guarantee. The model can still produce unsupported claims, and your current code does not independently verify the answer.

**5. Call the chat completion API**

```python
response = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
```

- `model`: Specifies the model identifier.
- `messages`: Provides the conversation messages.
- `role="user"`: Sends the combined prompt as a user message.
- `content=prompt`: Supplies the actual instructions, context, and question.

**6. Extract the answer**

```python
return response.choices[0].message.content
```

Retrieves the generated text from the first returned completion choice and returns it.

### Interview question: Does the LLM search Qdrant directly?

No. In this implementation, Python calls Qdrant first, builds the context, and then sends that context to the LLM through Groq.

The LLM receives the retrieved text; it does not perform the vector search itself.

---

## 16. Part 12 — Connect the Complete RAG Pipeline

### Your current code

```python
question = "how many vacation days do i get?"

result = search(question, top_k=3)

context = "\n".join(
    result.payload["text"] for result in results
)

answer = ask_llm(question, context)

print("\nFinal Answer: ")
print(answer)
```

### Explanation

**1. Define the question**

```python
question = "how many vacation days do i get?"
```

This is the question for the final RAG pipeline.

**2. Retrieve relevant documents**

```python
result = search(question, top_k=3)
```

Runs the semantic search and returns a list of Qdrant search results.

**3. Combine retrieved text into context**

```python
context = "\n".join(
    result.payload["text"] for result in results
)
```

The intention is to extract the text from every retrieved point and combine the documents into one string separated by newline characters.

For example:

```text
Employees receive 24 days of paid leave per year.
Employees work from the office on Tuesday, Wednesday and Thursday.
Monday and Friday are optional work-from-home days.
```

The actual context depends on the documents returned by Qdrant.

### A subtle bug to fix

You assigned the search output to `result`, but the comprehension iterates over `results`.

Because `results` was populated earlier in Part 9, the code may still run, but it can use stale search results. It is also confusing because `result` is a list here, not a single search result.

Use this corrected version:

```python
question = "how many vacation days do i get?"

results = search(question, top_k=3)

context = "\n".join(
    item.payload["text"] for item in results
)

answer = ask_llm(question, context)

print("\nFinal Answer:")
print(answer)
```

This consistently uses `results` for the list and `item` for each retrieved point.

**4. Generate the answer**

```python
answer = ask_llm(question, context)
```

Sends the question and retrieved context to the LLM.

**5. Print the final answer**

```python
print(answer)
```

Displays the generated response.

### The complete logic in one sentence

The application embeds the user's question, retrieves relevant text from Qdrant, combines the retrieved text into a context string, and sends the question and context to an LLM to generate the final answer.

---

## 17. What Is the Difference Between This Project and Your Previous RAG Version?

Your earlier implementation stored document embeddings in a Python variable and calculated cosine similarity by looping over them.

This implementation stores document embeddings in Qdrant Cloud and delegates vector retrieval to the database.

| Feature                    | Earlier implementation          | Current implementation                                                  |
| -------------------------- | ------------------------------- | ----------------------------------------------------------------------- |
| Document storage           | Python list                     | Qdrant Cloud                                                            |
| Embedding generation       | Sentence Transformers           | Sentence Transformers                                                   |
| Similarity search          | Manual NumPy calculation        | Qdrant vector search                                                    |
| Metadata storage           | Original text in Python         | Text in Qdrant payload                                                  |
| Retrieval                  | Top-1 in the earlier version    | Top-k, default 3                                                        |
| Language model             | Groq API                        | Groq API                                                                |
| Persistent vector storage  | No                              | Yes, in Qdrant; although this script deletes the collection on each run |
| Scaling to larger datasets | Manual search becomes expensive | Qdrant offers vector indexing and scalable search options               |

The key architectural improvement is the separation of embedding generation, vector storage, and retrieval.

---

## 18. Common Interview Questions and Answers

### Q1. What is RAG?

RAG stands for Retrieval-Augmented Generation. It retrieves relevant external information and provides it to a language model as context so the model can generate a more grounded answer.

### Q2. What is Qdrant?

Qdrant is a vector database that stores embeddings and supports similarity search. It can also store metadata alongside vectors.

### Q3. What is an embedding?

An embedding is a numerical vector representation of text or other data. Here, the sentence embedding model generates 384-dimensional vectors for documents and queries.

### Q4. Why use Qdrant instead of NumPy?

NumPy can calculate similarity directly, which is appropriate for a small demonstration. Qdrant provides a dedicated vector-search system, persistent storage, metadata handling, and indexing options for larger collections.

### Q5. What is a Qdrant collection?

A collection is a logical group of points configured with vector settings, such as embedding dimensions and the distance metric.

### Q6. What is a Qdrant point?

A point is a stored record containing an ID, one or more vectors, and optional payload data.

### Q7. What is the purpose of the payload?

The payload stores metadata associated with a vector. In this project, it stores the original document text so that retrieved content can be sent to the LLM.

### Q8. Why use `embedding.tolist()`?

Sentence Transformers returns NumPy arrays. `.tolist()` converts the array into a regular Python list, which can be passed as the vector to the Qdrant client.

### Q9. What does `Distance.COSINE` do?

It configures cosine-based vector comparison for the collection. Qdrant uses this metric to rank vectors according to their similarity.

### Q10. What does `top_k=3` mean?

It asks Qdrant to return up to three highest-ranked results for the query.

### Q11. Why retrieve multiple documents?

A question may require information from more than one document. Retrieving multiple candidates can improve coverage, but irrelevant documents can also introduce noise into the LLM's context.

### Q12. Does a high similarity score guarantee a correct answer?

No. It indicates that the vectors are similar according to the embedding model and configured metric. It does not prove that the document contains the answer.

### Q13. What happens if the answer is absent from the knowledge base?

The current system can still retrieve the closest available documents. If those documents do not contain the answer, the LLM may still generate an unsupported response. A robust implementation should detect insufficient evidence and handle it explicitly.

### Q14. What is the difference between retrieval and generation?

Retrieval identifies relevant documents. Generation uses the question and retrieved context to produce a natural-language answer.

### Q15. Does this project train an LLM?

No. It uses a pre-trained embedding model and calls a hosted language model through Groq. It does not train or fine-tune either model.

### Q16. What is the difference between indexing and querying?

Indexing processes the knowledge base, generates document embeddings, and stores the vectors. Querying encodes the question, searches the stored vectors, and uses retrieved information to generate an answer.

### Q17. What is the purpose of `client.upsert()`?

It inserts new points or updates existing points with matching IDs in the collection.

### Q18. Why does the code delete the collection before creating it?

It ensures that the demo starts with a clean collection. However, it destroys previously stored data, so this approach is not suitable when the application needs to preserve its knowledge base between runs.

### Q19. What is the main limitation of this implementation?

It performs a simple line-based indexing process, retrieves the top-k vectors without a relevance threshold, and relies on prompt instructions for grounded generation. It also rebuilds the collection and embeddings each time the script runs.

### Q20. How would you improve this project?

I would preserve the collection, avoid recomputing embeddings for unchanged documents, use better document chunking, add relevance thresholds and fallback logic, evaluate retrieval quality, and implement API error handling. For larger datasets, I would also investigate metadata filtering, hybrid search, and reranking.

---

## 19. Important Limitations to Remember

- **Collection deletion:** Existing Qdrant data is deleted each time the script runs.
- **Repeated embedding generation:** Document embeddings are recomputed on every execution.
- **Line-based documents:** Related lines may be split into separate retrieval units.
- **No relevance threshold:** The system does not explicitly reject weak matches.
- **Prompt-based grounding:** The LLM is instructed to use the context, but its output is not independently verified.
- **No explicit API error handling:** Network failures, rate limits, invalid credentials, and unavailable models can interrupt execution.
- **Potentially noisy context:** Top-k retrieval may include documents that are less useful for the question.
- **No formal evaluation:** The code does not measure retrieval accuracy or answer faithfulness.

These are reasonable limitations for a learning project. Being able to identify them is useful in interviews.

---

## 20. Quick Revision Sheet

| Concept                   | Remember                                           |
| ------------------------- | -------------------------------------------------- |
| RAG                       | Retrieval-Augmented Generation                     |
| Knowledge base            | `knowledge.txt`                                  |
| Embedding model           | `all-MiniLM-L6-v2`                               |
| Embedding dimension       | 384                                                |
| Vector database           | Qdrant Cloud                                       |
| Collection                | Container for vectors and their associated records |
| Point                     | ID + vector + payload                              |
| Payload                   | Stores original text and other metadata            |
| Distance metric           | Cosine                                             |
| `model.encode()`        | Converts text into vectors                         |
| `client.upsert()`       | Inserts or updates points                          |
| `client.query_points()` | Searches for similar vectors                       |
| `top_k=3`               | Requests up to three results                       |
| Groq                      | API provider used for language-model generation    |
| `ask_llm()`             | Generates an answer using the question and context |
| Indexing                  | Embed documents and store them                     |
| Querying                  | Embed the question and retrieve relevant documents |
| Current limitation        | Collection is rebuilt on every run                 |

---

## 21. The 30-Second Interview Explanation

"I built a basic RAG application using Sentence Transformers, Qdrant Cloud, and the Groq API. First, I load employee policy documents and convert them into 384-dimensional embeddings using `all-MiniLM-L6-v2`. I store those embeddings in a Qdrant collection along with the original text in each point's payload. When a user asks a question, I encode the query and use Qdrant's cosine-based vector search to retrieve the top three relevant documents. I combine the retrieved text into a context string and pass it with the question to a language model through Groq. The model generates the final answer using the supplied context. The current implementation is a basic RAG pipeline, and I would improve it with persistent indexing, relevance thresholds, better chunking, and retrieval evaluation."

---

## 22. Recommended Practice Before an Interview

Try explaining these concepts without looking at the README:

1. Why do you need an embedding model?
2. Why is the embedding dimension 384?
3. What is a Qdrant collection?
4. What information does a point contain?
5. Why do you store the original text in the payload?
6. What happens when you call `query_points()`?
7. Why retrieve the top three documents?
8. How does the retrieved context reach the LLM?
9. Why does RAG not completely eliminate hallucinations?
10. What would you change before deploying this application?

If you can answer these questions and trace a query from input to final answer, you understand the architecture rather than merely memorizing individual lines.
