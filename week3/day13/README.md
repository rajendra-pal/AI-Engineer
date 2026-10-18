
# Basic RAG (Retrieval-Augmented Generation) Using Sentence Transformers and Groq

## 1. Project Overview

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline using Sentence Transformers for semantic retrieval and a language model accessed through the Groq API for answer generation.

The system stores a small collection of employee policy documents, finds the document most relevant to a user's question, and passes that document to an LLM to generate an answer.

### Example

**User question:**

> How much vacations can i get?

**Retrieved context:**

> Employees receive 24 days of paid leave per year.

**Generated answer:**

> Employees receive 24 days of paid leave per year.

The exact wording of the generated answer can vary.

### Technologies Used

- Python
- Sentence Transformers
- NumPy
- Groq API
- Python dotenv
- `all-MiniLM-L6-v2` embedding model
- `openai/gpt-oss-120b` language model identifier

## 2. What Is RAG?

RAG stands for **Retrieval-Augmented Generation**.

It is an approach that combines information retrieval with text generation. Instead of relying only on the language model's existing knowledge, the application retrieves relevant information from an external knowledge source and provides it to the model as context.

### Why do we need RAG?

Suppose a company has the following internal policy:

"Employees receive 24 days of paid leave per year."

A general-purpose LLM may not know this company's specific policy. RAG allows the application to retrieve the relevant policy and use it to answer the question.

### The three main stages

1. **Retrieval:** Find the most relevant information from the document collection.
2. **Augmentation:** Include the retrieved information in the LLM's prompt.
3. **Generation:** Generate an answer using the question and retrieved context.

## 3. Architecture and Workflow

```text
              EMPLOYEE POLICY DOCUMENTS
                         |
                         v
               Sentence Transformer
                         |
                         v
                Document Embeddings
                         |
                         |
USER QUESTION            |
      |                  |
      v                  |
 Sentence Transformer     |
      |                  |
      v                  |
  Query Embedding        |
      |                  |
      +--------+---------+
               |
               v
       Cosine Similarity
               |
               v
     Retrieve Top-1 Document
               |
               v
     Question + Retrieved Context
               |
               v
             Groq API
               |
               v
       LLM-Generated Answer
               |
               v
          Print Answer
```

**Key idea:** The embedding model retrieves information, while the LLM generates the answer. These are two separate responsibilities.

---

## 4. Installation and Setup

### Step 1: Install dependencies

```bash
pip install sentence-transformers numpy groq python-dotenv
```

### Step 2: Create a `.env` file

Store your Groq API key in a `.env` file in the project's root directory:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Replace the placeholder with your own API key.

**Security:** Never commit your `.env` file or expose your API key in a public GitHub repository.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
```

### Step 3: Create `main.py`

Add the project code to `main.py`.

### Step 4: Run the project

```bash
python main.py
```

The first execution may download the embedding model. The Groq API also requires a valid API key, network connectivity, and access to the specified model.

---

## 5. Understanding the Imports and Model Setup

### Imports

```python
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
```

| Import                  | Purpose                           |
| ----------------------- | --------------------------------- |
| `os`                  | Reads environment variables       |
| `load_dotenv`         | Loads variables from `.env`     |
| `Groq`                | Creates a client for the Groq API |
| `SentenceTransformer` | Converts text into embeddings     |
| `numpy`               | Performs vector calculations      |

### Loading the embedding model

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

This loads a pre-trained sentence embedding model.

Its output dimension is 384, meaning each individual sentence is represented by a vector containing 384 numerical values.

The model is used to convert both the documents and the user's question into vectors that can be compared mathematically.

### Selecting the LLM

```python
groq_model = "openai/gpt-oss-120b"
```

This specifies the model identifier passed to Groq's chat completion API.

It is separate from the Sentence Transformers model:

- `all-MiniLM-L6-v2`: generates embeddings.
- `openai/gpt-oss-120b`: generates the final text answer through the Groq API.

The model identifier must be supported by the account and API endpoint at execution time.

---

## 6. Understanding Environment Variables and API Keys

```python
load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api key is not working")
```

### What does `load_dotenv()` do?

It loads environment variables from a `.env` file into the process environment.

### What does `os.getenv()` do?

```python
my_api_key = os.getenv("GROQ_API_KEY")
```

It retrieves the value associated with the environment variable `GROQ_API_KEY`.

If the variable is missing, `os.getenv()` returns `None` by default.

### Why do we check the API key?

```python
if not my_api_key:
    raise ValueError("Api key is not working")
```

This stops execution when the variable is missing or empty.

However, it does not actually verify that the API key is valid. A non-empty but incorrect key can still pass this check and fail when the API request is made.

A more accurate error message would be:

```python
if not my_api_key:
    raise ValueError("GROQ_API_KEY is missing")
```

### Creating the Groq client

```python
client = Groq(api_key=my_api_key)
```

This creates a client configured to authenticate API requests using the supplied key.

---

## 7. Understanding the Documents

```python
documents = [
    "Employees receive 24 days of paid leave per year.",

    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",

    "Employees receive Rs 3000 per month for gym reimbursement.",

    "Employees can claim Rs 2000 per month for home internet.",

    "Employees have a 90 day notice period."
]
```

This list acts as the application's small knowledge base.

It contains five employee policy statements:

| Index | Information                               |
| ----- | ----------------------------------------- |
| 0     | Annual paid leave                         |
| 1     | Office attendance and work-from-home days |
| 2     | Gym reimbursement                         |
| 3     | Home internet reimbursement               |
| 4     | Notice period                             |

The application searches this list to answer a user's question.

### Why is a list sufficient here?

A list is easy to understand and suitable for a small demonstration. Each item is a separate text document.

For a real application containing thousands or millions of documents, a vector index or vector database would be more suitable.

### Important Python detail

Adjacent string literals inside parentheses or a list can be concatenated automatically:

```python
"Employees work from the office on Tuesday, Wednesday and Thursday. "
"Monday and Friday are optional work-from-home days."
```

Python treats these as one string, not two separate documents.

---

## 8. Generating Document Embeddings

```python
document_embeddings = model.encode(documents)
```

This line converts all five documents into numerical embeddings.

Conceptually, the output has the following shape:

```text
5 documents
     |
     v
Sentence Transformer
     |
     v
5 vectors, each with 384 values
     |
     v
Shape: (5, 384)
```

Each row corresponds to one document.

| Document                      | Embedding shape |
| ----------------------------- | --------------- |
| Paid leave policy             | `(384,)`      |
| Office attendance policy      | `(384,)`      |
| Gym reimbursement policy      | `(384,)`      |
| Internet reimbursement policy | `(384,)`      |
| Notice period policy          | `(384,)`      |

When encoded together, these vectors form a two-dimensional array with shape `(5, 384)`.

### Why encode documents in advance?

The documents remain unchanged while users ask different questions.

Encoding them once avoids repeatedly generating their embeddings for every query. Only the new question needs to be encoded each time.

For a larger application, these embeddings would normally be persisted so they would not need to be regenerated every time the application starts.

---

## 9. Understanding Cosine Similarity

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
```

Cosine similarity measures the directional alignment between two nonzero vectors.

### Formula

\[
\operatorname{cosine}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Where:

- \(A\) is the query embedding.
- \(B\) is a document embedding.
- \(A\cdot B\) is their dot product.
- \(\|A\|\) and \(\|B\|\) are their L2 norms.

### Explanation of each operation

**1. Dot product**

```python
np.dot(a, b)
```

Calculates the sum of the products of corresponding vector elements.

**2. L2 norm**

```python
np.linalg.norm(a)
```

Calculates the magnitude of vector `a`.

**3. Normalization**

Dividing the dot product by the product of the two magnitudes makes the score independent of the vectors' lengths.

For nonzero vectors, the result lies between -1 and 1.

For semantic retrieval, a higher score generally indicates greater similarity according to the embedding model. The score is not a probability and does not guarantee that a document answers the question correctly.

---

## 10. Understanding the `retrieve()` Function

This is the most important function in the retrieval stage.

```python
def retrieve(q_embedding):
    scores = []

    for i, document in enumerate(document_embeddings):
        score = cosine_similarity(q_embedding, document)
        scores.append((score, documents[i]))

    scores.sort(reverse=True)
    return scores[0]
```

Its job is to compare a query embedding against every stored document embedding and return the highest-scoring document.

### Step 1: Initialize a list

```python
scores = []
```

Creates an empty list that will store each similarity score along with its corresponding document.

### Step 2: Iterate over the embeddings

```python
for i, document in enumerate(document_embeddings):
```

`enumerate()` returns both the index and the corresponding embedding.

For example:

```text
i = 0  -> paid leave embedding
i = 1  -> office attendance embedding
i = 2  -> gym reimbursement embedding
i = 3  -> internet reimbursement embedding
i = 4  -> notice period embedding
```

Here, `document` is actually a numerical embedding vector, not the original text string.

Using a name such as `document_embedding` would make this clearer.

### Step 3: Calculate similarity

```python
score = cosine_similarity(q_embedding, document)
```

Compares the user's query embedding with the current document embedding.

### Step 4: Store the score and original text

```python
scores.append((score, documents[i]))
```

Each entry is a tuple containing:

1. The similarity score.
2. The original document text.

For example, a hypothetical entry might look like:

```python
(0.82, "Employees receive 24 days of paid leave per year.")
```

The score above is illustrative, not an actual result.

### Step 5: Sort the results

```python
scores.sort(reverse=True)
```

Sorts the tuples in descending lexicographic order. The similarity score is compared first, so the highest score normally comes first.

If scores tie, Python compares the second tuple elements as well. This can affect tie ordering.

### Step 6: Return the best match

```python
return scores[0]
```

Returns the first tuple: the highest-scoring document and its score.

**Interview takeaway:** This is a brute-force top-1 semantic retrieval method. It checks every document and returns only one result.

---

## 11. Understanding the Query

```python
query = "How much vacations can i get?"
q_embedding = model.encode(query)
```

The user asks a natural-language question.

The embedding model converts that question into a 384-dimensional vector.

The query does not need to use the exact words from the stored policy. The purpose of semantic embeddings is to identify relevant text even when the wording differs.

For example:

```text
Question: How much vacations can i get?

Document: Employees receive 24 days of paid leave per year.
```

The phrases "vacations" and "paid leave" differ, but the model may recognize their semantic relationship.

### Why encode the query separately?

The document embeddings represent the knowledge base, while the query embedding represents the user's information need.

Retrieval compares the query vector with the stored document vectors.

---

## 12. Understanding `ask_llm()`

```python
def ask_llm(question, context):

    sys_prompt = f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""

    system_message = {
        "role": "system",
        "content": sys_prompt
    }

    message = {
        "role": "user",
        "content": question
    }

    messages = [system_message, message]

    response = client.chat.completions.create(
        model=groq_model,
        messages=messages
    )

    answer = response.choices[0].message.content
    return answer
```

This function sends the user's question and the retrieved document to the language model through Groq.

### Step 1: Build the system prompt

```python
sys_prompt = f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""
```

The system prompt contains three instructions:

- Answer in one line.
- Use only the provided context.
- Avoid making unsupported claims.

It also includes the retrieved document using an f-string.

If `context` contains the paid leave policy, the prompt will contain that policy as context.

### Step 2: Create the system message

```python
system_message = {
    "role": "system",
    "content": sys_prompt
}
```

A system message provides instructions that guide the LLM's response.

### Step 3: Create the user message

```python
message = {
    "role": "user",
    "content": question
}
```

This contains the actual question asked by the user.

### Step 4: Combine both messages

```python
messages = [system_message, message]
```

The API receives the system instructions and user question as separate messages.

### Step 5: Call the LLM

```python
response = client.chat.completions.create(
    model=groq_model,
    messages=messages
)
```

This sends the request to the configured model through the Groq API.

The API returns a response object containing the generated completion and associated metadata.

### Step 6: Extract the answer

```python
answer = response.choices[0].message.content
```

This accesses the content of the first returned choice.

### Step 7: Return the answer

```python
return answer
```

The function returns the generated text to the caller.

### Important limitation

A system prompt cannot guarantee that the LLM will never hallucinate. It reduces the risk by constraining the instructions, but the model can still misinterpret the context or produce unsupported information.

For production use, you should also validate retrieval quality, handle missing evidence, and test the model's behavior.

---

## 13. Understanding the Main Execution Flow

```python
query = "How much vacations can i get?"

q_embedding = model.encode(query)

score, context = retrieve(q_embedding)

answer = ask_llm(query, context)

print(answer)
```

This is the main pipeline connecting retrieval and generation.

### Line 1: Define the question

```python
query = "How much vacations can i get?"
```

The user asks about their available vacation days.

### Line 2: Generate the query embedding

```python
q_embedding = model.encode(query)
```

Converts the question into a numerical vector.

### Line 3: Retrieve the most relevant document

```python
score, context = retrieve(q_embedding)
```

The `retrieve()` function returns a tuple.

Python unpacks the tuple into two variables:

- `score`: the highest cosine similarity score.
- `context`: the text of the matching document.

For this question, the expected relevant document is:

"Employees receive 24 days of paid leave per year."

### Line 4: Generate the answer

```python
answer = ask_llm(query, context)
```

Passes the question and retrieved document to the LLM.

The LLM generates an answer based on the supplied context.

### Line 5: Print the answer

```python
print(answer)
```

Displays the generated answer in the terminal.

### Complete execution summary

```text
Question
   |
   v
Query Embedding
   |
   v
Compare with 5 Document Embeddings
   |
   v
Select Highest Similarity Score
   |
   v
Retrieve Matching Policy Text
   |
   v
Send Question + Context to Groq
   |
   v
Extract Generated Answer
   |
   v
Print Result
```

---

## 14. Why Is This RAG Instead of Just an LLM Call?

A plain LLM application might send only the question:

```text
User question
     |
     v
     LLM
     |
     v
   Answer
```

Your RAG application adds a retrieval step:

```text
User question
     |
     v
Embedding model
     |
     v
Search the document collection
     |
     v
Relevant policy
     |
     v
Question + policy
     |
     v
     LLM
     |
     v
   Answer
```

The key difference is that the LLM receives relevant external information rather than relying only on its existing learned knowledge.

This is particularly useful for company policies, internal documentation, product manuals, and other information that may not be available in the model's training data.

---

## 15. Limitations and Improvements

Understanding the limitations is important in interviews because it shows that you can evaluate a system, not just implement it.

### Limitation 1: Only one document is retrieved

```python
return scores[0]
```

The function returns only the highest-scoring document.

This works for simple questions, but a question requiring several policies may need multiple documents.

**Improvement:** Implement top-k retrieval and return the best few documents.

### Limitation 2: No similarity threshold

The function always returns the highest-scoring document, even if all documents are irrelevant.

For example, a question about employee medical insurance could still return the least irrelevant policy even though the knowledge base contains no insurance information.

**Improvement:** Introduce a validation threshold and return a fallback response when no suitable document is found. The threshold must be calibrated on representative questions rather than chosen arbitrarily.

### Limitation 3: No persistent vector storage

The embeddings are generated in memory each time the program starts.

**Improvement:** Persist embeddings and document metadata or use a vector database for larger collections.

### Limitation 4: Brute-force search

Every query is compared with every stored embedding.

For \(N\) documents and embedding dimension \(d\), a brute-force search requires approximately \(O(Nd)\) work per query.

For five documents, this is perfectly reasonable. For millions of documents, a vector index can improve search efficiency.

### Limitation 5: The prompt does not enforce correctness

The instruction "do not hallucinate" does not guarantee factual accuracy.

**Improvement:** Provide clear context boundaries, handle missing evidence, evaluate outputs, and test adversarial or ambiguous questions.

### Limitation 6: No error handling for API failures

The code does not handle API timeouts, rate limits, authentication failures, or unavailable models.

**Improvement:** Add appropriate exception handling, logging, and retry behavior where suitable.

### Limitation 7: The context is inserted directly into the system prompt

```python
Context: {context}
```

For this small, trusted example, it is easy to understand. In a production system, retrieved documents could contain malicious or misleading instructions.

**Improvement:** Delimit the retrieved context clearly and instruct the model to treat it as untrusted reference data rather than as instructions.

### Limitation 8: No automated evaluation

The application prints an answer but does not measure retrieval accuracy or answer quality.

**Improvement:** Create a test set of questions with expected relevant documents and correct answers. Evaluate retrieval and generation separately.

---

## 16. Common Interview Questions and Answers

### Q1. What is RAG?

RAG stands for Retrieval-Augmented Generation. It retrieves relevant external information and supplies it to a language model to help generate a context-informed answer.

### Q2. What are the main components of your RAG pipeline?

The main components are the document collection, embedding model, cosine similarity function, retriever, prompt construction, Groq API call, and answer extraction.

### Q3. Why use embeddings instead of keyword matching?

Embeddings can capture semantic relationships between different expressions. For example, "vacations" and "paid leave" may be recognized as related even though the exact words differ.

Keyword matching can be useful too, especially for exact names, identifiers, or policy codes. Hybrid search can combine both methods.

### Q4. What does the embedding model do?

It converts text into numerical vectors. In this project, each document and query is represented by a 384-dimensional vector.

### Q5. Why calculate cosine similarity?

It measures the directional alignment between the query vector and each document vector. The retriever uses the score to rank the candidate documents.

### Q6. Why encode the documents before handling the query?

The document embeddings can be reused for many questions. Encoding the documents in advance avoids recomputing their vectors for every request.

### Q7. What is the purpose of the `retrieve()` function?

It compares the query embedding with all document embeddings, ranks the results by cosine similarity, and returns the highest-scoring document and its score.

### Q8. Why does `retrieve()` return both a score and a context?

The score indicates the embedding similarity, while the context contains the original text that will be passed to the LLM.

The score can also be used for retrieval diagnostics or threshold-based filtering.

### Q9. What is the purpose of the system prompt?

It instructs the LLM to answer in one line and restrict its answer to the retrieved context. These instructions guide behavior but do not guarantee perfect adherence.

### Q10. Why use Groq?

Groq provides an API through which the application can request language model completions. The code uses it to generate the final answer from the question and retrieved context.

### Q11. Is Groq the embedding model?

No. Groq is the API provider used here for generation. Sentence Transformers generates the embeddings.

### Q12. What happens if the retriever selects the wrong document?

The LLM may generate an incorrect answer because it receives irrelevant or insufficient context. Retrieval quality directly affects answer quality.

### Q13. What happens if the answer is not present in the knowledge base?

The current retriever still returns its highest-scoring document. The LLM may then give an incorrect answer despite the prompt instruction.

A stronger system should detect insufficient evidence and return an explicit fallback response.

### Q14. What is top-k retrieval?

Top-k retrieval returns the k highest-ranked documents rather than only the single best match.

For example, top-3 retrieval returns three documents that score highest for the query.

### Q15. What is the difference between a document embedding and a query embedding?

A document embedding represents stored content. A query embedding represents the user's information need. Both are produced by the same embedding model in this project so they can be compared in the same vector space.

### Q16. Does this project use a vector database?

No. It stores document embeddings in an in-memory array and searches them using a loop. A vector database is optional for a small demonstration but useful for larger systems.

### Q17. Does RAG eliminate hallucinations?

No. RAG can reduce the need for the model to rely on unsupported internal knowledge, but it cannot eliminate hallucinations. Retrieval failures, ambiguous context, and generation errors can still occur.

### Q18. Is this a complete production-ready RAG system?

No. It is a basic working demonstration of retrieval and generation. It lacks persistent vector storage, top-k retrieval, calibrated relevance thresholds, robust error handling, and systematic evaluation.

### Q19. What is the difference between retrieval and generation?

Retrieval selects relevant information from the knowledge base. Generation uses the retrieved information and the user's question to produce a natural-language answer.

### Q20. How would you scale this application?

I would persist the embeddings, use a vector index or database, implement top-k retrieval, calibrate relevance thresholds, add error handling, and evaluate retrieval accuracy and answer faithfulness. I would consider hybrid search or reranking if the quality of retrieval required further improvement.

---

## 17. Quick Revision Sheet

| Concept                                 | Remember                                           |
| --------------------------------------- | -------------------------------------------------- |
| RAG                                     | Retrieval-Augmented Generation                     |
| Knowledge base                          | List of employee policy documents                  |
| Embedding model                         | `all-MiniLM-L6-v2`                               |
| Embedding dimension                     | 384                                                |
| Document embeddings                     | Numerical representations of stored documents      |
| Query embedding                         | Numerical representation of the user's question    |
| Cosine similarity                       | Measures vector directional alignment              |
| Retriever                               | Finds the highest-scoring document                 |
| Top-1 retrieval                         | Returns one best match                             |
| Context                                 | Retrieved text provided to the LLM                 |
| Groq                                    | API used to access the generation model            |
| System prompt                           | Guides the LLM's response                          |
| `response.choices[0].message.content` | Extracts the generated answer                      |
| Main limitation                         | No relevance threshold or fallback retrieval logic |
| Vector database                         | Not used in this implementation                    |
| Complete RAG flow                       | Retrieve → Augment → Generate                    |

---

## 18. The 30-Second Interview Explanation

"I built a basic RAG pipeline using Sentence Transformers and the Groq API. First, I encoded a small collection of employee policy documents into 384-dimensional embeddings using `all-MiniLM-L6-v2`. When a user asks a question, I encode the query and calculate cosine similarity against every stored document embedding. I retrieve the highest-scoring document and pass it, along with the question, to a language model through Groq. The model generates an answer based on the supplied context. The current version uses brute-force top-1 retrieval and in-memory storage, so the next improvements would be top-k retrieval, relevance thresholds, persistent vector storage, and evaluation."

---

## 19. Suggested Next Steps

Build the project in this order:

1. **Add top-k retrieval:** Return the three most relevant documents instead of just one.
2. **Add a relevance threshold:** Handle questions for which the knowledge base has no suitable answer.
3. **Improve the prompt:** Clearly delimit the context and define what to do when evidence is insufficient.
4. **Add a test dataset:** Measure whether the correct document is retrieved for representative questions.
5. **Add persistent storage:** Save embeddings so they do not need to be recomputed at every startup.
6. **Build a more complete RAG system:** Add document ingestion, chunking, a vector index, and systematic evaluation.

The most important interview distinction to remember is this: **embeddings represent the text, retrieval finds the relevant information, and the LLM generates the answer.**
