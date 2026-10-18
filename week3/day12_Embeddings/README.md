
# Sentence Embeddings and Cosine Similarity

## 1. Project Overview

This project demonstrates how to convert text into numerical vectors called **embeddings** and compare the semantic similarity between two pieces of text using cosine similarity.

It uses the `all-MiniLM-L6-v2` model from Sentence Transformers.

### What does this project do?

1. Loads a pre-trained sentence embedding model.
2. Converts a sentence into a 384-dimensional vector.
3. Displays the vector's shape and first 10 values.
4. Converts two different sentences into embeddings.
5. Calculates cosine similarity between those embeddings.

### Technologies Used

- Python
- NumPy
- Sentence Transformers
- Pre-trained `all-MiniLM-L6-v2` model

---

## 2. Installation and Setup

### Step 1: Install dependencies

```bash
pip install sentence-transformers numpy python-dotenv
```

### Step 2: Save the code

Create a Python file named `main.py` and add the following code:

```python
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Machine Learning is fun!"

embedding = model.encode(text)

print(embedding.shape)
print(embedding[:10])

t1 = "There are 24 paid leaves"
t2 = "cat is a wild animal"

v1 = model.encode(t1)
v2 = model.encode(t2)

print(cosine_similarity(v1, v2))
```

### Step 3: Run the program

```bash
python main.py
```

**Note:** The first run may download the pre-trained model. An internet connection is generally required for the initial download.

The imports `os`, `Path`, and `load_dotenv` are currently unused. You can remove them from this particular script.

---

## 3. What Is an Embedding?

An embedding is a numerical representation of data, such as text, images, or audio, in a vector space.

Computers cannot directly perform mathematical operations on the meaning of a sentence. Embeddings represent text numerically so that mathematical operations can help compare its meaning.

For example:

- `"I love machine learning"`
- `"I enjoy studying AI"`

These sentences use different words but express similar ideas. A good sentence embedding model can represent their meanings with vectors that are relatively close together.

### Remember this for interviews

**Embedding = Text converted into a numerical vector that represents semantic information.**

Embeddings are commonly used in:

- Semantic search
- Retrieval-Augmented Generation (RAG)
- Recommendation systems
- Document clustering
- Duplicate detection
- Question-answering systems

---

## 4. Understanding the Model

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

This line loads a pre-trained sentence embedding model.

### What is `all-MiniLM-L6-v2`?

It is a compact, pre-trained sentence transformer model designed to generate sentence embeddings.

| Property            | Explanation                        |
| ------------------- | ---------------------------------- |
| Model name          | `all-MiniLM-L6-v2`               |
| Architecture family | MiniLM                             |
| Transformer layers  | 6                                  |
| Embedding dimension | 384                                |
| Main purpose        | Sentence and short-text embeddings |

The model maps a sentence to a fixed-size vector containing 384 numerical values.

### Why use a pre-trained model?

A pre-trained model has already learned useful language representations from training data. We can use it to generate embeddings without training a new model from scratch.

**Interview answer:** "I used `all-MiniLM-L6-v2` because it is a relatively lightweight sentence embedding model that produces 384-dimensional vectors and is useful for semantic similarity and retrieval tasks."

---

## 5. Understanding `model.encode()`

```python
text = "Machine Learning is fun!"

embedding = model.encode(text)
```

The `encode()` method converts the input text into a numerical embedding.

Conceptually:

```text
"Machine Learning is fun!"
              |
              v
     Sentence Transformer
              |
              v
   384-dimensional vector
              |
              v
[0.12, -0.04, 0.08, ..., 0.02]
```

The values above are illustrative, not the actual output of the model.

### Why are there 384 numbers?

The model's output dimension is 384. Therefore, one sentence is represented by 384 numerical features.

```python
print(embedding.shape)
```

Expected shape:

```text
(384,)
```

The comma matters: `(384,)` means a one-dimensional array containing 384 elements.

It is not a 384-by-384 matrix.

### What does this line do?

```python
print(embedding[:10])
```

It prints only the first 10 elements of the vector.

The slice `[:10]` means: start at index 0 and stop before index 10.

This displays 10 values, not the entire embedding.

### Important distinction

An embedding is not a list of word meanings or a probability distribution. Its individual dimensions generally do not have simple, human-readable meanings. The vector is useful because of the relationships between vectors in the embedding space.

---

## 6. What Is Cosine Similarity?

Cosine similarity measures the cosine of the angle between two nonzero vectors.

It is commonly used to estimate how similar two text embeddings are.

### Formula

\[
\operatorname{cosine}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Where:

- \(A\) = first embedding vector
- \(B\) = second embedding vector
- \(A\cdot B\) = dot product of the two vectors
- \(\|A\|\) = magnitude, or L2 norm, of vector A
- \(\|B\|\) = magnitude, or L2 norm, of vector B

### Understanding the formula in Python

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
```

**Part 1: Dot product**

```python
np.dot(a, b)
```

Multiplies corresponding vector elements and adds the results.

**Part 2: Magnitude of vector A**

```python
np.linalg.norm(a)
```

Calculates the L2 norm, or Euclidean magnitude, of vector A.

**Part 3: Magnitude of vector B**

```python
np.linalg.norm(b)
```

Calculates the magnitude of vector B.

**Part 4: Divide the dot product by both magnitudes**

```python
np.dot(a, b) / (
    np.linalg.norm(a) * np.linalg.norm(b)
)
```

This normalizes the dot product so that the comparison depends on the vectors' directions rather than simply their lengths.

### How to interpret the result

For nonzero vectors, cosine similarity lies between -1 and 1.

| Score       | General interpretation               |
| ----------- | ------------------------------------ |
| Close to 1  | Vectors point in similar directions  |
| Close to 0  | Vectors are approximately orthogonal |
| Close to -1 | Vectors point in opposite directions |

For sentence embeddings, these are mathematical interpretations, not universal semantic thresholds. A score of 0 does not automatically mean two sentences have absolutely no relationship.

---

## 7. Understanding the Two Sentences

```python
t1 = "There are 24 paid leaves"
t2 = "cat is a wild animal"

v1 = model.encode(t1)
v2 = model.encode(t2)

print(cosine_similarity(v1, v2))
```

Let's understand what happens.

### Step 1: Encode the first sentence

```python
v1 = model.encode(t1)
```

Converts the sentence about paid leaves into a 384-dimensional vector.

### Step 2: Encode the second sentence

```python
v2 = model.encode(t2)
```

Converts the sentence about a cat into another 384-dimensional vector.

### Step 3: Compare the vectors

```python
cosine_similarity(v1, v2)
```

Calculates how closely their vector directions align.

Since the sentences discuss different topics, we would generally expect a relatively low similarity score compared with two sentences expressing similar meanings. The exact value depends on the model and the text; it should not be guessed in advance.

### Why doesn't it return 0 for unrelated sentences?

This is an important interview question.

**Unrelated sentences do not necessarily produce a cosine similarity of zero.**

A score of exactly zero means that the vectors are orthogonal. Semantic unrelatedness does not mathematically require orthogonality.

The embedding model may also encode general language patterns or associations shared by the sentences.

---

## 8. A Better Experiment to Understand Similarity

Try comparing sentences that express the same meaning in different words.

```python
sentences = [
    "There are 24 paid leaves",
    "Employees receive 24 days of paid leave",
    "A cat is a wild animal",
]

embeddings = model.encode(sentences)

print(cosine_similarity(embeddings[0], embeddings[1]))
print(cosine_similarity(embeddings[0], embeddings[2]))
```

The first comparison is between two sentences about paid leave. The second compares a sentence about paid leave with a sentence about a cat.

Generally, the first pair should be more semantically similar than the second pair. Actual scores can vary, and the model may not interpret every domain-specific expression perfectly.

**Experiment goal:** Observe how changing the wording while preserving meaning differs from changing the topic.

---

## 9. Is This RAG?

No. This code demonstrates two building blocks that can be used in a RAG system: embedding generation and similarity comparison.

A basic RAG pipeline typically contains:

1. **Document loading:** Collect documents or text.
2. **Chunking:** Split long documents into smaller sections.
3. **Embedding generation:** Convert chunks into vectors.
4. **Vector storage:** Store embeddings and their associated text.
5. **Retrieval:** Find relevant chunks for a user's query.
6. **Augmentation:** Add retrieved context to the prompt.
7. **Generation:** Use a language model to generate an answer based on the context.

Your code only generates embeddings and compares two vectors. It does not store a document collection, retrieve relevant documents for a query, or generate an answer with an LLM.

### How this code fits into RAG

```text
Document
   |
   v
Embedding Model
   |
   v
Document Vector

User Query
   |
   v
Embedding Model
   |
   v
Query Vector
   |
   v
Similarity Search
   |
   v
Relevant Documents
```

A complete RAG system continues from retrieval to context construction and answer generation.

---

## 10. Common Interview Questions and Answers

### Q1. What is an embedding?

An embedding is a numerical vector representation of data that captures useful relationships or semantic information. In NLP, sentence embeddings help compare the meanings of text.

### Q2. Why do we convert text into embeddings?

To represent text numerically so that mathematical operations and similarity searches can be performed efficiently.

### Q3. What is the dimension of your embedding?

The embedding dimension is 384 because `all-MiniLM-L6-v2` produces a vector of 384 values for each sentence.

### Q4. What does `embedding.shape` return?

It returns `(384,)`, which indicates a one-dimensional NumPy array with 384 elements.

### Q5. What does `embedding[:10]` do?

It extracts the first 10 elements of the embedding array.

### Q6. What is cosine similarity?

It is the cosine of the angle between two nonzero vectors. It is commonly used to compare text embeddings because it measures their directional alignment.

### Q7. Why use cosine similarity instead of the raw dot product?

A raw dot product is affected by vector magnitudes. Cosine similarity divides by the product of the vector magnitudes, making the score independent of their lengths.

### Q8. Does a higher cosine similarity always mean the sentences have the same meaning?

No. A higher score often indicates greater similarity according to the embedding model, but it does not prove that the sentences are equivalent. The model can make mistakes, and similarity scores depend on the model and domain.

### Q9. Is cosine similarity the same as Euclidean distance?

No. Cosine similarity measures directional alignment, while Euclidean distance measures the straight-line distance between vectors. For unit-normalized vectors, they are related by:

\[
\|A-B\|^2=2(1-\cos(A,B))
\]

This relationship holds when both vectors have unit norm.

### Q10. What is the role of NumPy?

NumPy performs numerical calculations. In this code, `np.dot()` calculates the dot product and `np.linalg.norm()` calculates vector magnitudes.

### Q11. Does this code train a model?

No. It loads a pre-trained model and uses it for inference to generate embeddings. There is no training loop or model fine-tuning in this code.

### Q12. What happens if one of the vectors has zero magnitude?

The denominator becomes zero, so cosine similarity is undefined and the calculation can produce a division-by-zero warning or a non-finite result. A robust implementation should check for zero norms.

### Q13. Why not compare the original sentences directly?

Ordinary string comparison checks characters or exact text matches. Embeddings let us compare numerical representations of text, making it possible to recognize similarity even when sentences use different words.

### Q14. Where are embeddings stored in this code?

They are stored in memory as NumPy arrays in variables such as `embedding`, `v1`, and `v2`. The code does not use a vector database.

### Q15. What is the difference between an embedding model and an LLM?

An embedding model converts input into vectors designed for tasks such as retrieval and similarity. A generative LLM produces text, such as answers, explanations, or summaries. Some models and architectures can support multiple capabilities, but the roles are distinct.

---

## 11. Important Limitations

- The code compares only two sentences at a time.
- It does not store or search a collection of documents.
- It does not use a vector database.
- It does not include a generative LLM.
- It does not handle zero-norm vectors.
- Similarity scores are not guaranteed to correspond perfectly to human judgments of meaning.

For a larger retrieval system, you could use a vector database or a nearest-neighbor index to search among many embeddings.

---

## 12. Quick Revision Sheet

Review these points immediately before an interview.

| Concept                | Remember                                           |
| ---------------------- | -------------------------------------------------- |
| Embedding              | Numerical vector representation of text            |
| Model                  | `all-MiniLM-L6-v2`                               |
| Embedding dimension    | 384                                                |
| `model.encode(text)` | Converts text into an embedding                    |
| `embedding.shape`    | `(384,)` for one sentence                        |
| `embedding[:10]`     | First 10 vector values                             |
| `np.dot(a, b)`       | Dot product                                        |
| `np.linalg.norm(a)`  | L2 norm of vector A                                |
| Cosine similarity      | Dot product divided by both vector norms           |
| Higher similarity      | Usually indicates stronger directional alignment   |
| Zero similarity        | Orthogonal vectors, not necessarily unrelated text |
| RAG                    | Retrieval + context augmentation + generation      |
| Current project        | Embedding generation and similarity demonstration  |

### The 30-second interview explanation

"I used Sentence Transformers with the `all-MiniLM-L6-v2` model to convert sentences into 384-dimensional embeddings. I then implemented cosine similarity using NumPy. The function calculates the dot product of two embeddings and divides it by the product of their L2 norms. This provides a normalized measure of how closely the vectors align. This technique is useful in semantic search and RAG, although my current implementation demonstrates only embedding generation and pairwise similarity, not a complete RAG pipeline."

---

## 13. Suggested Next Steps

To build on this project, work through the following progression:

1. Compare multiple sentences and observe their similarity scores.
2. Add a check for zero-norm vectors.
3. Build a small semantic search system over 10–20 text documents.
4. Store and retrieve document embeddings using a vector index.
5. Connect retrieval to an LLM to build a basic RAG application.

The most important thing to understand is not just the code syntax. You should be able to explain **why embeddings are useful, what cosine similarity measures, and how these concepts fit into semantic search and RAG**.
