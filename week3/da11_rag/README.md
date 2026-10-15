
# Basic RAG (Retrieval-Augmented Generation)

## 1. Project Overview

This project demonstrates the basic concept of **Retrieval-Augmented Generation (RAG)** using Python and the Groq API.

The application retrieves relevant information from a small, predefined knowledge base and sends that information to a Large Language Model (LLM) to generate an answer.

Instead of relying entirely on the LLM's internal knowledge, the model is instructed to answer based on the retrieved context.

### Technologies Used

- **Python** — application logic
- **Groq API** — access to an LLM
- **python-dotenv** — load API keys from environment variables

## 2. What Is RAG?

**RAG stands for Retrieval-Augmented Generation.**

It is a technique that combines information retrieval with the text-generation capabilities of an LLM.

RAG has three basic steps:

1. **Retrieve:** Find relevant information from a knowledge source.
2. **Augment:** Add the retrieved information to the LLM's prompt as context.
3. **Generate:** Ask the LLM to produce an answer using that context.

### Simple Example

Suppose our knowledge base contains:

```python
knowledge_base = {
    "age": "the age of pratyush is 25 years",
    "net_worth": "the net worth of pratyush is 2000"
}
```

When the user asks, `What is the age?`, the application retrieves the age information and sends it to the LLM.

**Output:**

```text
The age of Pratyush is 25 years.
```

The LLM receives the relevant information as context instead of having to answer from its general knowledge alone.

## 3. How the Code Works

### Step 1: Load the API Key

```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api key not found")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"
```

**Explanation:**

- `load_dotenv()` loads environment variables from the `.env` file.
- `os.getenv()` retrieves the Groq API key.
- The `if` condition checks whether the API key exists.
- `Groq()` creates the API client.
- `model` specifies which LLM to use.

The API key should be stored in `.env`, not hardcoded into the Python source code.

Example `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

### Step 2: Create the Knowledge Base

```python
knowledge_base = {
    "age": "the age of pratyush is 25 years",
    "net_worth": "the net worth of pratyush is 2000"
}
```

The knowledge base stores the information that the application can retrieve.

In this basic implementation, the knowledge base is a Python dictionary.

In a more advanced RAG system, the knowledge source might contain PDF files, documents, websites, or database records. The retrieved information might be stored in a vector database.

### Step 3: Retrieve Relevant Information

```python
def retrieve_info(question):
    question = question.lower()

    if "age" in question:
        return knowledge_base["age"]

    elif "net worth" in question:
        return knowledge_base["net_worth"]

    else:
        return None
```

This function searches for keywords in the user's question.

**How it works:**

1. Converts the question to lowercase.
2. Checks whether the question contains `"age"`.
3. If it does, returns the age information.
4. Otherwise, checks whether it contains `"net worth"`.
5. If neither keyword is found, returns `None`.

This is the **retrieval step** in our basic RAG implementation.

**Important limitation:** This is rule-based retrieval, not semantic search. It matches specific keywords rather than understanding the meaning of a question.

### Step 4: Add the Retrieved Context to the Prompt

```python
def ask_llm(question):
    context = retrieve_info(question)

    system_msg = {
        "role": "system",
        "content": (
            "answer in one line only and answer only based "
            "on this context. Do not hallucinate. "
            f"Context: {context}"
        )
    }

    message = {
        "role": "user",
        "content": question
    }

    messages = [system_msg, message]
```

First, `retrieve_info(question)` finds the relevant information.

Then, the retrieved information is inserted into the system message as context.

The system message tells the LLM to:

- Answer in one line.
- Use only the supplied context.
- Avoid making unsupported claims.

The user's original question is sent separately as a user message.

This is the **augmentation step**: the retrieved information is added to the prompt before generation.

### Step 5: Generate the Answer

```python
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    answer = response.choices[0].message.content
    return answer
```

The application sends the system and user messages to the Groq API.

The LLM generates a response, which is extracted from the API result and returned to the caller.

This is the **generation step**.

The instruction to avoid hallucinations helps guide the model, but it cannot guarantee that every answer will be correct.

### Step 6: Ask a Question

```python
question = "what is age?"
print(ask_llm(question))
```

The function retrieves the age information, passes it to the LLM, and prints the generated answer.

## 4. Complete RAG Workflow

```text
User Question
     |
     v
retrieve_info(question)
     |
     v
Retrieve Relevant Context
     |
     v
Add Context to System Prompt
     |
     v
Send Prompt + User Question to LLM
     |
     v
Generate Answer
     |
     v
Display Answer
```

## 5. Why Do We Need RAG?

An LLM may not know private, company-specific, or recently updated information.

RAG helps by providing relevant information from an external knowledge source at query time.

### Main Benefits

- **Grounded answers:** The model can use retrieved information as evidence.
- **External knowledge:** Information can be supplied without retraining the LLM.
- **Reduced unsupported answers:** Relevant context can help reduce hallucinations.
- **Easier knowledge updates:** The knowledge source can be updated separately from the model.

RAG does not automatically guarantee factual accuracy; the quality of retrieval and the model's response still matter.

## 6. Limitations of This Project

This project demonstrates the basic RAG concept, but it is not a complete production-grade RAG system.

1. **Keyword-based retrieval:** It only recognizes predefined keywords.
2. **Small knowledge base:** The data is manually written in a Python dictionary.
3. **No embeddings:** The project does not convert text into vector representations.
4. **No vector database:** It does not use FAISS, ChromaDB, Pinecone, or another vector store.
5. **No document chunking:** The project does not split large documents into smaller chunks.
6. **No retrieval evaluation:** It does not measure whether the retrieved information is relevant.
7. **Missing-context handling:** When no keyword matches, the context becomes `None`. The LLM may still generate an answer, so explicit handling should be added.

## 7. Basic RAG vs. Advanced RAG

| Feature          | This Project               | Advanced RAG                           |
| ---------------- | -------------------------- | -------------------------------------- |
| Knowledge source | Python dictionary          | Documents, databases, or other sources |
| Retrieval        | Keyword matching           | Semantic or hybrid search              |
| Embeddings       | Not used                   | Commonly used                          |
| Vector database  | Not used                   | Often used                             |
| Context          | Retrieved dictionary value | Relevant document chunks               |
| Generation       | Groq API                   | An LLM API or a locally hosted model   |

**Interview tip:** A system does not need a vector database to demonstrate the core RAG idea. This project is a simplified, rule-based implementation of retrieval followed by context-grounded generation.

## 8. Common Interview Questions and Answers

### Q1. What is RAG?

RAG stands for Retrieval-Augmented Generation. It retrieves relevant information from an external knowledge source and provides that information to an LLM as context so that the model can generate a more grounded answer.

### Q2. Explain your project in 30 seconds.

I built a basic RAG application using Python and the Groq API. I created a small knowledge base containing information about age and net worth. When a user asks a question, a retrieval function searches for relevant information using keywords. The retrieved information is added to the LLM's system prompt, and the model generates an answer using that context.

### Q3. What are the three main steps of RAG?

Retrieval, augmentation, and generation.

### Q4. What is the role of the knowledge base?

It stores the external information that the application retrieves and supplies to the LLM.

### Q5. What does the `retrieve_info()` function do?

It checks the user's question for predefined keywords and returns the corresponding information from the knowledge base.

### Q6. Why do you pass context to the LLM?

The context supplies relevant external information that the model can use to answer the user's question instead of relying only on its pretrained knowledge.

### Q7. What is hallucination in an LLM?

Hallucination occurs when an LLM generates information that is false, unsupported, or not grounded in the available evidence.

### Q8. Does your project completely prevent hallucinations?

No. I instruct the LLM to answer only from the supplied context, but prompt instructions alone cannot guarantee that every response is accurate.

### Q9. Does your project use embeddings?

No. It uses simple keyword matching. Embeddings would represent text as numerical vectors, which could be compared to find semantically similar information.

### Q10. What is a vector database?

A vector database stores and retrieves vector representations of data. In RAG, it can help find document chunks that are semantically similar to a user's query.

### Q11. What is the difference between RAG and fine-tuning?

RAG supplies relevant external information to the model at query time. Fine-tuning updates the model's learned parameters using training examples. They solve different problems and can be used together.

### Q12. What happens when the retrieval function returns `None`?

The current implementation inserts `None` into the context. The LLM might still attempt to answer. A better implementation would detect missing context and return a message such as, "I don't have enough information to answer that question."

### Q13. Why use Groq?

Groq provides an API through which an application can access supported language models. In this project, the Groq Python client sends the messages to the selected model and retrieves the response.

### Q14. How would you improve this project?

I would load real documents, split them into chunks, generate embeddings, store them in a vector database, retrieve the most relevant chunks, and provide them to the LLM. I would also add missing-context handling and evaluate retrieval quality and answer accuracy.

### Q15. Is your project a complete RAG pipeline?

It demonstrates the core idea of retrieval, context augmentation, and generation, but it uses simple keyword-based retrieval. It does not implement the embeddings and semantic-search components commonly found in larger RAG applications.

## 9. Quick Revision Before an Interview

Remember these points:

- **RAG:** Retrieval-Augmented Generation.
- **Retrieval:** Find relevant information.
- **Augmentation:** Add retrieved information to the prompt.
- **Generation:** Let the LLM produce an answer.
- **Knowledge base:** Stores external information.
- **Context:** Retrieved information provided to the model.
- **Hallucination:** Unsupported or incorrect model-generated information.
- **Embeddings:** Numerical representations of text.
- **Vector database:** Stores and searches vector representations.
- **Your retrieval method:** Keyword-based, rule-driven matching.
- **Your main limitation:** It does not perform semantic search.
- **Your next improvement:** Document chunking, embeddings, vector search, and missing-context handling.

### One-line summary

**This project demonstrates how retrieved external information can be added to an LLM prompt to generate a context-grounded answer.**
