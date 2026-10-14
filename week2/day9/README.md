
# Groq API: Non-Streaming vs. Streaming Responses

## 1. Project Overview

This project demonstrates how to interact with a Large Language Model (LLM) using the Groq API in Python.

The user enters a prompt, and the program sends it to the selected model. It then demonstrates two ways to receive the model's response:

1. **Non-streaming:** Wait for the complete response and print it.
2. **Streaming:** Receive and print the response incrementally as chunks arrive.

**Technologies used**

- Python
- Groq API
- `groq` Python SDK
- `python-dotenv`
- Environment variables

**Model:** `openai/gpt-oss-120b`

> Model availability may change. Use a model currently supported by your Groq account.

---

## 2. Project Workflow

```text
Start
  |
  v
Load .env file
  |
  v
Read GROQ_API_KEY
  |
  v
Initialize Groq client
  |
  v
Take user input
  |
  v
Create message dictionary
  |
  v
Send prompt to Groq
  |
  +-----------------------+
  |                       |
  v                       v
Non-streaming           Streaming
  |                       |
  v                       v
Wait for complete       Receive chunks
response                incrementally
  |                       |
  v                       v
Print complete          Print each chunk
answer                  immediately
```

---

## 3. Environment Setup

### Install dependencies

```bash
python -m pip install groq python-dotenv
```

### Configure the API key

Create a `.env` file in your project directory:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

Never commit `.env` to GitHub. Add it to `.gitignore`.

### Load the environment variable

```python
import os
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
```

- `load_dotenv()` loads environment variables from the `.env` file.
- `os.getenv()` retrieves the API key.
- Keeping the key outside the source code helps protect the credential.

### Validate the API key

```python
if not my_api_key:
    raise ValueError("API key does not exist")
```

This stops execution if the API key is missing.

### Initialize the client

```python
from groq import Groq

client = Groq(api_key=my_api_key)
```

The client is used to send requests to the Groq API.

---

## 4. Taking User Input

```python
prompt = input("Just tell me what you want to know?\n")
```

The `input()` function pauses program execution and waits for the user to enter text.

The entered text is stored in the `prompt` variable.

For example:

```text
Just tell me what you want to know?
Explain Python generators.
```

The variable now contains:

```python
prompt = "Explain Python generators."
```

This makes the program interactive because the user can enter different prompts without changing the source code.

---

## 5. Understanding Message Structure

```python
role = "user"

message = {
    "role": role,
    "content": prompt
}

messages = [message]
```

### What is a dictionary?

A Python dictionary stores data as key-value pairs.

The `message` dictionary looks like this:

```python
{
    "role": "user",
    "content": "Explain Python generators."
}
```

- `role` identifies who is sending the message.
- `content` contains the actual text.

### Why use a list?

```python
messages = [message]
```

The Chat Completions API expects a list of messages.

Even when there is only one message, it must be placed inside a list.

The list can also contain multiple messages, such as system instructions, user prompts, and previous assistant responses.

**Important:** This implementation sends only a user message. It does not define a system message.

---

## 6. Non-Streaming Response

### Code

```python
response = client.chat.completions.create(
    model=model,
    messages=messages
)

print(response.choices[0].message.content)
```

### How it works

1. The program sends the prompt to Groq.
2. The API processes the request.
3. The program waits for the completion response.
4. The complete response is received.
5. The generated text is printed.

### Understanding the response object

```python
response.choices[0].message.content
```

- `response`: The API response object.
- `choices`: The list of completion choices returned by the API.
- `[0]`: Selects the first choice.
- `message`: Accesses the generated message.
- `content`: Retrieves the generated text.

### Example

Input:

```text
Explain Python generators.
```

Output:

```text
Python generators are functions that produce values lazily
using the yield keyword. They allow iteration over values
without storing the entire sequence in memory.
```

The exact output depends on the model.

### Key characteristic

**Non-streaming waits for the complete response before your program prints the answer.**

This is useful when the application needs the full result before processing it.

---

## 7. Streaming Response

### Code

```python
stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content

    if content:
        print(content, end="", flush=True)
```

### What is streaming?

Streaming allows the API to deliver a response incrementally rather than waiting for the entire completion.

Instead of receiving one complete answer, the program processes multiple chunks.

### Understanding `stream=True`

```python
stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True
)
```

Setting `stream=True` requests a streaming response.

The returned stream can be iterated over to process the incoming chunks.

### Understanding the `for` loop

```python
for chunk in stream:
```

The loop processes each chunk as it becomes available.

A chunk is a part of the streamed response. It may contain text, metadata, or no text content.

### Understanding `delta.content`

```python
content = chunk.choices[0].delta.content
```

In a streaming completion, `delta` contains the incremental update for a choice.

`delta.content` contains the text fragment when the chunk carries text.

Unlike non-streaming responses, which use `message.content`, streaming responses use `delta.content`.

### Why use `if content`?

```python
if content:
```

Not every chunk contains text. Some chunks may contain metadata or indicate completion.

This condition prevents the program from trying to print an empty or missing text fragment.

### Understanding `end=""`

```python
print(content, end="")
```

By default, Python's `print()` adds a newline after each call.

Setting `end=""` prevents a newline from being inserted after every chunk, so the fragments appear as one continuous answer.

### Understanding `flush=True`

```python
print(content, end="", flush=True)
```

`flush=True` tells Python to flush the output buffer immediately.

This helps the user see text as it arrives rather than waiting for buffered output to appear.

**Key characteristic:** Streaming improves perceived responsiveness because users can begin reading before the complete answer is generated.

---

## 8. Non-Streaming vs. Streaming

| Feature                  | Non-streaming                    | Streaming                               |
| ------------------------ | -------------------------------- | --------------------------------------- |
| API parameter            | `stream` defaults to `False` | `stream=True`                         |
| Response handling        | Complete response object         | Sequence of chunks                      |
| Text access              | `message.content`              | `delta.content`                       |
| Printing                 | After completion                 | Incrementally                           |
| Perceived responsiveness | User waits for the full answer   | Text appears as it arrives              |
| Processing               | Simple                           | Requires iteration                      |
| Best use                 | Short answers, batch tasks       | Chat interfaces, interactive assistants |

Both approaches use the same underlying model and can produce similar content. Streaming changes how the response is delivered, not the fundamental task performed by the model.

Streaming does not necessarily reduce the model's total generation time; it allows output to be displayed earlier.

---

## 9. Understanding Python Iteration and Generators

The streaming section introduces an important Python concept: iteration.

```python
for chunk in stream:
    ...
```

An iterable provides elements one at a time when iterated over.

A generator is a particular kind of iterator that can produce values lazily, often using `yield`.

The Groq SDK's streaming object is iterable. The loop retrieves its chunks as they become available.

**Interview distinction:** Do not claim that the `stream` variable is definitely a Python generator. It is safer to say that it is an iterable streaming response supplied by the SDK.

### Why is incremental processing useful?

Suppose a model generates a long answer.

Without streaming, the application waits until the completion is ready.

With streaming, it can process each arriving chunk instead of waiting for the entire response.

This is useful for:

- Chatbots
- AI assistants
- Interactive coding tools
- Long-form text generation
- Applications that display a live response

---

## 10. Important Python Concepts

| Concept               | Usage                                  |
| --------------------- | -------------------------------------- |
| `import`            | Imports required libraries             |
| Environment variables | Store the API key outside source code  |
| `input()`           | Accepts user input                     |
| Variables             | Store the prompt, client, and response |
| Dictionary            | Represents an API message              |
| List                  | Holds messages                         |
| Function calls        | Sends API requests                     |
| Object attributes     | Access response fields                 |
| `for` loop          | Processes streamed chunks              |
| Conditional           | Checks whether a chunk contains text   |
| `end=""`            | Prevents a newline after each chunk    |
| `flush=True`        | Flushes buffered terminal output       |

---

## 11. Common Errors and Improvements

### Missing API key

Possible error:

```text
ValueError: API key does not exist
```

Check that `.env` contains `GROQ_API_KEY` and that `load_dotenv()` loads the correct file.

### Invalid model

A model-not-found error can occur if the selected model is no longer available or your account cannot access it.

Check the current model list in the official Groq documentation.

### Rate limits or service errors

Requests may fail because of rate limits, temporary service problems, or connectivity issues.

Production code should handle API exceptions and retry only appropriate transient errors.

### Empty streaming chunks

Some chunks may have no text content, so the condition below is useful:

```python
if content:
    print(content, end="", flush=True)
```

### Unused imports

The current script imports `json` and `Path`, but neither is used.

They can be removed:

```python
import os
from dotenv import load_dotenv
from groq import Groq
```

### Avoid duplicate API calls when unnecessary

Your program calls the model twice: once for non-streaming and once for streaming.

This is intentional for demonstration, but it creates two API requests and may consume additional quota. In a real application, choose one approach based on the user experience you need.

---

# Interview Questions and Answers

## Q1. What is the purpose of this project?

**Answer:** This project demonstrates how to integrate an LLM using the Groq API. It accepts a user prompt and displays the model's response using both non-streaming and streaming approaches.

## Q2. What is the difference between streaming and non-streaming?

**Answer:** Non-streaming returns the completed response for processing, whereas streaming delivers incremental chunks that the application can process as they arrive.

## Q3. Why did you use `stream=True`?

**Answer:** I used `stream=True` to receive the response incrementally, allowing the program to display generated text before the complete answer is available.

## Q4. Why do you use `response.choices[0].message.content` in non-streaming mode?

**Answer:** It selects the first completion choice and retrieves the generated message's text.

## Q5. Why do you use `chunk.choices[0].delta.content` in streaming mode?

**Answer:** Streaming responses contain incremental updates in `delta`. Its `content` field holds the text fragment for that chunk.

## Q6. What is the purpose of `flush=True`?

**Answer:** It flushes Python's output buffer so that each text fragment is displayed promptly.

## Q7. Why do you use `end=""`?

**Answer:** It prevents Python from adding a newline after each chunk, allowing the fragments to appear as continuous text.

## Q8. Why is the message represented as a dictionary?

**Answer:** The API expects each message to contain fields such as `role` and `content`. A dictionary represents these key-value pairs.

## Q9. Why is the message placed inside a list?

**Answer:** The API accepts a list of messages, allowing a conversation to contain multiple messages and roles.

## Q10. What is the purpose of `load_dotenv()`?

**Answer:** It loads variables from the `.env` file into the environment so the application can retrieve its API key without hardcoding it.

## Q11. Does streaming make the model generate the answer faster?

**Answer:** Not necessarily. Streaming allows the application to display partial output earlier, improving perceived responsiveness, but it does not guarantee a shorter total generation time.

## Q12. Why is the code calling the API twice?

**Answer:** The script demonstrates both response-handling methods independently. In a production application, I would normally select one method to avoid making unnecessary duplicate requests.

## Q13. What is `chunk`?

**Answer:** It is an individual streaming response object yielded by the SDK's iterable stream. It may contain an incremental text update or other response information.

## Q14. Is `stream` necessarily a generator?

**Answer:** It is an iterable streaming response from the SDK. I would not assume its concrete type is a generator without checking the SDK implementation.

## Q15. What improvements would you make for production?

**Answer:** I would add exception handling, logging, configurable model selection, request timeouts where supported, appropriate retry logic, and tests. I would also ensure the API key is protected and avoid making duplicate requests.

---

# 12. One-Minute Revision

Remember these core differences:

**Non-streaming**

```python
response = client.chat.completions.create(
    model=model,
    messages=messages
)

print(response.choices[0].message.content)
```

- Wait for the complete response.
- Extract text from `message.content`.
- Print the answer after completion.

**Streaming**

```python
stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content

    if content:
        print(content, end="", flush=True)
```

- Receive incremental chunks.
- Extract text from `delta.content`.
- Print each fragment as it arrives.

**Final interview takeaway:** Streaming is a response-delivery technique. It lets an application display generated text incrementally, while non-streaming makes the complete response available before it is displayed. Both approaches can use the same LLM.
