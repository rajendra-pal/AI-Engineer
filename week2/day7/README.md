# ReAct AI Agent — Interview Explanation and README

## 1. What is ReAct?

ReAct stands for Reasoning and Acting. It is a prompting approach that allows a Large Language Model (LLM) to solve a task by deciding what to do next, using tools, observing their results, and then deciding the next step.

Instead of answering a question immediately, the model follows a cycle:

Thought

Decide the next step

Action

Call a tool, such as a calculator

Observation

Receive the tool's actual result

Repeat or Finish

Choose another action or return the final answer

The main idea is simple: the LLM decides, external tools execute, and the LLM uses the results to continue.

## 2. A simple real-world example

Suppose a user asks:

> I have ₹5,000. What is the price of an iPhone 17, and how much money will I have left after buying it?

Your agent has two tools:

- `get_product_price(product)` — retrieves a product's price.
- `calculator(expression)` — calculates the remaining balance.

Here is how the ReAct cycle works.

Step 1 — Thought

The agent determines that it needs the iPhone 17 price before calculating the balance.

This is the model's proposed next step.

Step 2 — Action

```
get_product_price("iPhone 17")
```

Python executes the function and returns `1000`.

Step 3 — Observation

```
Observation: 1000
```

The agent receives the actual result from the tool.

Step 4 — Next Action

```
calculator("5000 - 1000")
```

Python returns `4000`.

Step 5 — Final Answer

The iPhone 17 costs ₹1,000 according to the mock data, leaving ₹4,000 from the original ₹5,000 budget.

These prices are hardcoded examples in your project, not real-world iPhone prices.

## 3. How ReAct works in your Python project

Your implementation has five important components.

| Component       | Responsibility                                                               |
| --------------- | ---------------------------------------------------------------------------- |
| LLM (Groq)      | Decides the next step and generates a response.                              |
| System prompt   | Instructs the model how to format thoughts, actions, and final answers.      |
| Tools           | Python functions that perform specific tasks.                                |
| Message history | Stores the conversation, actions, and observations for subsequent LLM calls. |
| Agent loop      | Repeats the process until the task is complete or the step limit is reached. |

The workflow in your code is:

```
response = client.chat.completions.create(    model=model,    messages=messages,    temperature=0)answer = response.choices[0].message.content
```

The model generates a response. Your Python code then parses an `Action`, executes the corresponding tool, appends the result to the conversation, and asks the model what to do next.

Important interview distinction: Your project implements ReAct using text-based prompting and regular expressions. It is a custom implementation, not native structured tool calling. The model proposes actions, but Python controls and executes them.

## 4. ReAct vs. a normal LLM

| Normal LLM                                            | ReAct agent                                                    |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| Typically generates an answer directly.               | Can alternate between reasoning, actions, and observations.    |
| May lack access to live or external data.             | Can use connected tools to retrieve or compute information.    |
| Does not automatically execute your Python functions. | Your agent code can execute functions requested by the model.  |
| Often suited to straightforward questions.            | Useful for tasks that require multiple steps and tool results. |

ReAct does not automatically make an answer correct. The agent can still select the wrong tool, misunderstand an observation, or stop too early. Tool validation and error handling remain important.

## 5. README file — ready to use

Copy the following Markdown into your project's `README.md`.

# ReAct AI Agent: Reasoning and Acting

## Overview

ReAct stands for **Reasoning and Acting**. It is a prompting approach that enables a Large Language Model (LLM) to solve tasks by deciding what to do, calling external tools, observing their results, and determining the next step.

Unlike a basic LLM interaction that typically generates a response directly, a ReAct agent can interact with external functions and use their results to complete multi-step tasks.

## How ReAct Works

A ReAct agent follows an iterative cycle:

1. **Thought:** The LLM determines the next step required to solve the task.
2. **Action:** The agent requests a tool call.
3. **Observation:** The Python application executes the tool and returns its result.
4. **Repeat:** The LLM uses the observation to determine the next action.
5. **Final Answer:** The agent returns the completed answer when sufficient information is available.

### Workflow

```
User Question
     |
     v
    Thought
     |
     v
    Action
     |
     v
 Execute Tool
     |
     v
  Observation
     |
     v
Update Message History
     |
     v
Next Thought / Action
     |
     v
 Final Answer
```

## Project Example

Consider the following user query:

"I have ₹5,000. What is the price of an iPhone 17, and how much money will I have left after buying it?"

The agent has two tools:

- `get_product_price(product)`: Returns the configured price of a product.
- `calculator(expression)`: Calculates a mathematical expression.

### Step 1: Thought

The agent determines that it must retrieve the product price first.

### Step 2: Action

```
get_product_price("iPhone 17")
```

### Step 3: Observation

```
1000
```

The Python function returns the configured product price.

### Step 4: Next Action

The agent requests a calculation:

```
calculator("5000 - 1000")
```

### Step 5: Observation

```
4000
```

### Step 6: Final Answer

The agent responds that the configured product price is ₹1,000 and the remaining balance is ₹4,000.

*Note: The prices in this example are hardcoded mock data for demonstrating the agent workflow. They are not actual market prices.*

## Technologies Used

- Python
- Groq API
- `openai/gpt-oss-120b` model
- `python-dotenv` for environment variable management
- Python regular expressions for parsing tool actions
- Custom Python functions for tool execution

## Project Architecture

### 1. LLM

The Groq API provides access to the language model, which generates the next action or final answer.

### 2. System Prompt

The system prompt defines the expected response format and instructs the model to use one action at a time.

### 3. Tools

Python functions perform operations requested by the model. The model does not execute these functions directly.

### 4. Message History

The application maintains the conversation history, including the model's responses and tool observations, so the model can continue the task with the available context.

### 5. Agent Loop

The loop repeatedly calls the LLM, identifies tool actions, executes them, and returns observations until the agent finishes or reaches the maximum number of steps.

## Key Benefits

- Supports multi-step problem-solving workflows.
- Enables interaction with external functions and data sources.
- Allows results from tools to inform subsequent actions.
- Separates language-model decision-making from deterministic Python execution.
- Can be extended with additional tools, such as database queries, APIs, and information retrieval.

## Limitations and Considerations

- The model may generate incorrectly formatted actions.
- Regular-expression parsing can fail when the output format changes.
- Tool outputs and errors must be handled carefully.
- A maximum step limit is necessary to prevent endless loops.
- Tool inputs should be validated before execution.
- Avoid using Python `eval()` on untrusted expressions; use a restricted arithmetic evaluator instead.
- A prompt does not guarantee that the model will always follow the required format.

## ReAct vs. Native Tool Calling

This project implements a custom, text-based ReAct loop. The LLM generates actions as text, and Python parses those actions and executes the corresponding functions.

Native tool calling instead provides structured tool-call information through the model API. Both approaches can enable tool use, but they differ in how actions are represented and processed.

## Conclusion

ReAct combines LLM-generated decisions with external tool execution and feedback. By following the Thought–Action–Observation cycle, an agent can solve tasks that require multiple operations before producing a final answer.

This project demonstrates the fundamentals of agent loops, prompt design, tool execution, message history, and stopping conditions.

## 6. Interview answer to memorize

If the interviewer asks, “What is ReAct, and how did you implement it?”, you can say:

30–45 SECOND ANSWER

ReAct stands for Reasoning and Acting. It is an approach where an LLM iteratively decides what to do, calls a tool, observes the result, and then decides the next step until it can provide a final answer.

In my Python project, I used Groq with the GPT-OSS 120B model and created two tools: one to retrieve a product's price and another to calculate the remaining balance. I maintain message history and use a loop to parse the model's requested action, execute the corresponding Python function, and return the observation to the model. The loop stops when a final answer is generated or the step limit is reached.



## 7. Likely follow-up interview questions

### 1. Why do we need ReAct instead of a normal LLM?
ReAct allows an LLM to use external tools and their results to solve multi-step tasks instead of relying entirely on generated text.
### 2. What is an observation in ReAct?
An observation is the result returned by a tool after an action. In my project, the product-price function returns 1000, which becomes an observation.
### 3. Does the LLM execute the tool itself?
No. The LLM generates a requested action. My Python application parses the action, executes the corresponding function, and sends the result back to the LLM.
### 4. Why do you maintain message history?
Message history preserves the original question, previous actions, and tool observations. This allows the LLM to use previous results when deciding what to do next.
### 5. What happens if the model never finishes?
The agent enforces a maximum number of steps. If the limit is reached without a final answer, it stops and returns a message indicating that the maximum number of steps was reached.
### 6. What is the main weakness of your implementation?
My implementation uses regular expressions to parse text-based tool calls, which can fail if the model changes the output format. Native structured tool calling and input validation would make it more reliable.

My recommendation: understand the difference between the LLM's decision, the Python tool's execution, and the observation. If you can explain those three parts clearly and walk through your iPhone example without reading from notes, you understand the core of ReAct well enough to discuss this project in an interview.