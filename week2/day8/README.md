
# LLM-Based Resume–Job Description Matching System

## 1. Project Overview

This project uses a Large Language Model (LLM) through the Groq API to analyze a candidate's resume and compare it with a job description.

The system extracts skills from the resume, identifies the skills required by the job description, and generates a candidate-fit assessment with a score and a short verdict.

**Technologies Used**

- Python
- Groq API
- Large Language Models (LLMs)
- `groq` Python SDK
- `python-dotenv`
- Environment variables for API key management

**Model:** `openai/gpt-oss-120b`

> Model availability and API support can change. Use a model currently available to your Groq account.

---

## 2. Project Workflow

The project follows a sequential LLM pipeline with three main steps.

```text
Resume + Job Description
          |
          v
Step 1: Resume Skill Extraction
          |
          v
Step 2: Job Description Skill Extraction
          |
          v
Step 3: Candidate–Job Matching
          |
          v
Matching Score + Verdict
```

### Step 1: Resume Skill Extraction

The first function extracts the skills explicitly mentioned in the candidate's resume.

Example input:

```text
Skills:
Python, FastAPI, MySQL, Docker,
REST APIs, Git
```

Example output:

```text
Python, FastAPI, MySQL, Docker,
REST APIs, Git
```

Function:

`step1_res_extract()`

### Step 2: Job Description Skill Extraction

The second function extracts the technical requirements from the job description.

Example input:

```text
Requirements:
Python, FastAPI or Django,
PostgreSQL, Docker, AWS, REST APIs
```

Example output:

```text
Python, FastAPI, Django, PostgreSQL,
Docker, AWS, REST APIs
```

Function:

`step2_JD_extract()`

The model should extract only requirements supported by the job description. It should not invent additional requirements.

### Step 3: Candidate–Job Matching

The third function compares the extracted candidate skills with the job requirements.

Function:

`step3_match(candidate, jd)`

It evaluates:

- Matching skills
- Missing skills
- Relevant experience
- Evidence of project experience
- Overall suitability
- A short explanation of the result

The model returns a score between 1 and 100, along with a verdict.

**Important:** The score is an LLM-generated assessment, not a statistically validated probability of getting hired.

---

## 3. Understanding the `ask_llm()` Function

The `ask_llm()` function is the core reusable component of the project.

It communicates with the Groq API and returns the generated text.

```python
def ask_llm(system_prompt, user_prompt):

    sys_msg = {
        "role": "system",
        "content": system_prompt
    }

    user_msg = {
        "role": "user",
        "content": user_prompt
    }

    messages = [sys_msg, user_msg]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    answer = response.choices[0].message.content

    return answer
```

### How it works

**1. Accepts two arguments**

- `system_prompt`: Instructions that define how the model should behave.
- `user_prompt`: The actual task and input data.

**2. Creates the system message**

```python
sys_msg = {
    "role": "system",
    "content": system_prompt
}
```

The system message provides instructions to the model.

**3. Creates the user message**

```python
user_msg = {
    "role": "user",
    "content": user_prompt
}
```

The user message contains the task or information to analyze.

**4. Combines the messages**

```python
messages = [sys_msg, user_msg]
```

The API receives a list of message dictionaries, each containing a role and content.

**5. Sends the API request**

```python
response = client.chat.completions.create(
    model=model,
    messages=messages
)
```

This requests a chat completion from the selected Groq-hosted model.

**6. Extracts the generated text**

```python
answer = response.choices[0].message.content
```

- `choices` contains the completion choices returned by the API.
- `[0]` selects the first choice using zero-based indexing.
- `message` accesses the generated message.
- `content` retrieves its text.

**7. Returns the result**

```python
return answer
```

The returned text can be used by another function in the pipeline.

### Why use a reusable function?

Without `ask_llm()`, the API request logic would need to be repeated in all three pipeline functions.

By creating one reusable function, the code becomes easier to maintain, test, and extend.

---

## 4. System Prompt vs. User Prompt

| System prompt                             | User prompt                          |
| ----------------------------------------- | ------------------------------------ |
| Defines the model's instructions and role | Provides the specific task and input |
| Establishes expected behavior             | Supplies the information to process  |
| Can be reused across similar tasks        | Usually changes between tasks        |

Example:

```python
system_prompt = """
You are a professional HR assistant.
Extract skills explicitly mentioned in the resume.
Do not invent any skills.
"""

user_prompt = f"""
Extract skills from this resume:

{RESUME}
"""
```

The `f` prefix creates an **f-string**, allowing Python to insert the value of `RESUME` into the prompt.

Without the `f` prefix, `{RESUME}` would remain literal text rather than being replaced with the resume content.

---

## 5. Understanding the Groq Client

### Loading environment variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
```

- `os`: Provides access to operating-system functionality, including environment variables.
- `load_dotenv()`: Loads variables from a `.env` file into the environment.
- `os.getenv()`: Retrieves the value of an environment variable.

### Initializing the client

```python
from groq import Groq

client = Groq(api_key=my_api_key)
```

The client uses the API key to authenticate requests to Groq.

### API key validation

```python
if not my_api_key:
    raise ValueError("GROQ_API_KEY is missing")
```

This prevents the program from continuing when the required API key is absent.

### Example `.env` file

```env
GROQ_API_KEY=your_actual_api_key
```

Never commit `.env` to GitHub. Add it to `.gitignore`.

---

## 6. Python Concepts Used in This Project

| Concept                  | How it is used                                               |
| ------------------------ | ------------------------------------------------------------ |
| Functions                | Divide the workflow into independent tasks                   |
| Parameters and arguments | Pass prompts, extracted skills, and job requirements         |
| Dictionaries             | Represent API messages                                       |
| Lists                    | Store the sequence of messages                               |
| Variables                | Store the API key, model, responses, and extracted results   |
| f-strings                | Insert resume and job-description text into prompts          |
| Return statements        | Pass generated results back to the calling function          |
| External libraries       | Communicate with the Groq API and load environment variables |
| Exception handling       | Handle API failures and unexpected responses                 |
| Sequential execution     | Run extraction and matching in the correct order             |

### Parameters vs. arguments

```python
def step3_match(candidate, jd):
    ...
```

Here, `candidate` and `jd` are parameters.

```python
score = step3_match(candidate, jd)
```

Here, the values passed to the function are arguments.

Calling `step3_match()` without arguments causes a `TypeError` because both parameters are required.

### Why use `return` instead of `print`?

`print()` displays information in the terminal.

`return` passes a value back to the caller so the rest of the program can use it.

For example:

```python
candidate = step1_res_extract()
jd = step2_JD_extract()

score = step3_match(candidate, jd)

print(score)
```

The first two results become inputs to the third function.

---

## 7. Understanding `response.choices[0]`

Consider:

```python
answer = response.choices[0].message.content
```

Python uses zero-based indexing.

- `choices[0]`: First choice
- `choices[1]`: Second choice
- `choices[2]`: Third choice

The Groq API generally returns one choice by default, so `choices[0]` is appropriate.

Using `choices[1]` when only one choice exists raises:

```text
IndexError: list index out of range
```

Multiple choices can be requested with `n` when supported by the model and API configuration, but they are unnecessary for this project's normal workflow.

---

## 8. Why Use an LLM?

Traditional rule-based matching might check whether a keyword appears in both documents.

For example:

```python
if "Python" in resume and "Python" in job_description:
    print("Python skill matched")
```

This approach is simple, but it cannot reliably interpret context, equivalent terminology, experience descriptions, or project evidence.

An LLM can interpret natural language and produce a structured explanation of how a candidate's experience relates to a role.

However, LLM output can be inconsistent or incorrect. For reliable hiring support, the system should validate extracted data, use transparent scoring criteria, and avoid treating the generated score as an objective truth.

---

## 9. Limitations and Future Improvements

### Current limitations

1. The model can omit skills or extract them incorrectly.
2. A generated suitability score may vary between runs.
3. A resume may contain unsupported or exaggerated claims.
4. API requests can fail because of rate limits, network problems, or model availability.
5. The pipeline does not guarantee a fair or accurate hiring decision.
6. Text embedded in resumes could contain prompt-injection instructions.

### Future improvements

- Return structured JSON instead of free-form text.
- Validate the response using Pydantic.
- Define a transparent scoring rubric for skills, experience, and project evidence.
- Add retry logic for temporary API failures.
- Add exception handling and logging.
- Support PDF and DOCX resume uploads.
- Add unit tests for individual pipeline functions.
- Evaluate results against a labeled test dataset.
- Treat resume content as untrusted data and instruct the model not to follow embedded instructions.
- Add a user interface using FastAPI or Streamlit.

For hiring applications, the model should support human review rather than make autonomous employment decisions.

---

# Interview Questions and Answers

## Q1. What does your project do?

**Answer:** My project uses a Large Language Model through the Groq API to extract skills from a candidate's resume, extract required skills from a job description, and generate a candidate-fit assessment with matching and missing skills.

## Q2. What is Groq?

**Answer:** Groq provides an API for running supported language models. I use its Python SDK to send chat completion requests and receive model-generated responses.

## Q3. What is an LLM?

**Answer:** A Large Language Model is a machine-learning model trained on large amounts of text to understand and generate language. In my project, it interprets resume and job-description text.

## Q4. What is the purpose of `ask_llm()`?

**Answer:** It is a reusable wrapper around the Groq Chat Completions API. It constructs the system and user messages, sends the request, extracts the response text, and returns it.

## Q5. Why do you use system and user roles?

**Answer:** The system role provides instructions for the model's behavior, while the user role provides the actual task and input data.

## Q6. What does `client.chat.completions.create()` do?

**Answer:** It sends a chat completion request to the selected model with the supplied messages and returns a response object.

## Q7. What is the purpose of `response.choices[0].message.content`?

**Answer:** It accesses the first completion choice and retrieves the generated message's text.

## Q8. Why does Python use `[0]` instead of `[1]` for the first element?

**Answer:** Python uses zero-based indexing, so the first element has index zero. Index one refers to the second element.

## Q9. Why did you create separate functions for resume extraction, JD extraction, and matching?

**Answer:** Each function has a distinct responsibility. This separation makes the workflow easier to understand, test, debug, and maintain.

## Q10. Why do you use f-strings?

**Answer:** F-strings allow me to insert variables such as the resume and job description directly into prompt strings.

## Q11. Why use `return` instead of `print()` inside the functions?

**Answer:** Returning values allows later stages of the pipeline to process the extracted results. Printing only displays information and does not pass it to the next stage.

## Q12. What is prompt engineering?

**Answer:** Prompt engineering is the process of designing instructions and input formats to guide an LLM toward useful, relevant, and consistent outputs.

## Q13. Why do you use `load_dotenv()`?

**Answer:** It loads environment variables from a `.env` file so the API key can be configured outside the source code.

## Q14. Why should an API key not be hardcoded?

**Answer:** Hardcoding secrets risks exposing them through source control or shared code. Environment variables help separate configuration from application logic.

## Q15. What is the purpose of `sleep(2)`?

**Answer:** It pauses execution for two seconds between calls. It can space out requests, but it does not guarantee compliance with API rate limits or fix API failures.

## Q16. Is the matching score objective?

**Answer:** No. The current score is generated by an LLM based on its interpretation of the provided information. To make it more reliable, I would define a scoring rubric, validate outputs, and evaluate the system on labeled examples.

## Q17. What happens if the API request fails?

**Answer:** The current basic implementation may raise an exception. I would improve it with appropriate error handling, limited retries for transient failures, logging, and response validation.

## Q18. Is this an AI agent?

**Answer:** Not in its current form. It is a sequential LLM pipeline. It calls an LLM for predefined tasks but does not autonomously select tools or decide which action to perform next.

## Q19. What is the difference between an LLM pipeline and a ReAct agent?

**Answer:** A pipeline follows a predefined sequence of operations. A ReAct agent alternates between reasoning, choosing an action, observing the result, and deciding what to do next, often using tools.

## Q20. How would you improve this project for production?

**Answer:** I would add structured outputs, schema validation, error handling, logging, tests, a defined scoring rubric, privacy controls, prompt-injection defenses, and evaluation against a labeled dataset. I would also keep a human involved in employment decisions.

---

# 10. One-Minute Interview Revision

Remember this sequence:

**Input → Extract → Compare → Return**

1. Load the Groq API key from environment variables.
2. Initialize the Groq client.
3. Extract candidate skills using an LLM.
4. Extract job requirements using an LLM.
5. Compare the candidate's information with the job requirements.
6. Return the generated assessment.

The core reusable function is `ask_llm()`.

The key concepts are:

- LLM API integration
- System and user prompts
- Python dictionaries and lists
- Function parameters and return values
- f-strings and environment variables
- Sequential pipelines
- Error handling and output validation

**Final interview takeaway:** Explain not only what the code does, but also why you separated the functions, how data moves between stages, and what you would change to make the results more reliable.
