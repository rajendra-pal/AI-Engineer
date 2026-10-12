import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv('GROQ_API_KEY')

if not my_api_key:
    raise ValueError("API key not found")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"


def llm_ans(prompt):
    message={
        "role": "user",
        "content": prompt
    }
    
    messages = [message]
    response = client.chat.completions.create(model = model, messages = messages)
    ans = response.choices[0].message.content
    return ans

# to get the better result from the llm you must mention the role, task, constraints, output format, example and the fallback


bad_prompt = """
#ROLE: You are a support assistant of a mobile/laptop company
#TASK: You have to classify the issue in a category
#CONSTRAINT: You have to classify the issue in one of the three categories namely Billing, Technical, Return
#OUTPUT FORMAT: Your answer should be in one word only. The one word should be one of the categories given in the constraints.
#EXAMPLE: For instance if an user complaints say he wants refund then the category is return
#FALLBACK: If the issue is unrelated to any of the category then mention the answer should be OTHER
this is an user complaint:
My laptop is not working
"""
print((llm_ans(bad_prompt)))