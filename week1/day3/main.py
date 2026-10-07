import os
import json
from pathlib import Path 
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key not exist")

client=Groq(api_key=my_api_key)

model="openai/gpt-oss-120b"
role="user"

# 3 different prompts
prompt1 = "Hi!"
prompt2 = "Explain time travel in Detail"
prompt3 = "Write a 1000 word essay on Machine Learning"

prompts = [prompt1, prompt2, prompt3]
for prompt in prompts:
    message={
        "role": role,
        "content": prompt
    }
    messages=[message]
    response=client.chat.completions.create(model=model, messages=messages, max_tokens=50)
    
    usage = response.usage
    print(f"Prompt: {prompt} ---> your token: {usage.prompt_tokens}, completion token {usage.completion_tokens}, total token: {usage.prompt_tokens + usage.completion_tokens}, Finish reason: {response.choices[0].finish_reason}")
    # print("############################################################")
    # print(response.choices[0].message.content)






# print("############################################################")
# print(json.dumps(response.model_dump(), indent=2))