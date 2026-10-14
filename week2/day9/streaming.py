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
prompt=input("Just tell me what you want to know?\n")
message={
    "role": role,
    "content": prompt
}
messages=[message]

# without streaming
response=client.chat.completions.create(model=model, messages=messages)
print(response.choices[0].message.content)
print("############################################################")


# with streaming
stream=client.chat.completions.create(model=model, messages=messages, stream=True) # by default stream is false

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)