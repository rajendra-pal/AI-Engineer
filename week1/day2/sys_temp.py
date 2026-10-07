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
message_system={
    "role": "system",
    "content": "You are a brand manager who suggest name for clothing brand. Name should be in one word. Suggest only one name."
}

# message me role and the system
message={
    "role": role,
    "content": prompt
}
messages=[message_system, message]
response=client.chat.completions.create(model=model, messages=messages, temperature=0) # temperature by default is 0 and it's range is between 0 to 2 
print("############################################################")
print(response.choices[0].message.content)

# print("############################################################")
# # print(json.dumps(response.model_dump(), indent=2))