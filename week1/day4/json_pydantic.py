import os
import json
from pathlib import Path 
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KeY")

if not my_api_key:
    raise ValueError("API key not exist")

client=Groq(api_key=my_api_key)

model="openai/gpt-oss-120b"
role="user"

# structure the output
from pydantic import BaseModel
class Ticket(BaseModel):
    name: str
    email: str
    issue: str
    
schema = Ticket.model_json_schema()

response_format={
    "type": "json_object"
}
system_prompt = f"""
Extract the personal information from the ticket strictly based on this schema, and give me json output {schema}
"""
message_system={
    "role": "system",
    "content": system_prompt
}

text = "Hello My name is Rajendra. I have purchased an iphone which is not working at all. My address is kolkata. My email is test@test.com. My contact number is 814452"
prompt=f"""
This is a customer ticket. Please extract the personal information from this. 
{text}
"""

# prompt=input("Just tell me what you want to know?\n")
message={
    "role": role,
    "content": prompt
}
messages=[message_system, message]
response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)
print("############################################################")
answer = response.choices[0].message.content
print(answer)
# print("############################################################")
# print(json.dumps(response.model_dump(), indent=2))


import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)

print(ticket.name)
print(ticket.email)
print(ticket.issue)