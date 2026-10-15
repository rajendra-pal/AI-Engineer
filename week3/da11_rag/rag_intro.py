import os
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api key not found")

client = Groq(api_key = my_api_key)
model = "openai/gpt-oss-120b"


# step 1
knowledge_base = {
    "age": "the age of pratyush is 25 years",
    "net_worth": "the net worth of pratyush is 2000" 
}

# step 2
def retrieve_info(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["net_worth"]
    else:
        return None
    
    
def ask_llm(question):
    context = retrieve_info(question)
    system_msg = {
        "role": "system",
        "content": f"answer in one line only and answer only based on this context. Do not hallucinate. Context: {context}"
    }
    messsage={
        "role": "user",
        "content": question
    }
    messages = [system_msg, messsage]
    
    response = client.chat.completions.create(
        model = model, messages = messages
    )
    answer = response.choices[0].message.content
    return answer



question = "what is age?"
print(ask_llm(question))