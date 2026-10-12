import os
from dotenv import load_dotenv
from groq import Groq
import re
from time import sleep

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("api key not found")

client = Groq(api_key = my_api_key)
model = "openai/gpt-oss-120b"


# tools
def get_product_price(product):
    if product == 'iPhone 17':
        return 1000
    elif product == 'iPhone 15':
        return 500
    else:
        return 0
    
def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calc error"
    
    
tools = {
    "get_product_price" : get_product_price,
    "calculator" : calculator
}

system_prompt="""
You are a shopping assistant. you have these 2 tools:
get_product_price(product)
calculator(expression)

Call tools exactly like these examples:

Action: get_product_price("iPhone 17")
Action: calculator("5000 - 1000")

Never Write:
get_product_price(product = "iPhone 17")
Action: calculator(expression = "5000 - 1000")


Follow these rules:

1. Decide what to do next.
2. Call only one tool at a time. 
3. After writing an action, stop immediately. 
4. Never guess or invent a tool result. 
5. Wait until you receive an observation. 
6. Then decide your next action 
7. When the task is complete, give the final answer.

Every tool call must contain a brief Thought before Action.
Return exactly one Action per response.
Never combine Action and Final Answer in the same response.

Output Format:
Thought: explain what you need to do
Action: tool_name(argument)

when finished:
Final Answer: your answer
"""

def run_agent(question):
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]
    for step in range(5):
        print("\n..........................")
        print("STEP", step+1)
        print("............................")
            
        response = client.chat.completions.create(
            model = "openai/gpt-oss-120b",
            messages = messages,
            temperature = 0
        )
        answer = response.choices[0].message.content
        answer = answer.strip()
        print(answer)
            
        # Agent has finished
        if "final answer: " in answer.lower():
            break
                
        # find the action
        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)", answer
        )
            
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')
                
            # run the tool
            if tool_name in tools:
                tool = tools[tool_name]
                    
                observation = tool(tool_input)
            else:
                observation = "Tool not found"
                    
            print("observation", observation)
                
            # Add LLM response to memory
            messages.append({
                "role": "assistant",
                "content": answer
            })
                
            # give tool result back to LLM
            messages.append({
                "role": "assistant",
                "content": "observation: " + str(observation)
            })
            sleep(5)
    
    
prompt = """
I have 5000 rupees. What is the price of iPhone 17?
And how much money will i have left after buying it?
"""
run_agent(prompt)