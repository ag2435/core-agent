# Script to test tool calling functionality according to this tutorial:
# https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/#tool-calling
# Usage:
#   python test_langchain_tool_calling.py llama-3-8b
#   python test_langchain_tool_calling.py gpt-4o-mini
# Output:
# [{'name': 'get_current_weather', 'args': {'location': 'Ho Chi Minh City', 'unit': 'celsius'}, 'id': 'call__0_get_current_weather_cmpl-a7b670ed-3024-4946-bf06-4831b6a52662', 'type': 'tool_call'}]

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("model", type=str, help="model name")
args = parser.parse_args()
model_name = args.model

# remaining imports
from core_agent.base import get_llm
from langchain_core.pydantic_v1 import BaseModel, Field

# get the LangChain chat model
llm = get_llm(
    model_name=model_name,
    temperature=0.0,
)

# define output schemas
class Location(BaseModel):
    city: str = Field(description="The city")
    state: str = Field(description="The state")

class WeatherInput(BaseModel):
    location: Location = Field(description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(enum=["celsius", "fahrenheit"])

# bind tools
llm_with_tools = llm.bind_tools(
    tools=[WeatherInput],
    tool_choice={"type": "function", "function": {"name": "WeatherInput"}},
)

messages = [
    {
        'role': "system",
        'content': "You are a helpful assistant.",
    },
    {
        'role': "human", 
        'content': "What is the weather like in HCMC in celsius.",
    },
]
ai_msg = llm_with_tools.invoke(
    messages
)
# print((ai_msg))
print(ai_msg.tool_calls)
# save ai_msg to a file with pretty json
# import json
# with open("ai_msg.json", "w") as f:
#     json.dump(ai_msg.dict(), f, indent=2)