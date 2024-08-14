# Script to test reproducibility of completions using different LangChain chat models
# Usage:
#   python test_langchain_chat.py gpt-3.5-turbo
#   python test_langchain_chat.py llama-3-8b

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("model", type=str, help="model name")
args = parser.parse_args()

# remaining imports
from time import time
import sys
from core_agent.base import get_llm

# get the LangChain chat model
model_name = args.model
llm = get_llm(
    model_name=model_name,
    temperature=0.0,
)

messages = [
    {
        'role': "system",
        'content': "You are a helpful assistant.",
    },
    {
        'role': "human", 
        'content': "Write me a poem.",
    },
]
results = []
for i in range(1):
    start = time()
    ai_msg = llm.invoke(messages)
    elapsed = time() - start
    # get generated content
    result = ai_msg.content
    # print result to stdout
    print(f"trial #{i+1}: completion char length={len(result)}")
    print("START>>>>")
    print(result)
    print("<<<<END")
    # print metadata to stderr
    usage = ai_msg.response_metadata['token_usage']
    print(usage, file=sys.stderr)
    print(f"elapsed time: {elapsed:.4f}s", file=sys.stderr)
    print(f"avg time per token: {elapsed / usage['total_tokens'] * 1000:.2f}ms", file=sys.stderr)
    results.append(result)
# print result lengths
print([len(result) for result in results])
# assert that all the results are the same
assert all([result == results[0] for result in results])