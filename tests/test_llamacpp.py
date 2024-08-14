# Script to test reproducibility of completions using different LangChain chat models
# Usage:
#   python test_llamacpp.py

# Path to your model weights
# Download source: https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF
# local_model = "/share/nikola/ag2435/Hermes-2-Pro-Llama-3-8B-GGUF/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf"

# https://huggingface.co/mradermacher/Hermes-2-Pro-Llama-3-70B-GGUF
local_model = "/share/nikola/ag2435/Hermes-2-Pro-Llama-3-70B-GGUF/Hermes-2-Pro-Llama-3-70B.Q4_K_M.gguf"

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

# 
# From: https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#high-level-api
# High-level API reference: https://llama-cpp-python.readthedocs.io/en/latest/api-reference/
# 
from time import time
import sys
from llama_cpp import Llama

llm = Llama(
      model_path=local_model,
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # default is LLAMA_DEFAULT_SEED
      # n_ctx=2048, # Uncomment to increase the context window
)

results = []
for i in range(1):
    start = time()
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.0,
    )
    elapsed = time() - start

    # print generated content to stdout
    result = output["choices"][0]["message"]["content"]
    print(f"trial #{i+1}: completion char length={len(result)}")
    print("START>>>>")
    print(result)
    print("<<<<END")
    # print metadata to stderr
    usage = output["usage"]
    print(usage, file=sys.stderr)
    print(f'elapsed time: {elapsed:.4f}s', file=sys.stderr)
    print(f"avg time per token: {elapsed/usage['total_tokens'] * 1000:.2f}ms", file=sys.stderr)
    results.append(result)
# print result lengths
print([len(result) for result in results])
# assert that all the results are the same
assert all([result == results[0] for result in results])
# Note that prefixes are cached, so to avoid caching functionality, only run the loop once