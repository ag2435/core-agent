# Model macros
GPT_MODELS = [
    "gpt-3.5-turbo", 
    "gpt-4-turbo", 
    "gpt-4o", 
    "gpt-4o-mini"
]
LLAMA_MODELS = [
    'llama-3-8b', 
    'llama-3-70b'
]
LLAMA_MODELS_LOCAL = [
    'llama-3-8b:local', 
    'llama-3-70b:local'
]
GEMINI_MODELS = [
    'gemini-1.5-flash', 
    'gemini-1.5-pro'
]
CLAUDE_MODELS = [
    'claude-3-5-sonnet-20240620', 
]
ALL_MODELS = GPT_MODELS + LLAMA_MODELS + LLAMA_MODELS_LOCAL + GEMINI_MODELS

# Helper function to get the LangChain chat model
def get_llm(model_name="gpt-3.5-turbo", temperature=1., **kwargs):
    """
    Get the LangChain chat model for the given model name.

    Args:
        model_name: The name of the model to use
        temperature: The temperature to use for sampling (see notes below)
        kwargs: Additional keyword arguments to pass to the model
    """
    if model_name in GPT_MODELS:
        # Note: this model is not deterministic even if you set seed and temperature to 0!
        from langchain_openai import ChatOpenAI
        # substitute alias for exact 3.5 model name to improve reproducibility
        model_name = 'gpt-3.5-turbo-0125' if model_name == "gpt-3.5-turbo" else model_name
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            # seed=123,
        )
    elif model_name in LLAMA_MODELS_LOCAL:
        # Use this model for deterministic completions
        # https://python.langchain.com/v0.2/docs/integrations/chat/llamacpp/
        from langchain_community.chat_models import ChatLlamaCpp
        if model_name == 'llama-3-8b:local':
            local_model = "/share/nikola/ag2435/Hermes-2-Pro-Llama-3-8B-GGUF/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf"
        elif model_name == 'llama-3-70b:local':
            local_model = "/share/nikola/ag2435/Hermes-2-Pro-Llama-3-70B-GGUF/Hermes-2-Pro-Llama-3-70B.Q4_K_M.gguf"
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        LLAMA_DEFAULT_SEED = 4294967295
        llm = ChatLlamaCpp(
            temperature=temperature,
            model_path=local_model,
            n_ctx=8192,
            n_gpu_layers=-1,
            max_tokens=512,
            verbose=False,
            seed=LLAMA_DEFAULT_SEED,
            streaming=False,
            **kwargs,
        )
    elif model_name in LLAMA_MODELS:
        raise NotImplementedError("todo: implement llama-3-8b and llama-3-70b via ChatTogether")
    
    elif model_name in GEMINI_MODELS:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            **kwargs,
        )
    
    elif model_name in CLAUDE_MODELS:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=temperature,
            # max_tokens=1024,
            timeout=None,
            max_retries=2,
            # other params...
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return llm

# parser = JsonOutputToolsParser(return_id=True)

# str_parser = StrOutputParser()