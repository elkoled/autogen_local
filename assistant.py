"""Example of how to add MemGPT into an AutoGen groupchat

Based on the official AutoGen example here: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb

Begin by doing:
  pip install "pyautogen[teachable]"
  pip install pymemgpt
  or
  pip install -e . (inside the MemGPT home directory)
"""


import os
import autogen
from memgpt.autogen.memgpt_agent import create_autogen_memgpt_agent, create_memgpt_autogen_agent_from_config
from memgpt.presets.presets import DEFAULT_PRESET
from memgpt.constants import LLM_MAX_TOKENS


# Example using llama.cpp on a local machine
# You will have to change the parameters based on your setup

# for openai api >v1 use following config
# config_list = [
#     {
#         "model": "NULL",  # not needed
#         # ex. "http://127.0.0.1:5001/v1" if you are using webui, "http://localhost:1234/v1/" if you are using LM Studio
#         "base_url": "http://192.168.188.39:8080/v1",
#         "api_key": "NULL",  # not needed
#         # "api_type": "open_ai",
#     },
# ]
config_list = [
    {
        "model": "NULL",  # not needed
        # ex. "http://127.0.0.1:5001/v1" if you are using webui, "http://localhost:1234/v1/" if you are using LM Studio
        "api_base": "http://0.0.0.0:8080/v1",
        "api_key": "NULL",  # not needed
        "api_type": "open_ai",
    },
]

# MemGPT-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
config_list_memgpt = [
    {
        "preset": DEFAULT_PRESET,
        "model": None,  # only required for Ollama, see: https://memgpt.readthedocs.io/en/latest/ollama/
        # airoboros is the default wrapper and should work for most models
        "model_wrapper": "airoboros-l2-70b-2.1-grammar",
        # can use webui, ollama, llamacpp, etc.
        "model_endpoint_type": "llamacpp",
        # the IP address of your LLM backend
        "model_endpoint": "http://0.0.0.0:8080",
        # the context window of your model (for Mistral 7B-based models, it's likely 8192)
        "context_window": 8000,
    },
]

# If USE_MEMGPT is False, then this example will be the same as the official AutoGen repo
# (https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb)
# If USE_MEMGPT is True, then we swap out the "coder" agent with a MemGPT agent
USE_MEMGPT = True

# Set to True if you want to print MemGPT's inner workings.
DEBUG = True

interface_kwargs = {
    "debug": DEBUG,
    "show_inner_thoughts": DEBUG,
    "show_function_outputs": DEBUG,
}

llm_config = {"config_list": config_list, "seed": 42}
llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}


# create an AssistantAgent named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "cache_seed": 42,  # seed for caching and reproducibility
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0,  # temperature for sampling
    },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # set to True or image name like "python:3" to use docker
    },
)
# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""create plots as .png files that make sense for this csv file: https://github.com/OSkrk/Electric-vehicles-EV-Database/blob/main/Data/EVs_data_base.csv""",
)