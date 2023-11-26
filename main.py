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

# Non-MemGPT agents will still use local LLMs, but they will use the ChatCompletions endpoint
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
        "api_base": "http://192.168.188.39:8080/v1",
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
        "model_wrapper": "airoboros-l2-70b-2.1",
        # can use webui, ollama, llamacpp, etc.
        "model_endpoint_type": "llamacpp",
        # the IP address of your LLM backend
        "model_endpoint": "http://192.168.188.39:8080",
        # the context window of your model (for Mistral 7B-based models, it's likely 8192)
        "context_window": 8000,
    },
]

# If USE_MEMGPT is False, then this example will be the same as the official AutoGen repo
# (https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb)
# If USE_MEMGPT is True, then we swap out the "coder" agent with a MemGPT agent
USE_MEMGPT = True

# Set to True if you want to print MemGPT's inner workings.
DEBUG = False

interface_kwargs = {
    "debug": DEBUG,
    "show_inner_thoughts": DEBUG,
    "show_function_outputs": DEBUG,
}

llm_config = {"config_list": config_list, "seed": 42}
llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}

# The user agent
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE",  # needed?
    # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
    default_auto_reply="...",
)

# The agent playing the role of the product manager (PM)
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
    # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
    default_auto_reply="...",
)

if not USE_MEMGPT:
    # In the AutoGen example, we create an AssistantAgent to play the role of the coder
    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config,
    )

else:
    # In our example, we swap this AutoGen agent with a MemGPT agent
    # This MemGPT agent will have all the benefits of MemGPT, ie persistent memory, etc.
    coder = create_memgpt_autogen_agent_from_config(
        "MemGPT_coder",
        llm_config=llm_config_memgpt,
        system_message=f"I am a 10x engineer, trained in Python. I was the first engineer at Uber "
        f"(which I make sure to tell everyone I work with).\n"
        f"You are participating in a group chat with a user ({user_proxy.name}) "
        f"and a product manager ({pm.name}).",
        interface_kwargs=interface_kwargs,
        # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
        default_auto_reply="...",
    )

# Initialize the group chat between the user and two LLM agents (PM and coder)
groupchat = autogen.GroupChat(
    agents=[user_proxy, pm, coder], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Begin the group chat with a message from the user
user_proxy.initiate_chat(
    manager,
    message="print random data on a matplotlib graph and write to png file.",
)
