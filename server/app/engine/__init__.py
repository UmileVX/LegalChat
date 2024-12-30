import os
from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import CondensePlusContextChatEngine

# custom modules
from app.engine.index import get_index
from app.settings import Settings

from .prompts.base import BASE_SYS_PROMPT
from .prompts.herlab import HERLAB_PROMPT


FOR_SIHM_SVC = os.getenv("FOR_SIHM_SVC", "false").lower() == "true"


def get_chat_engine():
    # chatmode: <https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern/#available-chat-modes>
    return get_index().as_chat_engine(
        similarity_top_k=3,
        chat_mode="condense_plus_context",
    )


def get_custom_chat_engine(last_msg: str, chat_history: list, verbose: bool = False):
    index = get_index()
    retriever = index.as_retriever()
    llm = Settings.llm

    if FOR_SIHM_SVC:
        custom_prompt_str = HERLAB_PROMPT
    else:
        custom_prompt_str = f"""As a user, I want to know more about "{last_msg}".\n{BASE_SYS_PROMPT}"""

    custom_prompt = PromptTemplate(custom_prompt_str)

    # context_prompt = PromptTemplate(DEFAULT_CONTEXT_PROMPT_TEMPLATE)
    # refine_prompt = PromptTemplate(DEFAULT_CONTEXT_REFINE_PROMPT_TEMPLATE)
    # condense_prompt = PromptTemplate(DEFAULT_CONDENSE_PROMPT_TEMPLATE)

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever,
        llm=llm,
        context_prompt=custom_prompt,
        chat_history=chat_history,
        verbose=verbose,
    )
    return chat_engine
