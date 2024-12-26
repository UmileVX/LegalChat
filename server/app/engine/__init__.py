from llama_index.core import PromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

# custom modules
from app.engine.index import get_index


def _use_hyde(query_engine, use_hyde: bool):
    if use_hyde:
        hyde = HyDEQueryTransform(include_original=True)
        query_engine = TransformQueryEngine(query_engine, hyde)
    return query_engine


def get_chat_engine():
    # chatmode: <https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern/#available-chat-modes>
    return get_index().as_chat_engine(
        similarity_top_k=3,
        chat_mode="condense_plus_context",
    )


def get_custom_chat_engine(last_msg: str, chat_history: list, verbose: bool = False, use_hyde: bool = False):
    index = get_index()
    query_engine = index.as_query_engine()
    query_engine = _use_hyde(query_engine, use_hyde=use_hyde)

    custom_prompt_str = f"""As a user, I want to know more about "{last_msg}".
    Answer in same language as the question. If you think the question is ambiguous, please ask for clarification.
    If you think the question is too violent/sexual/illegal, please let the user know that you can't answer it due to policy reasons.
    """
    custom_prompt = PromptTemplate(custom_prompt_str)

    # return CondensePlusContextChatEngine.from_defaults(
    #     retriever=self.as_retriever(**kwargs),
    #     llm=llm,
    #     **kwargs,
    # )

    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        condense_question_prompt=custom_prompt,
        chat_history=chat_history,
        verbose=verbose,
    )

    return chat_engine
