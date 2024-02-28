import os

from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI


def init_settings():
    model = os.getenv("MODEL", "gpt-4")
    Settings.llm = OpenAI(model=model)
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20
