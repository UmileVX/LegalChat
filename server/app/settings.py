import os

from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def init_settings():
    model = os.getenv("MODEL", "gpt-4o")
    Settings.llm = OpenAI(model=model)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        embed_batch_size=100,
        dimensions=256,
    )
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20
