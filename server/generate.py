import os

from dotenv import load_dotenv

load_dotenv()

import logging

from app.engine.loader import get_documents
from app.engine.utils import init_pg_vector_store_from_env
from app.settings import init_settings
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.storage import StorageContext


# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_datasource():
    logger.info("Creating new index")
    # load the documents and create the index
    documents = get_documents()
    store = init_pg_vector_store_from_env()
    storage_context = StorageContext.from_defaults(vector_store=store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,  # this will show you a progress bar as the embeddings are created  # noqa: E501
    )
    logger.info(f"Successfully created embeddings in the PG vector store, schema={store.schema_name} table={store.table_name}")  # noqa: E501


if __name__ == "__main__":
    init_settings()

    generate_datasource()
