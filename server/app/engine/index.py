from llama_index.core.indices.vector_store import VectorStoreIndex

# custom module
from app.engine.utils import init_pg_vector_store_from_env
from app.utils.logging import Logger


logger = Logger()


def get_index():
    logger.log_info("Connecting to index from PGVector...")
    store = init_pg_vector_store_from_env()
    index = VectorStoreIndex.from_vector_store(store)
    logger.log_info("Finished connecting to index from PGVector.")
    return index
