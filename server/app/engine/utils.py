import os
from urllib.parse import urlparse
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text

# custom modules
from app.engine.constants import (
    PGVECTOR_SCHEMA,
    PGVECTOR_TABLE,
    PGVECTOR_EMBED_DIM,
)
from app.utils.singleton import Singleton


def init_pg_vector_store_from_env(init_table: bool = False):
    # use singleton to ensure only one instance of the vector store is created
    vectorstore = VectorStoreContainer(init_table=init_table)
    return vectorstore.get_store()


class VectorStoreContainer(metaclass=Singleton):
    """
    A singleton class to hold the vector store instance.

    The vector store instance creates a connection pool to the database.
    So, if we create multiple instances of the vector store, we will end up creating multiple connection pools.
    This class ensures that only one instance of the vector store is created and shared across the application.

    As postgres creates a new process for each connection, we should avoid creating multiple connection pools.
    """  # noqa: E501
    def __init__(self, init_table: bool = False):
        if init_table:
            self._create_table()

        self._build_vector_store()


    def _build_vector_store(self):
        original_conn_string = os.environ.get("PG_CONNECTION_STRING")
        if original_conn_string is None or original_conn_string == "":
            raise ValueError("PG_CONNECTION_STRING environment variable is not set.")

        # The PGVectorStore requires both two connection strings, one for psycopg2 and one for asyncpg
        # Update the configured scheme with the psycopg2 and asyncpg schemes
        original_scheme = urlparse(original_conn_string).scheme + "://"
        conn_string = original_conn_string.replace(
            original_scheme, "postgresql+psycopg2://"
        )
        async_conn_string = original_conn_string.replace(
            original_scheme, "postgresql+asyncpg://"
        )

        # create the vector store
        self.store = PGVectorStore(
            connection_string=conn_string,
            async_connection_string=async_conn_string,
            schema_name=PGVECTOR_SCHEMA,
            table_name=PGVECTOR_TABLE,
            embed_dim=PGVECTOR_EMBED_DIM,
            cache_ok=True,
            # hybrid_search=True,
            # text_search_config="english",
        )


    def _create_table(self):
        original_conn_string = os.environ.get("PG_CONNECTION_STRING")
        original_scheme = urlparse(original_conn_string).scheme + "://"
        conn_string = original_conn_string.replace(
            original_scheme, "postgresql+psycopg2://"
        )
        engine = create_engine(conn_string)
        conn = engine.connect()

        try:
            # must be a super user to create extension
            conn.execute(text("""CREATE EXTENSION IF NOT EXISTS vector;"""))

            # Create table if not exists
            #
            # The table has the following columns:
            # - text: the text to be indexed
            # - metadata_: metadata for the document
            # - node_id: the unique identifier for the document (UUID)
            # - vector: the vector representation of the document (vector)
            conn.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS {PGVECTOR_SCHEMA}.{PGVECTOR_TABLE} (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    metadata_ JSONB,
                    node_id UUID,
                    embedding VECTOR({PGVECTOR_EMBED_DIM}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
            )

            # add HNSW index on the embedding column
            conn.execute(
                text(f"""
                CREATE INDEX IF NOT EXISTS idx_{PGVECTOR_TABLE}_embedding ON {PGVECTOR_SCHEMA}.{PGVECTOR_TABLE} USING HNSW (embedding vector_l2_ops);
                """)
            )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()


    def get_store(self):
        return self.store

    def refresh(self):
        self._build_vector_store()
        return self.store
