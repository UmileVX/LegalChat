import os

PGVECTOR_SCHEMA = os.getenv("PGVECTOR_SCHEMA", "public")
PGVECTOR_TABLE = "document_embeddings"
PGVECTOR_EMBED_DIM = int(os.getenv("PGVECTOR_EMBED_DIM", 256))
