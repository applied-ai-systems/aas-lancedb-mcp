"""AAS LanceDB MCP Server - Enhanced vector database with sentence transformers."""

__version__ = "0.1.0"

from .server import server, run, set_db_uri
from .models import (
    VectorData,
    TextData,
    SearchQuery,
    TableConfig,
    DatastoreInfo,
    EmbeddingConfig,
)
from .embedding import embedding_manager

__all__ = [
    "server",
    "run",
    "set_db_uri",
    "VectorData",
    "TextData",
    "SearchQuery",
    "TableConfig",
    "DatastoreInfo",
    "EmbeddingConfig",
    "embedding_manager",
]
