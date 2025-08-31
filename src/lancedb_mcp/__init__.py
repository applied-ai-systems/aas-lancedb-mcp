"""AAS LanceDB MCP Server - Enhanced vector database with sentence transformers."""

__version__ = "0.1.0"

from .embedding import embedding_manager
from .models import (
    DatastoreInfo,
    EmbeddingConfig,
    SearchQuery,
    TableConfig,
    TextData,
    VectorData,
)
from .server import run, server, set_db_uri

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
