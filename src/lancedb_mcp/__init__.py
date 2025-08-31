"""AAS LanceDB MCP Server - Enhanced vector database with sentence transformers."""

__version__ = "0.1.0"

from .embedding import embedding_manager
from .models import (
    ColumnSchema,
    EmbeddingConfig,
    InsertData,
    MigrationPlan,
    QueryData,
    SearchQuery,
    TableInfo,
    TableSchema,
    UpdateData,
)
from .server import run, server, set_db_uri

__all__ = [
    "server",
    "run",
    "set_db_uri",
    "ColumnSchema",
    "TableSchema",
    "InsertData",
    "UpdateData",
    "QueryData",
    "SearchQuery",
    "TableInfo",
    "MigrationPlan",
    "EmbeddingConfig",
    "embedding_manager",
]
