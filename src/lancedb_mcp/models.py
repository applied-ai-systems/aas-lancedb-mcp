"""Models for AAS LanceDB MCP with sentence transformers integration."""

from typing import Any

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model name"
    )
    dimension: int = Field(default=384, description="Vector dimension")
    device: str = Field(default="cpu", description="Device to run model on (cpu/cuda)")
    normalize_embeddings: bool = Field(
        default=True, description="Whether to normalize embeddings"
    )


class TableConfig(BaseModel):
    """Configuration for creating a table."""

    name: str = Field(..., min_length=1, description="Name of the table")
    dimension: int = Field(
        default=384,
        gt=0,
        description="Vector dimension (sentence-transformers default)",
    )
    metric: str = Field(default="cosine", description="Distance metric")
    description: str | None = Field(default=None, description="Table description")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model name"
    )


class VectorDataBase(BaseModel):
    """Base class for vector data without LanceDB-specific types."""

    vector: list[float] = Field(..., description="Vector data")
    text: str = Field(default="", description="Text description")
    uri: str | None = Field(default=None, description="Optional URI")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional metadata"
    )
    source: str | None = Field(default=None, description="Data source identifier")
    timestamp: str | None = Field(default=None, description="Creation timestamp")


class TextData(BaseModel):
    """Text data for automatic embedding generation."""

    text: str = Field(..., min_length=1, description="Text to embed")
    uri: str | None = Field(default=None, description="Optional URI")
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional metadata"
    )
    source: str | None = Field(default=None, description="Data source identifier")


class SearchQuery(BaseModel):
    """Search query for finding similar vectors."""

    text: str | None = Field(default=None, description="Text query to embed and search")
    vector: list[float] | None = Field(
        default=None, description="Pre-computed query vector"
    )
    limit: int = Field(default=10, gt=0, description="Maximum number of results")
    filter_expr: str | None = Field(default=None, description="SQL filter expression")
    source_filter: str | None = Field(default=None, description="Filter by source")
    threshold: float | None = Field(default=None, description="Similarity threshold")


class DatastoreInfo(BaseModel):
    """Information about a datastore."""

    name: str = Field(..., description="Table name")
    row_count: int = Field(..., description="Number of rows")
    table_schema: dict[str, Any] = Field(..., description="Table schema")
    dimension: int = Field(..., description="Vector dimension")
    embedding_model: str | None = Field(
        default=None, description="Embedding model used"
    )
    created_at: str | None = Field(default=None, description="Creation timestamp")
    description: str | None = Field(default=None, description="Table description")


# Factory function for creating LanceDB models dynamically
def create_vector_data_model(dimension: int = 384):
    """Create a VectorData LanceModel with the specified dimension."""
    try:
        from lancedb.pydantic import LanceModel, Vector

        class VectorData(LanceModel):
            """Vector data with text and optional metadata."""

            vector: Vector = Field(..., dim=dimension, description="Vector data")
            text: str = Field(default="", description="Text description")
            uri: str | None = Field(default=None, description="Optional URI")
            metadata: dict[str, Any] | None = Field(
                default_factory=dict, description="Additional metadata"
            )
            source: str | None = Field(
                default=None, description="Data source identifier"
            )
            timestamp: str | None = Field(
                default=None, description="Creation timestamp"
            )

        return VectorData
    except ImportError:
        # Fallback to base model if LanceDB is not available
        return VectorDataBase


# Default VectorData for backward compatibility (use the base class)
VectorData = VectorDataBase
