"""Database-like models for AAS LanceDB MCP with automatic embedding."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    model_name: str = Field(
        default="BAAI/bge-m3", description="Sentence transformer model name"
    )
    dimension: int = Field(default=1024, description="Vector dimension")
    device: str = Field(default="cpu", description="Device to run model on (cpu/cuda)")
    normalize_embeddings: bool = Field(
        default=True, description="Whether to normalize embeddings"
    )


class ColumnSchema(BaseModel):
    """Schema definition for a table column."""

    name: str = Field(..., min_length=1, description="Column name")
    type: Literal["text", "integer", "float", "boolean", "json"] = Field(..., description="Column data type")
    searchable: bool = Field(default=False, description="Whether this text column should be searchable via embeddings")
    required: bool = Field(default=False, description="Whether this column is required")
    description: str | None = Field(default=None, description="Column description")


class TableSchema(BaseModel):
    """Schema definition for creating/modifying a table."""

    name: str = Field(..., min_length=1, description="Table name")
    columns: list[ColumnSchema] = Field(..., min_length=1, description="Column definitions")
    description: str | None = Field(default=None, description="Table description")
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Embedding model for searchable text columns"
    )


class InsertData(BaseModel):
    """Data to insert into a table."""

    table_name: str = Field(..., description="Name of the table")
    data: dict[str, Any] = Field(..., description="Row data as key-value pairs")


class UpdateData(BaseModel):
    """Data to update in a table."""

    table_name: str = Field(..., description="Name of the table")
    data: dict[str, Any] = Field(..., description="Updated row data as key-value pairs")
    where: dict[str, Any] = Field(..., description="Conditions to match for update")


class QueryData(BaseModel):
    """Query parameters for data retrieval."""

    table_name: str = Field(..., description="Name of the table")
    columns: list[str] | None = Field(default=None, description="Columns to return (null for all)")
    where: dict[str, Any] | None = Field(default=None, description="Conditions to match")
    limit: int | None = Field(default=None, description="Maximum number of rows to return")
    order_by: str | None = Field(default=None, description="Column to sort by")


class SearchQuery(BaseModel):
    """Semantic search query for text similarity."""

    table_name: str = Field(..., description="Name of the table")
    query: str = Field(..., description="Natural language search query")
    columns: list[str] | None = Field(default=None, description="Columns to return (null for all)")
    limit: int = Field(default=10, gt=0, description="Maximum number of results")
    where: dict[str, Any] | None = Field(default=None, description="Additional filtering conditions")
    threshold: float | None = Field(default=None, description="Minimum similarity threshold")


class TableInfo(BaseModel):
    """Information about a table."""

    name: str = Field(..., description="Table name")
    row_count: int = Field(..., description="Number of rows")
    columns: list[ColumnSchema] = Field(..., description="Column definitions")
    searchable_columns: list[str] = Field(..., description="Columns with embedding search enabled")
    embedding_model: str | None = Field(default=None, description="Embedding model used")
    created_at: str | None = Field(default=None, description="Creation timestamp")
    description: str | None = Field(default=None, description="Table description")


class MigrationPlan(BaseModel):
    """Plan for migrating a table to a new schema."""

    source_table: str = Field(..., description="Name of the source table")
    target_schema: TableSchema = Field(..., description="Target schema")
    column_mappings: dict[str, str] | None = Field(default=None, description="Mapping from old to new column names")
    data_transformations: dict[str, str] | None = Field(default=None, description="SQL-like expressions for data transformation")
    backup_name: str | None = Field(default=None, description="Name for backup table (auto-generated if not provided)")


# Factory function for creating LanceDB models dynamically from user schemas
def create_table_model(schema: TableSchema, dimension: int = 1024):
    """Create a LanceDB model based on user-defined schema."""
    try:
        from lancedb.pydantic import LanceModel, Vector

        # Start with field definitions as proper annotations
        fields = {}
        annotations = {}

        # Add user-defined columns
        for col in schema.columns:
            field_default = ... if col.required else None

            if col.type == "text":
                annotations[col.name] = str
                fields[col.name] = Field(default=field_default, description=col.description)

                if col.searchable:
                    # Add embedding vector for searchable text (use suffix for Pydantic compatibility)
                    vector_name = f"{col.name}_vector"
                    annotations[vector_name] = Vector
                    fields[vector_name] = Field(default=None, dim=dimension, description=f"Embedding for {col.name}")

            elif col.type == "integer":
                annotations[col.name] = int
                fields[col.name] = Field(default=field_default, description=col.description)

            elif col.type == "float":
                annotations[col.name] = float
                fields[col.name] = Field(default=field_default, description=col.description)

            elif col.type == "boolean":
                annotations[col.name] = bool
                fields[col.name] = Field(default=field_default, description=col.description)

            elif col.type == "json":
                annotations[col.name] = dict[str, Any]
                if col.required:
                    fields[col.name] = Field(default=..., description=col.description)
                else:
                    fields[col.name] = Field(default_factory=dict, description=col.description)

        # Add system fields for metadata (use single underscore for Pydantic)
        annotations["created_at"] = str
        fields["created_at"] = Field(default=None, description="Creation timestamp")

        annotations["updated_at"] = str
        fields["updated_at"] = Field(default=None, description="Last update timestamp")

        annotations["table_metadata"] = str
        fields["table_metadata"] = Field(default=None, description="JSON table metadata")

        # Add annotations to the fields dict
        fields["__annotations__"] = annotations

        # Create the dynamic model class
        DynamicModel = type(f"{schema.name}Model", (LanceModel,), fields)
        return DynamicModel

    except (ImportError, NameError, Exception):
        # Fallback to dictionary-based storage if LanceDB is not available or other errors
        return dict


def get_searchable_columns(schema: TableSchema) -> list[str]:
    """Get list of columns that should be searchable via embeddings."""
    return [col.name for col in schema.columns if col.type == "text" and col.searchable]


def get_embedding_columns(schema: TableSchema) -> dict[str, str]:
    """Get mapping of text columns to their embedding vector column names."""
    return {col.name: f"{col.name}_vector" for col in schema.columns if col.type == "text" and col.searchable}
