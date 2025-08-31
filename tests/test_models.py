"""Test model functionality without LanceDB dependencies."""

import pytest

from lancedb_mcp.models import (
    ColumnSchema,
    EmbeddingConfig,
    InsertData,
    QueryData,
    SearchQuery,
    TableInfo,
    TableSchema,
    create_table_model,
)


def test_embedding_config():
    """Test EmbeddingConfig model."""
    config = EmbeddingConfig()
    assert config.model_name == "BAAI/bge-m3"
    assert config.dimension == 1024
    assert config.device == "cpu"
    assert config.normalize_embeddings is True


def test_column_schema():
    """Test ColumnSchema model."""
    col = ColumnSchema(name="title", type="text", searchable=True)
    assert col.name == "title"
    assert col.type == "text"
    assert col.searchable is True
    assert col.required is False


def test_table_schema():
    """Test TableSchema model."""
    columns = [
        ColumnSchema(name="title", type="text", searchable=True),
        ColumnSchema(name="price", type="float", required=True),
    ]
    schema = TableSchema(name="products", columns=columns)
    assert schema.name == "products"
    assert len(schema.columns) == 2
    assert schema.embedding_model == "BAAI/bge-m3"


def test_insert_data():
    """Test InsertData model."""
    data = InsertData(table_name="products", data={"title": "Widget", "price": 9.99})
    assert data.table_name == "products"
    assert data.data["title"] == "Widget"
    assert data.data["price"] == 9.99


def test_query_data():
    """Test QueryData model."""
    query = QueryData(table_name="products", where={"price": 9.99}, limit=5)
    assert query.table_name == "products"
    assert query.where["price"] == 9.99
    assert query.limit == 5


def test_search_query():
    """Test SearchQuery model."""
    query = SearchQuery(table_name="products", query="Widget")
    assert query.table_name == "products"
    assert query.query == "Widget"
    assert query.limit == 10


def test_table_info():
    """Test TableInfo model."""
    columns = [ColumnSchema(name="title", type="text", searchable=True)]
    info = TableInfo(
        name="products", row_count=100, columns=columns, searchable_columns=["title"]
    )
    assert info.name == "products"
    assert info.row_count == 100
    assert len(info.columns) == 1
    assert "title" in info.searchable_columns


def test_create_table_model_fallback():
    """Test create_table_model fallback behavior."""
    columns = [ColumnSchema(name="title", type="text", searchable=True)]
    schema = TableSchema(name="test", columns=columns)

    # The function should return either a class or dict fallback
    model = create_table_model(schema)
    assert model is not None

    # It's either a class (if LanceDB available) or dict (fallback)
    assert model is dict or hasattr(model, "__name__")


def test_model_validation():
    """Test model validation."""
    # Test column validation
    with pytest.raises(ValueError):
        ColumnSchema(name="", type="text")  # empty name should fail

    # Test table validation
    with pytest.raises(ValueError):
        TableSchema(name="", columns=[])  # empty name should fail
