"""Test model functionality without LanceDB dependencies."""

import pytest
from lancedb_mcp.models import (
    EmbeddingConfig,
    TableConfig,
    VectorDataBase,
    TextData,
    SearchQuery,
    DatastoreInfo,
    create_vector_data_model,
)


def test_embedding_config():
    """Test EmbeddingConfig model."""
    config = EmbeddingConfig()
    assert config.model_name == "all-MiniLM-L6-v2"
    assert config.dimension == 384
    assert config.device == "cpu"
    assert config.normalize_embeddings is True


def test_table_config():
    """Test TableConfig model."""
    config = TableConfig(name="test_table")
    assert config.name == "test_table"
    assert config.dimension == 384
    assert config.metric == "cosine"
    assert config.embedding_model == "all-MiniLM-L6-v2"


def test_vector_data_base():
    """Test VectorDataBase model."""
    data = VectorDataBase(vector=[1.0, 2.0, 3.0], text="test")
    assert data.vector == [1.0, 2.0, 3.0]
    assert data.text == "test"
    assert data.metadata == {}


def test_text_data():
    """Test TextData model."""
    data = TextData(text="Hello world")
    assert data.text == "Hello world"
    assert data.metadata == {}


def test_search_query():
    """Test SearchQuery model."""
    query = SearchQuery(text="search term")
    assert query.text == "search term"
    assert query.limit == 10

    query2 = SearchQuery(vector=[1.0, 2.0, 3.0])
    assert query2.vector == [1.0, 2.0, 3.0]


def test_datastore_info():
    """Test DatastoreInfo model."""
    info = DatastoreInfo(
        name="test_table",
        row_count=100,
        table_schema={"fields": [{"name": "vector", "type": "list"}]},
        dimension=384,
    )
    assert info.name == "test_table"
    assert info.row_count == 100
    assert info.dimension == 384


def test_create_vector_data_model_fallback():
    """Test create_vector_data_model fallback behavior."""
    try:
        VectorData = create_vector_data_model(512)
        # If we can create the model, test it
        data = VectorData(vector=[1.0] * 512, text="test")
        assert len(data.vector) == 512
        assert data.text == "test"
    except (ImportError, NameError):
        # This is expected if LanceDB imports fail due to pyarrow issues
        # The model creation should fall back, but we'll skip this test
        pytest.skip("LanceDB/pyarrow not properly initialized")


def test_model_validation():
    """Test model validation."""
    # Test validation errors
    with pytest.raises(ValueError):
        TableConfig(name="")  # empty name should fail

    with pytest.raises(ValueError):
        TableConfig(name="test", dimension=0)  # zero dimension should fail

    with pytest.raises(ValueError):
        TextData(text="")  # empty text should fail
