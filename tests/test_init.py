"""Tests for __init__.py functionality."""

from lancedb_mcp import __version__
from lancedb_mcp.models import ColumnSchema, TableSchema


def test_schema():
    """Test TableSchema model."""
    column = ColumnSchema(name="title", type="text", searchable=True)
    schema = TableSchema(name="test_table", columns=[column])
    assert schema.name == "test_table"
    assert len(schema.columns) == 1
    assert schema.columns[0].searchable is True


def test_version():
    """Test version getter."""
    assert __version__ == "0.1.0"
