"""Test server functionality."""

import os
import tempfile

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lancedb_mcp.models import (
    ColumnSchema,
    InsertData,
    QueryData,
    SearchQuery,
    TableSchema,
)
from lancedb_mcp.server import set_db_uri


@pytest.fixture
async def client():
    """Create a test client."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, "test.lance")
    set_db_uri(test_db)

    # Create server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "lancedb_mcp.server"],
        env={"LANCEDB_URI": test_db},
    )

    # Create client session
    read, write = await stdio_client(server_params).__aenter__()
    session = await ClientSession(read, write).__aenter__()
    await session.initialize()

    yield session

    # Cleanup
    await session.__aexit__(None, None, None)
    await stdio_client(server_params).__aexit__(None, None, None)
    os.rmdir(temp_dir)


@pytest.mark.asyncio
async def test_create_table(client):
    """Test creating a table with database-like interface."""
    # Create table schema
    columns = [
        ColumnSchema(name="title", type="text", searchable=True, required=True),
        ColumnSchema(name="price", type="float", required=True),
        ColumnSchema(name="description", type="text", searchable=True)
    ]
    schema = TableSchema(name="products", columns=columns)

    tools = await client.list_tools()
    assert len(tools) == 10  # We have 10 tools in our database-like server

    result = await client.call_tool("create_table", {"schema": schema.model_dump()})
    assert "Created table 'products'" in result[0].text


@pytest.mark.asyncio
async def test_insert_data(client):
    """Test inserting data with automatic embedding generation."""
    # Create table first
    columns = [
        ColumnSchema(name="title", type="text", searchable=True, required=True),
        ColumnSchema(name="price", type="float", required=True)
    ]
    schema = TableSchema(name="products", columns=columns)
    await client.call_tool("create_table", {"schema": schema.model_dump()})

    # Insert test data
    insert_data = InsertData(
        table_name="products",
        data={"title": "Test Widget", "price": 29.99}
    )
    result = await client.call_tool("insert", {"data": insert_data.model_dump()})
    assert "Inserted 1 row into table 'products'" in result[0].text


@pytest.mark.asyncio
async def test_select_data(client):
    """Test selecting data from a table."""
    # Create table and insert data
    columns = [
        ColumnSchema(name="title", type="text", required=True),
        ColumnSchema(name="price", type="float", required=True)
    ]
    schema = TableSchema(name="products", columns=columns)
    await client.call_tool("create_table", {"schema": schema.model_dump()})

    # Insert test data
    insert_data = InsertData(
        table_name="products",
        data={"title": "Test Widget", "price": 29.99}
    )
    await client.call_tool("insert", {"data": insert_data.model_dump()})

    # Test select
    query = QueryData(table_name="products", limit=10)
    result = await client.call_tool("select", {"query": query.model_dump()})
    assert "Test Widget" in result[0].text


@pytest.mark.asyncio
async def test_semantic_search(client):
    """Test semantic search across searchable text columns."""
    # Create table with searchable text
    columns = [
        ColumnSchema(name="title", type="text", searchable=True, required=True),
        ColumnSchema(name="description", type="text", searchable=True)
    ]
    schema = TableSchema(name="articles", columns=columns)
    await client.call_tool("create_table", {"schema": schema.model_dump()})

    # Insert test data
    insert_data = InsertData(
        table_name="articles",
        data={
            "title": "Machine Learning Basics",
            "description": "An introduction to machine learning algorithms"
        }
    )
    await client.call_tool("insert", {"data": insert_data.model_dump()})

    # Test semantic search
    search_query = SearchQuery(
        table_name="articles",
        query="AI algorithms",
        limit=5
    )
    result = await client.call_tool("search", {"query": search_query.model_dump()})
    assert "Machine Learning" in result[0].text


@pytest.mark.asyncio
async def test_list_tables(client):
    """Test listing tables."""
    # Create a test table
    columns = [ColumnSchema(name="name", type="text", required=True)]
    schema = TableSchema(name="test_table", columns=columns)
    await client.call_tool("create_table", {"schema": schema.model_dump()})

    # Test list tables
    result = await client.call_tool("list_tables", {})
    assert "test_table" in result[0].text
