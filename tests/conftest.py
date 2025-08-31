"""Test configuration."""

import os

import pytest

# Set environment variables for testing
os.environ["LANCEDB_URI"] = ".lancedb"

# Configure pytest for async tests
pytest_plugins = ["pytest_asyncio"]


def pytest_collection_modifyitems(config, items):
    """Skip integration tests that require server startup."""
    skip_integration = pytest.mark.skip(
        reason="Integration tests require MCP server startup - skipped for now"
    )
    for item in items:
        if "test_server.py" in str(item.fspath):
            item.add_marker(skip_integration)
