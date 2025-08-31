"""Test resources and prompts functionality."""

import pytest

from lancedb_mcp.server import (
    handle_get_prompt,
    handle_list_prompts,
    handle_list_resources,
    handle_read_resource,
)


@pytest.mark.asyncio
async def test_list_resources():
    """Test listing available resources."""
    resources = await handle_list_resources()

    # Should have at least the overview resource
    assert len(resources) >= 1

    # Check overview resource exists
    overview_found = False
    for resource in resources:
        if str(resource.uri) == "lancedb://overview":
            overview_found = True
            assert resource.name == "Database Overview"
            assert resource.mimeType == "application/json"

    assert overview_found, "Overview resource not found"


@pytest.mark.asyncio
async def test_read_overview_resource():
    """Test reading the overview resource."""
    overview_json = await handle_read_resource("lancedb://overview")

    # Should be valid JSON
    import json

    overview = json.loads(overview_json)

    # Check required fields
    assert "database_uri" in overview
    assert "total_tables" in overview
    assert "total_rows" in overview
    assert "default_embedding_model" in overview
    assert "tables" in overview
    assert "generated_at" in overview

    # Check values
    assert overview["default_embedding_model"] == "BAAI/bge-m3"
    assert isinstance(overview["tables"], list)


@pytest.mark.asyncio
async def test_read_nonexistent_resource():
    """Test reading a nonexistent resource."""
    result = await handle_read_resource("lancedb://nonexistent")

    # Should return error JSON
    import json

    error = json.loads(result)

    assert "error" in error
    assert "nonexistent" in error["error"]


@pytest.mark.asyncio
async def test_list_prompts():
    """Test listing available prompts."""
    prompts = await handle_list_prompts()

    # Should have expected prompts
    assert len(prompts) == 5

    prompt_names = [p.name for p in prompts]
    expected_prompts = [
        "analyze_table",
        "design_schema",
        "optimize_queries",
        "troubleshoot_performance",
        "migration_planning",
    ]

    for expected in expected_prompts:
        assert expected in prompt_names, f"Missing prompt: {expected}"


@pytest.mark.asyncio
async def test_get_design_schema_prompt():
    """Test getting the design_schema prompt."""
    result = await handle_get_prompt(
        "design_schema",
        {
            "use_case": "Test catalog",
            "data_types": "names, descriptions",
            "search_requirements": "basic search",
        },
    )

    assert result.description == "Generated design_schema prompt"
    assert len(result.messages) == 1

    message = result.messages[0]
    assert message.role == "user"
    assert "Test catalog" in message.content.text
    assert "BGE-M3" in message.content.text


@pytest.mark.asyncio
async def test_get_prompt_missing_required_arg():
    """Test getting a prompt with missing required arguments."""
    result = await handle_get_prompt("design_schema", {})

    # Should return error message
    assert len(result.messages) == 1
    message = result.messages[0]
    assert "Error generating prompt" in message.content.text
    assert "use_case is required" in message.content.text


@pytest.mark.asyncio
async def test_get_analyze_table_prompt_nonexistent_table():
    """Test analyze_table prompt with nonexistent table."""
    result = await handle_get_prompt(
        "analyze_table", {"table_name": "nonexistent_table"}
    )

    # Should return error message
    assert len(result.messages) == 1
    message = result.messages[0]
    assert "Error generating prompt" in message.content.text
    assert "does not exist" in message.content.text


@pytest.mark.asyncio
async def test_get_unknown_prompt():
    """Test getting an unknown prompt."""
    result = await handle_get_prompt("unknown_prompt", {})

    # Should return error message
    assert len(result.messages) == 1
    message = result.messages[0]
    assert "Error generating prompt" in message.content.text
    assert "Unknown prompt" in message.content.text
