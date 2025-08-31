"""Enhanced AAS LanceDB MCP server with sentence transformers integration."""

import json
import logging
import os
import pathlib
from datetime import datetime
from typing import Any

import lancedb
import mcp.server.stdio
import mcp.types as types
import pandas as pd
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .embedding import embedding_manager
from .models import (
    EmbeddingConfig,
    InsertData,
    MigrationPlan,
    QueryData,
    SearchQuery,
    TableInfo,
    TableSchema,
    UpdateData,
    create_table_model,
    get_searchable_columns,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database URI and embedding config
DB_URI = os.getenv("LANCEDB_URI", ".aas_lancedb")
# Default to BGE-M3 - excellent multilingual model
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def set_db_uri(uri: str) -> None:
    """Set the database URI."""
    global DB_URI
    DB_URI = uri


def get_db() -> lancedb.DBConnection:
    """Get database connection."""
    logger.info(f"Connecting to database at {DB_URI}")
    try:
        pathlib.Path(DB_URI).parent.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(DB_URI)
    except Exception as err:
        logger.error(f"Failed to connect to database: {err}")
        raise err


def get_embedding_config(model_name: str | None = None) -> EmbeddingConfig:
    """Get embedding configuration."""
    return EmbeddingConfig(
        model_name=model_name or DEFAULT_EMBEDDING_MODEL,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
    )


# Create MCP server instance
server = Server("aas-lancedb-server")


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources (table schemas, data samples, statistics)."""
    try:
        db = get_db()
        resources = []

        # Add overview resource
        resources.append(
            types.Resource(
                uri="lancedb://overview",
                name="Database Overview",
                description="Overview of all tables and database statistics",
                mimeType="application/json",
            )
        )

        # Add resources for each table
        for table_name in db.table_names():
            # Table schema resource
            resources.append(
                types.Resource(
                    uri=f"lancedb://tables/{table_name}/schema",
                    name=f"{table_name} Schema",
                    description=f"Schema definition for table {table_name}",
                    mimeType="application/json",
                )
            )

            # Table sample data resource
            resources.append(
                types.Resource(
                    uri=f"lancedb://tables/{table_name}/sample",
                    name=f"{table_name} Sample Data",
                    description=f"Sample data from table {table_name}",
                    mimeType="application/json",
                )
            )

            # Table statistics resource
            resources.append(
                types.Resource(
                    uri=f"lancedb://tables/{table_name}/stats",
                    name=f"{table_name} Statistics",
                    description=f"Statistics and metadata for table {table_name}",
                    mimeType="application/json",
                )
            )

        return resources
    except Exception as e:
        logger.error(f"Failed to list resources: {e}")
        return []


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource contents based on URI."""
    try:
        db = get_db()

        if uri == "lancedb://overview":
            # Database overview
            tables = []
            total_rows = 0

            for table_name in db.table_names():
                try:
                    table = db.open_table(table_name)
                    row_count = max(0, table.count_rows() - 1)  # Subtract metadata row
                    total_rows += row_count

                    # Get metadata if available
                    metadata = await _get_table_metadata(table)
                    schema_info = {}
                    if metadata:
                        schema = TableSchema.model_validate(metadata["schema"])
                        schema_info = {
                            "columns": len(schema.columns),
                            "searchable_columns": len(get_searchable_columns(schema)),
                            "embedding_model": schema.embedding_model,
                        }

                    tables.append(
                        {"name": table_name, "rows": row_count, **schema_info}
                    )
                except Exception as e:
                    logger.warning(f"Failed to get info for table {table_name}: {e}")
                    tables.append({"name": table_name, "error": str(e)})

            overview = {
                "database_uri": DB_URI,
                "total_tables": len(tables),
                "total_rows": total_rows,
                "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
                "tables": tables,
                "generated_at": datetime.now().isoformat(),
            }
            return json.dumps(overview, indent=2)

        elif uri.startswith("lancedb://tables/"):
            # Parse table name and resource type
            parts = uri.replace("lancedb://tables/", "").split("/")
            if len(parts) != 2:
                raise ValueError(f"Invalid table resource URI: {uri}")

            table_name, resource_type = parts

            if table_name not in db.table_names():
                raise ValueError(f"Table '{table_name}' does not exist")

            table = db.open_table(table_name)

            if resource_type == "schema":
                # Table schema
                metadata = await _get_table_metadata(table)
                if metadata:
                    schema = metadata["schema"]
                    return json.dumps(
                        {
                            "table_name": table_name,
                            "schema": schema,
                            "searchable_columns": metadata.get(
                                "searchable_columns", []
                            ),
                            "embedding_model": metadata.get(
                                "embedding_model", DEFAULT_EMBEDDING_MODEL
                            ),
                            "created_at": metadata.get("created_at"),
                            "generated_at": datetime.now().isoformat(),
                        },
                        indent=2,
                    )
                else:
                    return json.dumps(
                        {
                            "table_name": table_name,
                            "error": "No schema metadata found - table may have been created with older version",
                            "generated_at": datetime.now().isoformat(),
                        },
                        indent=2,
                    )

            elif resource_type == "sample":
                # Sample data
                df = table.to_pandas()
                user_data = (
                    df[df["table_metadata"].isna()]
                    if "table_metadata" in df.columns
                    else df
                )

                # Get up to 5 sample rows, excluding vector columns for readability
                sample_df = user_data.head(5)
                display_cols = [
                    col
                    for col in sample_df.columns
                    if not col.endswith("_vector") and col != "table_metadata"
                ]
                sample_df = sample_df[display_cols]

                sample = {
                    "table_name": table_name,
                    "sample_size": len(sample_df),
                    "total_rows": len(user_data),
                    "columns": display_cols,
                    "data": sample_df.to_dict(orient="records"),
                    "generated_at": datetime.now().isoformat(),
                }
                return json.dumps(sample, indent=2)

            elif resource_type == "stats":
                # Table statistics
                df = table.to_pandas()
                user_data = (
                    df[df["table_metadata"].isna()]
                    if "table_metadata" in df.columns
                    else df
                )

                # Calculate basic statistics
                stats = {
                    "table_name": table_name,
                    "total_rows": len(user_data),
                    "total_columns": len(df.columns),
                    "column_info": {},
                    "generated_at": datetime.now().isoformat(),
                }

                # Per-column statistics
                for col in df.columns:
                    if col not in ["table_metadata"] and not col.endswith("_vector"):
                        col_data = (
                            user_data[col] if col in user_data.columns else pd.Series()
                        )
                        stats["column_info"][col] = {
                            "type": str(col_data.dtype)
                            if not col_data.empty
                            else "unknown",
                            "non_null_count": int(col_data.count())
                            if not col_data.empty
                            else 0,
                            "null_count": int(col_data.isnull().sum())
                            if not col_data.empty
                            else 0,
                        }

                        # Add unique values for small datasets
                        if not col_data.empty and len(col_data.dropna()) <= 20:
                            unique_vals = col_data.dropna().unique().tolist()
                            # Convert numpy types to native Python types for JSON serialization
                            stats["column_info"][col]["unique_values"] = [
                                val.item() if hasattr(val, "item") else val
                                for val in unique_vals
                            ]

                return json.dumps(stats, indent=2)
            else:
                raise ValueError(f"Unknown resource type: {resource_type}")
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    except Exception as e:
        error_response = {
            "error": f"Failed to read resource {uri}: {str(e)}",
            "generated_at": datetime.now().isoformat(),
        }
        return json.dumps(error_response, indent=2)


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """List available prompts for database operations."""
    return [
        types.Prompt(
            name="analyze_table",
            description="Generate analysis questions and insights for a database table",
            arguments=[
                types.PromptArgument(
                    name="table_name",
                    description="Name of the table to analyze",
                    required=True,
                ),
                types.PromptArgument(
                    name="focus_area",
                    description="Specific area to focus on (e.g., 'data_quality', 'patterns', 'relationships')",
                    required=False,
                ),
            ],
        ),
        types.Prompt(
            name="design_schema",
            description="Help design a table schema based on use case requirements",
            arguments=[
                types.PromptArgument(
                    name="use_case",
                    description="Description of the intended use case for the table",
                    required=True,
                ),
                types.PromptArgument(
                    name="data_types",
                    description="Expected data types and examples",
                    required=False,
                ),
                types.PromptArgument(
                    name="search_requirements",
                    description="Text search and semantic search requirements",
                    required=False,
                ),
            ],
        ),
        types.Prompt(
            name="optimize_queries",
            description="Suggest query optimizations and indexing strategies",
            arguments=[
                types.PromptArgument(
                    name="table_name",
                    description="Name of the table to optimize",
                    required=True,
                ),
                types.PromptArgument(
                    name="query_patterns",
                    description="Common query patterns and use cases",
                    required=False,
                ),
            ],
        ),
        types.Prompt(
            name="troubleshoot_performance",
            description="Diagnose and resolve database performance issues",
            arguments=[
                types.PromptArgument(
                    name="symptoms",
                    description="Description of performance issues observed",
                    required=True,
                ),
                types.PromptArgument(
                    name="table_names",
                    description="Affected tables (comma-separated)",
                    required=False,
                ),
            ],
        ),
        types.Prompt(
            name="migration_planning",
            description="Plan safe table schema migrations and data transformations",
            arguments=[
                types.PromptArgument(
                    name="source_table",
                    description="Name of the table to migrate",
                    required=True,
                ),
                types.PromptArgument(
                    name="desired_changes",
                    description="Desired schema changes and transformations",
                    required=True,
                ),
                types.PromptArgument(
                    name="constraints",
                    description="Migration constraints (downtime, data safety, etc.)",
                    required=False,
                ),
            ],
        ),
    ]


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Generate prompt content based on name and arguments."""
    try:
        db = get_db()
        args = arguments or {}

        if name == "analyze_table":
            table_name = args.get("table_name", "")
            focus_area = args.get("focus_area", "general insights")

            if not table_name:
                raise ValueError("table_name is required")

            if table_name not in db.table_names():
                raise ValueError(f"Table '{table_name}' does not exist")

            # Get table information for context
            table_info = await _get_table_context(db, table_name)

            messages = [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Analyze the database table '{table_name}' with focus on {focus_area}.

Table Information:
{table_info}

Please provide:
1. Key insights about the data patterns and distribution
2. Data quality assessment
3. Potential issues or anomalies to investigate
4. Recommendations for optimization or improvements
5. Suggested queries for deeper analysis

Focus your analysis on: {focus_area}""",
                    ),
                )
            ]

        elif name == "design_schema":
            use_case = args.get("use_case", "")
            data_types = args.get("data_types", "not specified")
            search_requirements = args.get(
                "search_requirements", "basic search capabilities"
            )

            if not use_case:
                raise ValueError("use_case is required")

            messages = [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Help me design a table schema for this use case:

Use Case: {use_case}

Data Types Expected: {data_types}
Search Requirements: {search_requirements}

Please provide:
1. Recommended table schema with column definitions
2. Which columns should be searchable via embeddings
3. Appropriate data types and constraints
4. Embedding model recommendations
5. Example data structure
6. Potential scaling considerations

Current system supports:
- Text, integer, float, boolean, and JSON column types
- Automatic embedding generation for searchable text columns
- BGE-M3 multilingual embeddings (1024 dimensions)
- Safe schema migrations with validation""",
                    ),
                )
            ]

        elif name == "optimize_queries":
            table_name = args.get("table_name", "")
            query_patterns = args.get("query_patterns", "not specified")

            if not table_name:
                raise ValueError("table_name is required")

            if table_name not in db.table_names():
                raise ValueError(f"Table '{table_name}' does not exist")

            table_info = await _get_table_context(db, table_name)

            messages = [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Optimize queries and performance for table '{table_name}'.

Table Information:
{table_info}

Common Query Patterns: {query_patterns}

Please provide:
1. Query optimization strategies for common patterns
2. Indexing recommendations (if applicable)
3. Embedding search optimization tips
4. Data organization suggestions
5. Performance monitoring approaches
6. Specific query examples with improvements

Consider the LanceDB/vector database characteristics and BGE-M3 embeddings.""",
                    ),
                )
            ]

        elif name == "troubleshoot_performance":
            symptoms = args.get("symptoms", "")
            table_names = args.get("table_names", "")

            if not symptoms:
                raise ValueError("symptoms is required")

            # Get database overview for context
            overview_resource = await handle_read_resource("lancedb://overview")

            context_info = "Database Overview:\n" + overview_resource

            if table_names:
                table_list = [t.strip() for t in table_names.split(",")]
                for table_name in table_list:
                    if table_name in db.table_names():
                        table_info = await _get_table_context(db, table_name)
                        context_info += f"\n\nTable {table_name}:\n{table_info}"

            messages = [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Troubleshoot performance issues in the LanceDB database.

Symptoms: {symptoms}

Affected Tables: {table_names if table_names else "not specified"}

Context:
{context_info}

Please provide:
1. Likely root causes based on symptoms
2. Diagnostic steps to confirm the issues
3. Specific remediation recommendations
4. Prevention strategies for the future
5. Performance monitoring suggestions
6. When to consider schema migrations or optimizations

Consider LanceDB-specific performance characteristics, embedding operations, and vector search patterns.""",
                    ),
                )
            ]

        elif name == "migration_planning":
            source_table = args.get("source_table", "")
            desired_changes = args.get("desired_changes", "")
            constraints = args.get(
                "constraints", "minimize downtime, ensure data safety"
            )

            if not source_table or not desired_changes:
                raise ValueError("source_table and desired_changes are required")

            if source_table not in db.table_names():
                raise ValueError(f"Table '{source_table}' does not exist")

            table_info = await _get_table_context(db, source_table)

            messages = [
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Plan a safe migration for table '{source_table}'.

Current Table Information:
{table_info}

Desired Changes: {desired_changes}

Constraints: {constraints}

Please provide:
1. Step-by-step migration plan
2. Risk assessment and mitigation strategies
3. Backup and rollback procedures
4. Data validation approaches
5. Testing recommendations
6. Timeline estimates
7. Impact on existing applications

Consider the built-in migration tools available:
- Safe table migration with automatic backup
- Schema validation before migration
- Column mapping and data transformation
- Embedding regeneration for searchable columns""",
                    ),
                )
            ]

        else:
            raise ValueError(f"Unknown prompt: {name}")

        return types.GetPromptResult(
            description=f"Generated {name} prompt", messages=messages
        )

    except Exception as e:
        # Return error as a prompt message
        messages = [
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text", text=f"Error generating prompt '{name}': {str(e)}"
                ),
            )
        ]
        return types.GetPromptResult(
            description=f"Error in {name} prompt", messages=messages
        )


async def _get_table_context(db, table_name: str) -> str:
    """Get formatted context information about a table for prompts."""
    try:
        # Get schema info
        schema_resource = await handle_read_resource(
            f"lancedb://tables/{table_name}/schema"
        )
        schema_data = json.loads(schema_resource)

        # Get sample data
        sample_resource = await handle_read_resource(
            f"lancedb://tables/{table_name}/sample"
        )
        sample_data = json.loads(sample_resource)

        # Get statistics
        stats_resource = await handle_read_resource(
            f"lancedb://tables/{table_name}/stats"
        )
        stats_data = json.loads(stats_resource)

        context = f"""Schema: {json.dumps(schema_data.get("schema", {}), indent=2)}
Sample Data ({sample_data.get("sample_size", 0)} rows): {json.dumps(sample_data.get("data", []), indent=2)}
Statistics: {json.dumps(stats_data.get("column_info", {}), indent=2)}
Total Rows: {stats_data.get("total_rows", 0)}
Searchable Columns: {schema_data.get("searchable_columns", [])}
Embedding Model: {schema_data.get("embedding_model", "unknown")}"""

        return context

    except Exception as e:
        return f"Error retrieving context for table {table_name}: {str(e)}"


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available database-like tools."""
    return [
        # Schema Management
        types.Tool(
            name="create_table",
            description="Create a new table with defined schema (text columns can be made searchable)",
            arguments=[
                types.ToolArgument(
                    name="schema",
                    description="Table schema definition",
                    schema=TableSchema.model_json_schema(),
                )
            ],
        ),
        types.Tool(
            name="list_tables",
            description="List all tables with basic information",
            arguments=[],
        ),
        types.Tool(
            name="describe_table",
            description="Get detailed schema and information about a table",
            arguments=[
                types.ToolArgument(
                    name="table_name", description="Name of the table", type="string"
                ),
            ],
        ),
        types.Tool(
            name="drop_table",
            description="Delete a table permanently",
            arguments=[
                types.ToolArgument(
                    name="table_name",
                    description="Name of the table to delete",
                    type="string",
                ),
            ],
        ),
        # Data Operations (CRUD)
        types.Tool(
            name="insert",
            description="Insert data into a table (embeddings generated automatically for searchable text columns)",
            arguments=[
                types.ToolArgument(
                    name="data",
                    description="Insert data",
                    schema=InsertData.model_json_schema(),
                )
            ],
        ),
        types.Tool(
            name="select",
            description="Query data from a table with optional filtering and ordering",
            arguments=[
                types.ToolArgument(
                    name="query",
                    description="Query parameters",
                    schema=QueryData.model_json_schema(),
                )
            ],
        ),
        types.Tool(
            name="update",
            description="Update data in a table (embeddings updated automatically for searchable text columns)",
            arguments=[
                types.ToolArgument(
                    name="data",
                    description="Update data",
                    schema=UpdateData.model_json_schema(),
                )
            ],
        ),
        types.Tool(
            name="delete",
            description="Delete rows from a table based on conditions",
            arguments=[
                types.ToolArgument(
                    name="table_name", description="Name of the table", type="string"
                ),
                types.ToolArgument(
                    name="where",
                    description="Conditions for deletion (JSON object)",
                    type="object",
                ),
            ],
        ),
        # Semantic Search
        types.Tool(
            name="search",
            description="Semantic search across searchable text columns using natural language",
            arguments=[
                types.ToolArgument(
                    name="query",
                    description="Search query",
                    schema=SearchQuery.model_json_schema(),
                )
            ],
        ),
        # Migration & Schema Evolution
        types.Tool(
            name="migrate_table",
            description="Safely migrate a table to a new schema with validation and backup",
            arguments=[
                types.ToolArgument(
                    name="migration",
                    description="Migration plan",
                    schema=MigrationPlan.model_json_schema(),
                )
            ],
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle database-like tool calls."""
    try:
        db = get_db()

        # Schema Management Tools
        if name == "create_table":
            schema = TableSchema.model_validate(arguments["schema"])
            return await create_table_handler(db, schema)

        elif name == "list_tables":
            return await list_tables_handler(db)

        elif name == "describe_table":
            table_name = arguments["table_name"]
            return await describe_table_handler(db, table_name)

        elif name == "drop_table":
            table_name = arguments["table_name"]
            return await drop_table_handler(db, table_name)

        # Data Operations (CRUD)
        elif name == "insert":
            data = InsertData.model_validate(arguments["data"])
            return await insert_handler(db, data)

        elif name == "select":
            query = QueryData.model_validate(arguments["query"])
            return await select_handler(db, query)

        elif name == "update":
            data = UpdateData.model_validate(arguments["data"])
            return await update_handler(db, data)

        elif name == "delete":
            table_name = arguments["table_name"]
            where = arguments["where"]
            return await delete_handler(db, table_name, where)

        # Semantic Search
        elif name == "search":
            query = SearchQuery.model_validate(arguments["query"])
            return await search_handler(db, query)

        # Migration
        elif name == "migrate_table":
            migration = MigrationPlan.model_validate(arguments["migration"])
            return await migrate_table_handler(db, migration)

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as err:
        error_msg = f"Failed to execute {name}: {err}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


# Helper Functions
async def _get_table_metadata(table):
    """Extract metadata from a LanceDB table."""
    try:
        # Look for metadata in the table (stored in table_metadata column)
        df = table.to_pandas()
        if df.empty:
            return None

        # Check for metadata row
        metadata_rows = df[df["table_metadata"].notna()]
        if not metadata_rows.empty:
            metadata_json = metadata_rows.iloc[0]["table_metadata"]
            return json.loads(metadata_json)

        return None
    except Exception:
        return None


# Tool Handler Functions
async def create_table_handler(db, schema: TableSchema) -> list[types.TextContent]:
    """Create a new table with user-defined schema."""
    try:
        # Get embedding config
        config = get_embedding_config()

        # Create the dynamic model
        model_class = create_table_model(schema, config.dimension)

        if model_class is dict:  # Fallback case
            raise ImportError("LanceDB not available")

        # Create LanceDB table
        table = db.create_table(name=schema.name, schema=model_class, mode="overwrite")

        # Store schema metadata
        searchable_cols = get_searchable_columns(schema)
        metadata = {
            "schema": schema.model_dump(),
            "searchable_columns": searchable_cols,
            "embedding_model": schema.embedding_model,
            "created_at": datetime.now().isoformat(),
        }

        # Store metadata row
        metadata_row = {col.name: None for col in schema.columns}
        metadata_row["created_at"] = datetime.now().isoformat()
        metadata_row["table_metadata"] = json.dumps(metadata)

        df = pd.DataFrame([metadata_row])
        table.add(df)

        logger.info(f"Created table {schema.name} with {len(schema.columns)} columns")

        searchable_info = (
            f" ({len(searchable_cols)} searchable)" if searchable_cols else ""
        )
        return [
            types.TextContent(
                type="text",
                text=f"Created table '{schema.name}' with {len(schema.columns)} columns{searchable_info}",
            )
        ]

    except Exception as e:
        raise Exception(f"Failed to create table: {e}")


async def list_tables_handler(db) -> list[types.TextContent]:
    """List all tables with basic information."""
    try:
        table_names = db.table_names()
        tables_info = []

        for table_name in table_names:
            try:
                table = db.open_table(table_name)
                count = table.count_rows()

                # Try to get metadata
                searchable_count = 0
                try:
                    # This is a simplified check - would need proper metadata retrieval
                    searchable_count = 0  # Placeholder
                except Exception:
                    pass

                tables_info.append(
                    {
                        "name": table_name,
                        "row_count": max(0, count - 1),  # Subtract metadata row
                        "searchable_columns": searchable_count,
                    }
                )
            except Exception as e:
                tables_info.append({"name": table_name, "error": str(e)})

        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"tables": tables_info, "total_tables": len(table_names)}, indent=2
                ),
            )
        ]

    except Exception as e:
        raise Exception(f"Failed to list tables: {e}")


# Placeholder handlers for other operations
async def describe_table_handler(db, table_name: str) -> list[types.TextContent]:
    """Get detailed table schema and info."""
    try:
        # Open the table
        table = db.open_table(table_name)

        # Get table metadata
        metadata = await _get_table_metadata(table)
        if not metadata:
            # Fallback for tables without metadata
            df = table.to_pandas()
            user_data = (
                df[df["table_metadata"].isna()]
                if "table_metadata" in df.columns
                else df
            )

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "name": table_name,
                            "row_count": len(user_data),
                            "columns": list(df.columns),
                            "error": "No schema metadata found - this table may have been created with an older version",
                        },
                        indent=2,
                    ),
                )
            ]

        # Parse schema from metadata
        schema = TableSchema.model_validate(metadata["schema"])
        searchable_columns = metadata.get("searchable_columns", [])

        # Get current row count (excluding metadata)
        df = table.to_pandas()
        user_data = df[df["table_metadata"].isna()]
        row_count = len(user_data)

        # Build detailed table info
        table_info = TableInfo(
            name=table_name,
            row_count=row_count,
            columns=schema.columns,
            searchable_columns=searchable_columns,
            embedding_model=schema.embedding_model,
            created_at=metadata.get("created_at"),
            description=schema.description,
        )

        logger.info(
            f"Described table {table_name}: {len(schema.columns)} columns, {row_count} rows"
        )

        return [
            types.TextContent(type="text", text=table_info.model_dump_json(indent=2))
        ]

    except Exception as e:
        raise Exception(f"Failed to describe table: {e}")


async def drop_table_handler(db, table_name: str) -> list[types.TextContent]:
    """Drop a table permanently."""
    try:
        db.drop_table(table_name)
        logger.info(f"Dropped table {table_name}")
        return [types.TextContent(type="text", text=f"Dropped table '{table_name}'")]
    except Exception as e:
        raise Exception(f"Failed to drop table: {e}")


async def insert_handler(db, data: InsertData) -> list[types.TextContent]:
    """Insert data with automatic embedding generation."""
    try:
        # Open the table
        table = db.open_table(data.table_name)

        # Get table metadata to understand schema
        metadata = await _get_table_metadata(table)
        if not metadata:
            raise Exception(f"Could not retrieve metadata for table {data.table_name}")

        schema = TableSchema.model_validate(metadata["schema"])
        searchable_columns = metadata.get("searchable_columns", [])

        # Create row data starting with user data
        row_data = dict(data.data)

        # Generate embeddings for searchable text columns
        if searchable_columns:
            config = get_embedding_config(schema.embedding_model)

            for col_name in searchable_columns:
                if col_name in row_data and row_data[col_name]:
                    text_content = str(row_data[col_name])
                    embedding = embedding_manager.embed_text(text_content, config)
                    row_data[f"{col_name}_vector"] = embedding

        # Add system timestamps
        now = datetime.now().isoformat()
        row_data["created_at"] = now
        row_data["updated_at"] = now

        # Insert into table
        df = pd.DataFrame([row_data])
        table.add(df)

        logger.info(f"Inserted row into table {data.table_name}")

        embedding_info = (
            f" (with {len(searchable_columns)} embeddings)"
            if searchable_columns
            else ""
        )
        return [
            types.TextContent(
                type="text",
                text=f"Inserted 1 row into table '{data.table_name}'{embedding_info}",
            )
        ]

    except Exception as e:
        raise Exception(f"Failed to insert data: {e}")


async def select_handler(db, query: QueryData) -> list[types.TextContent]:
    """Query data from table."""
    try:
        # Open the table
        table = db.open_table(query.table_name)

        # Start with full table scan
        df = table.to_pandas()

        # Filter out metadata rows
        df = df[df["table_metadata"].isna()]

        # Apply WHERE conditions if provided
        if query.where:
            for column, value in query.where.items():
                if column in df.columns:
                    df = df[df[column] == value]

        # Select specific columns if requested
        if query.columns:
            # Include system columns for potential ordering
            available_cols = [col for col in query.columns if col in df.columns]
            if query.order_by and query.order_by not in available_cols:
                available_cols.append(query.order_by)
            df = df[available_cols]
        else:
            # Exclude embedding vectors from default output for readability
            display_cols = [
                col
                for col in df.columns
                if not col.endswith("_vector") and col != "table_metadata"
            ]
            df = df[display_cols]

        # Apply ordering
        if query.order_by and query.order_by in df.columns:
            df = df.sort_values(query.order_by)

        # Apply limit
        if query.limit:
            df = df.head(query.limit)

        # Convert to records for JSON output
        results = df.to_dict(orient="records")

        logger.info(f"Selected {len(results)} rows from table {query.table_name}")

        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "table": query.table_name,
                        "results": results,
                        "row_count": len(results),
                    },
                    indent=2,
                ),
            )
        ]

    except Exception as e:
        raise Exception(f"Failed to select data: {e}")


async def update_handler(db, data: UpdateData) -> list[types.TextContent]:
    """Update data with automatic embedding updates."""
    try:
        # Open the table
        table = db.open_table(data.table_name)

        # Get table metadata to understand schema
        metadata = await _get_table_metadata(table)
        if not metadata:
            raise Exception(f"Could not retrieve metadata for table {data.table_name}")

        schema = TableSchema.model_validate(metadata["schema"])
        searchable_columns = metadata.get("searchable_columns", [])

        # Get current data
        df = table.to_pandas()

        # Filter out metadata rows
        df = df[df["table_metadata"].isna()]

        # Apply WHERE conditions to find rows to update
        if data.where:
            for column, value in data.where.items():
                if column in df.columns:
                    df = df[df[column] == value]

            if df.empty:
                return [
                    types.TextContent(
                        type="text",
                        text=f"No rows matched the WHERE conditions in table '{data.table_name}'",
                    )
                ]
        else:
            return [
                types.TextContent(
                    type="text",
                    text="WHERE conditions are required for UPDATE operations",
                )
            ]

        # Prepare updates with embedding regeneration
        updated_data = dict(data.data)

        # Generate embeddings for updated searchable text columns
        if searchable_columns:
            config = get_embedding_config(schema.embedding_model)

            for col_name in searchable_columns:
                if col_name in updated_data and updated_data[col_name]:
                    text_content = str(updated_data[col_name])
                    embedding = embedding_manager.embed_text(text_content, config)
                    updated_data[f"{col_name}_vector"] = embedding

        # Add update timestamp
        updated_data["updated_at"] = datetime.now().isoformat()

        # Update matching rows
        update_count = 0
        for index in df.index:
            for col, value in updated_data.items():
                df.at[index, col] = value
            update_count += 1

        # Since LanceDB doesn't support in-place updates, we need to recreate the table
        # This is a limitation we'll need to work around
        full_df = table.to_pandas()

        # Update the matching rows in the full dataframe
        for index in df.index:
            for col, value in updated_data.items():
                if col in full_df.columns:
                    full_df.at[index, col] = value

        # Recreate table with updated data
        table = db.create_table(name=data.table_name, data=full_df, mode="overwrite")

        logger.info(f"Updated {update_count} rows in table {data.table_name}")

        embedding_info = (
            f" (with {len([c for c in searchable_columns if c in data.data])} embedding updates)"
            if searchable_columns
            else ""
        )
        return [
            types.TextContent(
                type="text",
                text=f"Updated {update_count} rows in table '{data.table_name}'{embedding_info}",
            )
        ]

    except Exception as e:
        raise Exception(f"Failed to update data: {e}")


async def delete_handler(
    db, table_name: str, where: dict[str, Any]
) -> list[types.TextContent]:
    """Delete rows based on conditions."""
    try:
        # Open the table
        table = db.open_table(table_name)

        # Get current data
        df = table.to_pandas()

        # Filter out metadata rows
        user_data = df[df["table_metadata"].isna()]
        metadata_data = df[df["table_metadata"].notna()]

        # Apply WHERE conditions to find rows to delete
        if not where:
            return [
                types.TextContent(
                    type="text",
                    text="WHERE conditions are required for DELETE operations",
                )
            ]

        original_count = len(user_data)
        rows_to_keep = user_data.copy()

        # Apply filters to find rows that should be deleted
        for column, value in where.items():
            if column in rows_to_keep.columns:
                rows_to_keep = rows_to_keep[rows_to_keep[column] != value]

        deleted_count = original_count - len(rows_to_keep)

        if deleted_count == 0:
            return [
                types.TextContent(
                    type="text",
                    text=f"No rows matched the DELETE conditions in table '{table_name}'",
                )
            ]

        # Combine remaining user data with metadata
        final_data = pd.concat([rows_to_keep, metadata_data], ignore_index=True)

        # Recreate table with remaining data
        table = db.create_table(name=table_name, data=final_data, mode="overwrite")

        logger.info(f"Deleted {deleted_count} rows from table {table_name}")

        return [
            types.TextContent(
                type="text",
                text=f"Deleted {deleted_count} rows from table '{table_name}'",
            )
        ]

    except Exception as e:
        raise Exception(f"Failed to delete data: {e}")


async def search_handler(db, query: SearchQuery) -> list[types.TextContent]:
    """Semantic search across searchable columns."""
    try:
        # Open the table
        table = db.open_table(query.table_name)

        # Get table metadata to understand searchable columns
        metadata = await _get_table_metadata(table)
        if not metadata:
            raise Exception(f"Could not retrieve metadata for table {query.table_name}")

        schema = TableSchema.model_validate(metadata["schema"])
        searchable_columns = metadata.get("searchable_columns", [])

        if not searchable_columns:
            return [
                types.TextContent(
                    type="text",
                    text=f"No searchable columns found in table '{query.table_name}'. Add searchable=True to text columns when creating the table.",
                )
            ]

        # Generate query embedding
        config = get_embedding_config(schema.embedding_model)
        query_embedding = embedding_manager.embed_text(query.query, config)

        # Perform vector search on each searchable column and combine results
        all_results = []

        for col_name in searchable_columns:
            vector_col = f"{col_name}_vector"
            try:
                # Search using the embedding vector for this column
                search_results = table.search(
                    query_embedding, vector_column_name=vector_col
                ).limit(query.limit)

                # Apply additional filters if provided
                if query.where:
                    for filter_col, filter_val in query.where.items():
                        search_results = search_results.where(
                            f"{filter_col} = '{filter_val}'"
                        )

                # Apply similarity threshold if provided
                if query.threshold:
                    search_results = search_results.where(
                        f"_distance <= {1.0 - query.threshold}"
                    )

                # Exclude metadata rows
                search_results = search_results.where("table_metadata IS NULL")

                # Convert to pandas for processing
                results_df = search_results.to_pandas()

                if not results_df.empty:
                    # Add column context and process results
                    for _, row in results_df.iterrows():
                        result_dict = row.to_dict()

                        # Clean up the result
                        if "_distance" in result_dict:
                            result_dict["similarity_score"] = 1.0 - float(
                                result_dict["_distance"]
                            )
                            result_dict["distance"] = float(result_dict["_distance"])
                            del result_dict["_distance"]

                        # Remove vector columns from display
                        result_dict = {
                            k: v
                            for k, v in result_dict.items()
                            if not k.endswith("_vector") and k != "table_metadata"
                        }

                        result_dict["_matched_column"] = col_name
                        all_results.append(result_dict)

            except Exception as e:
                logger.warning(f"Search failed for column {col_name}: {e}")
                continue

        # Sort by similarity score and limit results
        all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        all_results = all_results[: query.limit]

        # Select specific columns if requested
        if query.columns:
            filtered_results = []
            for result in all_results:
                filtered_result = {
                    col: result.get(col) for col in query.columns if col in result
                }
                filtered_result.update(
                    {
                        k: v
                        for k, v in result.items()
                        if k in ["similarity_score", "distance", "_matched_column"]
                    }
                )
                filtered_results.append(filtered_result)
            all_results = filtered_results

        logger.info(
            f"Semantic search in table {query.table_name} returned {len(all_results)} results"
        )

        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "query": query.query,
                        "table": query.table_name,
                        "results": all_results,
                        "result_count": len(all_results),
                        "searchable_columns": searchable_columns,
                    },
                    indent=2,
                ),
            )
        ]

    except Exception as e:
        raise Exception(f"Failed to perform semantic search: {e}")


async def migrate_table_handler(
    db, migration: MigrationPlan
) -> list[types.TextContent]:
    """Safely migrate table to new schema."""
    try:
        source_table_name = migration.source_table
        target_schema = migration.target_schema
        backup_name = (
            migration.backup_name
            or f"{source_table_name}_backup_{int(datetime.now().timestamp())}"
        )

        # Step 1: Validate source table exists
        if source_table_name not in db.table_names():
            raise Exception(f"Source table '{source_table_name}' does not exist")

        # Step 2: Open source table and get its data
        source_table = db.open_table(source_table_name)
        source_df = source_table.to_pandas()

        # Separate user data from metadata
        source_user_data = source_df[source_df["table_metadata"].isna()]

        logger.info(f"Source table has {len(source_user_data)} user rows")

        # Step 3: Create backup table
        logger.info(f"Creating backup table: {backup_name}")
        db.create_table(name=backup_name, data=source_df, mode="create")

        # Step 4: Validate new schema and prepare transformation
        new_searchable_columns = get_searchable_columns(target_schema)
        column_mappings = migration.column_mappings or {}

        # Step 5: Transform data according to the new schema
        transformed_data = []

        for _, row in source_user_data.iterrows():
            new_row = {}

            # Apply column mappings and transformations
            for target_col in target_schema.columns:
                col_name = target_col.name

                # Check if this column has a mapping from old schema
                source_col = column_mappings.get(col_name, col_name)

                if source_col in row.index and pd.notna(row[source_col]):
                    # Copy existing data with type conversion if needed
                    value = row[source_col]

                    # Basic type conversion
                    if target_col.type == "text":
                        new_row[col_name] = str(value)
                    elif target_col.type == "integer":
                        new_row[col_name] = int(float(value)) if pd.notna(value) else 0
                    elif target_col.type == "float":
                        new_row[col_name] = float(value) if pd.notna(value) else 0.0
                    elif target_col.type == "boolean":
                        new_row[col_name] = bool(value) if pd.notna(value) else False
                    elif target_col.type == "json":
                        if isinstance(value, dict | list):
                            new_row[col_name] = value
                        else:
                            try:
                                new_row[col_name] = json.loads(str(value))
                            except Exception:
                                new_row[col_name] = {}
                    else:
                        new_row[col_name] = value
                else:
                    # Set default values for missing columns
                    if target_col.required:
                        if target_col.type == "text":
                            new_row[col_name] = ""
                        elif target_col.type == "integer":
                            new_row[col_name] = 0
                        elif target_col.type == "float":
                            new_row[col_name] = 0.0
                        elif target_col.type == "boolean":
                            new_row[col_name] = False
                        elif target_col.type == "json":
                            new_row[col_name] = {}
                    else:
                        # Use default values for optional columns
                        if target_col.type == "text":
                            new_row[col_name] = ""
                        elif target_col.type in ["integer", "float", "boolean"]:
                            new_row[col_name] = None
                        elif target_col.type == "json":
                            new_row[col_name] = {}

            # Copy system timestamps if they exist
            if "created_at" in row.index:
                new_row["created_at"] = row["created_at"]
            new_row["updated_at"] = datetime.now().isoformat()

            transformed_data.append(new_row)

        # Step 6: Create new table with target schema and transformed data
        logger.info(f"Creating new table with {len(transformed_data)} transformed rows")

        # Generate embeddings for searchable columns
        if new_searchable_columns and transformed_data:
            config = get_embedding_config(target_schema.embedding_model)

            for row in transformed_data:
                for col_name in new_searchable_columns:
                    if col_name in row and row[col_name]:
                        text_content = str(row[col_name])
                        embedding = embedding_manager.embed_text(text_content, config)
                        row[f"{col_name}_vector"] = embedding

        # Create the target table
        target_model = create_table_model(
            target_schema, config.dimension if new_searchable_columns else 1024
        )

        if target_model is dict:
            raise ImportError("LanceDB not available")

        # Create new table with transformed data
        new_table = db.create_table(
            name=target_schema.name, schema=target_model, mode="overwrite"
        )

        # Add metadata row
        metadata = {
            "schema": target_schema.model_dump(),
            "searchable_columns": new_searchable_columns,
            "embedding_model": target_schema.embedding_model,
            "created_at": datetime.now().isoformat(),
            "migrated_from": source_table_name,
            "backup_table": backup_name,
        }

        if transformed_data:
            # Add all data including metadata
            metadata_row = {col.name: None for col in target_schema.columns}
            metadata_row["created_at"] = datetime.now().isoformat()
            metadata_row["table_metadata"] = json.dumps(metadata)

            all_data = transformed_data + [metadata_row]
            df = pd.DataFrame(all_data)
            new_table.add(df)
        else:
            # Just add metadata if no user data
            metadata_row = {col.name: None for col in target_schema.columns}
            metadata_row["created_at"] = datetime.now().isoformat()
            metadata_row["table_metadata"] = json.dumps(metadata)

            df = pd.DataFrame([metadata_row])
            new_table.add(df)

        # Step 7: Drop original table if migration succeeded
        if target_schema.name != source_table_name:
            db.drop_table(source_table_name)
            logger.info(f"Dropped original table: {source_table_name}")

        logger.info(
            f"Migration completed successfully: {source_table_name} -> {target_schema.name}"
        )

        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "status": "success",
                        "source_table": source_table_name,
                        "target_table": target_schema.name,
                        "backup_table": backup_name,
                        "rows_migrated": len(transformed_data),
                        "new_searchable_columns": new_searchable_columns,
                        "embedding_model": target_schema.embedding_model,
                    },
                    indent=2,
                ),
            )
        ]

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        # Attempt cleanup if something went wrong
        try:
            if backup_name and backup_name in db.table_names():
                logger.info(f"Backup table {backup_name} is available for recovery")
        except Exception:
            pass
        raise Exception(f"Failed to migrate table: {e}")


async def run():
    """Run the server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="aas-lancedb-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
