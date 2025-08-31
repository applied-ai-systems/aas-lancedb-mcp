"""Enhanced AAS LanceDB MCP server with sentence transformers integration."""

import logging
import os
import pathlib
from datetime import datetime
from typing import Any, List, Dict, Optional
import json

import lancedb
import mcp.server.stdio
import mcp.types as types
import pandas as pd
from lancedb.pydantic import pydantic_to_schema
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .models import (
    SearchQuery, TableConfig, VectorData, TextData, 
    DatastoreInfo, EmbeddingConfig
)
from .embedding import embedding_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database URI and embedding config
DB_URI = os.getenv("LANCEDB_URI", ".aas_lancedb")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


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


def get_embedding_config(model_name: Optional[str] = None) -> EmbeddingConfig:
    """Get embedding configuration."""
    return EmbeddingConfig(
        model_name=model_name or DEFAULT_EMBEDDING_MODEL,
        device=os.getenv("EMBEDDING_DEVICE", "cpu")
    )


# Create MCP server instance
server = Server("aas-lancedb-server")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="create_table",
            description="Create a new vector table with optional embedding model configuration",
            arguments=[
                types.ToolArgument(
                    name="config",
                    description="Table configuration",
                    schema=TableConfig.model_json_schema(),
                )
            ],
        ),
        types.Tool(
            name="add_text",
            description="Add text data with automatic embedding generation",
            arguments=[
                types.ToolArgument(
                    name="table_name", 
                    description="Name of the table", 
                    type="string"
                ),
                types.ToolArgument(
                    name="data",
                    description="Text data to embed and store",
                    schema=TextData.model_json_schema(),
                ),
            ],
        ),
        types.Tool(
            name="add_vector",
            description="Add pre-computed vector data to a table",
            arguments=[
                types.ToolArgument(
                    name="table_name", 
                    description="Name of the table", 
                    type="string"
                ),
                types.ToolArgument(
                    name="data",
                    description="Vector data",
                    schema=VectorData.model_json_schema(),
                ),
            ],
        ),
        types.Tool(
            name="search_semantic",
            description="Perform semantic search using text query with automatic embedding",
            arguments=[
                types.ToolArgument(
                    name="table_name", 
                    description="Name of the table", 
                    type="string"
                ),
                types.ToolArgument(
                    name="query",
                    description="Search query",
                    schema=SearchQuery.model_json_schema(),
                ),
            ],
        ),
        types.Tool(
            name="search_vectors",
            description="Search vectors using pre-computed query vector",
            arguments=[
                types.ToolArgument(
                    name="table_name", 
                    description="Name of the table", 
                    type="string"
                ),
                types.ToolArgument(
                    name="query",
                    description="Search query with vector",
                    schema=SearchQuery.model_json_schema(),
                ),
            ],
        ),
        types.Tool(
            name="list_tables",
            description="List all available tables/datastores",
            arguments=[],
        ),
        types.Tool(
            name="get_table_info",
            description="Get detailed information about a table",
            arguments=[
                types.ToolArgument(
                    name="table_name", 
                    description="Name of the table", 
                    type="string"
                ),
            ],
        ),
        types.Tool(
            name="delete_table",
            description="Delete a table/datastore",
            arguments=[
                types.ToolArgument(
                    name="table_name", 
                    description="Name of the table to delete", 
                    type="string"
                ),
            ],
        ),
        types.Tool(
            name="get_model_info",
            description="Get information about available embedding models",
            arguments=[
                types.ToolArgument(
                    name="model_name", 
                    description="Name of the model (optional)", 
                    type="string",
                    required=False
                ),
            ],
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Handle tool calls."""
    try:
        db = get_db()

        if name == "create_table":
            config = TableConfig.model_validate(arguments["config"])
            
            # Create VectorData schema with proper dimension
            class DynamicVectorData(VectorData):
                vector: lancedb.pydantic.Vector = lancedb.pydantic.Field(..., dim=config.dimension)
            
            schema = pydantic_to_schema(DynamicVectorData)
            
            # Store table metadata
            metadata = {
                "embedding_model": config.embedding_model,
                "dimension": config.dimension,
                "metric": config.metric,
                "description": config.description,
                "created_at": datetime.now().isoformat(),
            }
            
            table = db.create_table(
                name=config.name,
                schema=schema,
                mode="overwrite",
            )
            
            # Store metadata as table properties if possible
            try:
                # Add metadata as first row with special marker
                metadata_row = pd.DataFrame([{
                    "vector": [0.0] * config.dimension,
                    "text": f"__METADATA__{json.dumps(metadata)}",
                    "uri": None,
                    "metadata": metadata,
                    "source": "__SYSTEM__",
                    "timestamp": datetime.now().isoformat()
                }])
                table.add(metadata_row)
            except Exception as e:
                logger.warning(f"Could not store table metadata: {e}")
            
            logger.info(f"Created table {config.name} with dimension {config.dimension}")
            return [types.TextContent(
                type="text", 
                text=f"Created table '{config.name}' with {config.dimension}D vectors using model '{config.embedding_model}'"
            )]

        elif name == "add_text":
            table_name = arguments["table_name"]
            text_data = TextData.model_validate(arguments["data"])
            
            # Get table to determine embedding model
            table = db.open_table(table_name)
            
            # Try to get embedding model from table metadata
            embedding_model = DEFAULT_EMBEDDING_MODEL
            try:
                # Look for metadata row
                metadata_rows = table.search([0.0] * 384).where("source = '__SYSTEM__'").limit(1).to_pandas()
                if not metadata_rows.empty and metadata_rows.iloc[0]["text"].startswith("__METADATA__"):
                    metadata_json = metadata_rows.iloc[0]["text"].replace("__METADATA__", "")
                    metadata = json.loads(metadata_json)
                    embedding_model = metadata.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            except Exception as e:
                logger.warning(f"Could not retrieve table metadata, using default model: {e}")
            
            # Generate embedding
            config = get_embedding_config(embedding_model)
            embedding = embedding_manager.embed_text(text_data.text, config)
            
            # Create vector data
            vector_data = VectorData(
                vector=embedding,
                text=text_data.text,
                uri=text_data.uri,
                metadata=text_data.metadata or {},
                source=text_data.source,
                timestamp=datetime.now().isoformat()
            )
            
            # Add to table
            df = pd.DataFrame([vector_data.model_dump()])
            table.add(df)
            
            logger.info(f"Added text data to table {table_name}")
            return [types.TextContent(
                type="text", 
                text=f"Added text data to table {table_name} with auto-generated embedding"
            )]

        elif name == "add_vector":
            table_name = arguments["table_name"]
            data = VectorData.model_validate(arguments["data"])
            table = db.open_table(table_name)
            
            # Add timestamp if not provided
            if not data.timestamp:
                data.timestamp = datetime.now().isoformat()
            
            df = pd.DataFrame([data.model_dump()])
            table.add(df)
            
            logger.info(f"Added vector to table {table_name}")
            return [types.TextContent(
                type="text", 
                text=f"Added vector to table {table_name}"
            )]

        elif name == "search_semantic":
            table_name = arguments["table_name"]
            query = SearchQuery.model_validate(arguments["query"])
            
            if not query.text:
                raise ValueError("Text query is required for semantic search")
            
            table = db.open_table(table_name)
            
            # Get embedding model from table metadata
            embedding_model = DEFAULT_EMBEDDING_MODEL
            try:
                metadata_rows = table.search([0.0] * 384).where("source = '__SYSTEM__'").limit(1).to_pandas()
                if not metadata_rows.empty and metadata_rows.iloc[0]["text"].startswith("__METADATA__"):
                    metadata_json = metadata_rows.iloc[0]["text"].replace("__METADATA__", "")
                    metadata = json.loads(metadata_json)
                    embedding_model = metadata.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            except Exception:
                pass
            
            # Generate query embedding
            config = get_embedding_config(embedding_model)
            query_vector = embedding_manager.embed_text(query.text, config)
            
            # Build search
            search = table.search(query_vector).limit(query.limit)
            
            # Apply filters
            if query.filter_expr:
                search = search.where(query.filter_expr)
            if query.source_filter:
                search = search.where(f"source = '{query.source_filter}'")
            
            # Exclude metadata rows
            search = search.where("source != '__SYSTEM__'")
            
            results = search.to_pandas()
            
            # Process results
            results_dict = results.to_dict(orient="records")
            for result in results_dict:
                if "_distance" in result:
                    result["similarity_score"] = 1.0 - float(result["_distance"])  # Convert distance to similarity
                    result["distance"] = float(result["_distance"])
                    del result["_distance"]
                if "vector" in result:
                    # Optionally include vector in results (can be large)
                    if len(results_dict) <= 5:  # Only include vectors for small result sets
                        result["vector"] = result["vector"].tolist()
                    else:
                        del result["vector"]  # Remove vector for large result sets
            
            logger.info(f"Semantic search in table {table_name} returned {len(results_dict)} results")
            return [types.TextContent(
                type="text", 
                text=json.dumps({
                    "query": query.text,
                    "results": results_dict,
                    "total_results": len(results_dict)
                }, indent=2)
            )]

        elif name == "search_vectors":
            table_name = arguments["table_name"]
            query = SearchQuery.model_validate(arguments["query"])
            
            if not query.vector:
                raise ValueError("Vector is required for vector search")
            
            table = db.open_table(table_name)
            
            # Build search
            search = table.search(query.vector).limit(query.limit)
            
            # Apply filters
            if query.filter_expr:
                search = search.where(query.filter_expr)
            if query.source_filter:
                search = search.where(f"source = '{query.source_filter}'")
            
            # Exclude metadata rows
            search = search.where("source != '__SYSTEM__'")
            
            results = search.to_pandas()
            
            # Process results
            results_dict = results.to_dict(orient="records")
            for result in results_dict:
                if "_distance" in result:
                    result["similarity_score"] = 1.0 - float(result["_distance"])
                    result["distance"] = float(result["_distance"])
                    del result["_distance"]
                if "vector" in result:
                    del result["vector"]  # Remove vector from results
            
            logger.info(f"Vector search in table {table_name} returned {len(results_dict)} results")
            return [types.TextContent(
                type="text", 
                text=json.dumps({
                    "results": results_dict,
                    "total_results": len(results_dict)
                }, indent=2)
            )]

        elif name == "list_tables":
            table_names = db.table_names()
            
            tables_info = []
            for table_name in table_names:
                try:
                    table = db.open_table(table_name)
                    count = table.count_rows()
                    tables_info.append({
                        "name": table_name,
                        "row_count": count
                    })
                except Exception as e:
                    tables_info.append({
                        "name": table_name,
                        "row_count": "Error",
                        "error": str(e)
                    })
            
            return [types.TextContent(
                type="text", 
                text=json.dumps({
                    "tables": tables_info,
                    "total_tables": len(table_names)
                }, indent=2)
            )]

        elif name == "get_table_info":
            table_name = arguments["table_name"]
            table = db.open_table(table_name)
            
            # Get basic info
            row_count = table.count_rows()
            schema = table.schema
            
            # Try to get metadata
            metadata = {}
            try:
                metadata_rows = table.search([0.0] * 384).where("source = '__SYSTEM__'").limit(1).to_pandas()
                if not metadata_rows.empty and metadata_rows.iloc[0]["text"].startswith("__METADATA__"):
                    metadata_json = metadata_rows.iloc[0]["text"].replace("__METADATA__", "")
                    metadata = json.loads(metadata_json)
                    row_count -= 1  # Subtract metadata row
            except Exception:
                pass
            
            info = DatastoreInfo(
                name=table_name,
                row_count=row_count,
                schema={"fields": [{"name": f.name, "type": str(f.type)} for f in schema]},
                dimension=metadata.get("dimension", "unknown"),
                embedding_model=metadata.get("embedding_model"),
                created_at=metadata.get("created_at"),
                description=metadata.get("description")
            )
            
            return [types.TextContent(
                type="text", 
                text=info.model_dump_json(indent=2)
            )]

        elif name == "delete_table":
            table_name = arguments["table_name"]
            db.drop_table(table_name)
            
            logger.info(f"Deleted table {table_name}")
            return [types.TextContent(
                type="text", 
                text=f"Deleted table '{table_name}'"
            )]

        elif name == "get_model_info":
            model_name = arguments.get("model_name", DEFAULT_EMBEDDING_MODEL)
            info = embedding_manager.get_model_info(model_name)
            
            return [types.TextContent(
                type="text", 
                text=json.dumps(info, indent=2)
            )]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except FileNotFoundError as e:
        error_msg = f"Table {arguments.get('table_name', 'unknown')} not found"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]
    except Exception as err:
        error_msg = f"Failed to execute tool {name}: {err}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=f"Error: {error_msg}")]


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