# AAS LanceDB MCP Server

Enhanced LanceDB MCP (Model Context Protocol) server for arbitrary datastore management with sentence transformers integration.

## 🚀 Features

- **Sentence Transformers Integration**: Automatic text embedding with configurable models
- **Multi-Datastore Management**: Create and manage multiple vector tables/datastores  
- **Semantic Search**: Natural language queries with automatic embedding generation
- **Flexible Vector Operations**: Support for both text and pre-computed vector data
- **Rich Metadata**: Store additional context with vectors (source, timestamps, custom metadata)
- **Advanced Filtering**: SQL-like filtering on search results
- **Model Management**: Support for different embedding models per table
- **Production Ready**: Professional Python packaging with comprehensive tests

## 🛠️ Enhanced MCP Tools

### Core Operations
- `create_table` - Create vector tables with configurable embedding models
- `add_text` - Add text data with automatic embedding generation  
- `add_vector` - Add pre-computed vector data
- `search_semantic` - Semantic search using text queries
- `search_vectors` - Vector similarity search
- `list_tables` - List all available datastores
- `get_table_info` - Detailed table information and metadata
- `delete_table` - Remove tables/datastores
- `get_model_info` - Information about embedding models

## 📦 Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/your-org/aas-lancedb-mcp
cd aas-lancedb-mcp

# Install with UV (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Using UV Tool Install

```bash
# Install globally with uv
uv tool install aas-lancedb-mcp

# Run directly
aas-lancedb-mcp
```

## 🔧 Configuration

### Environment Variables

```bash
# Database location (default: .aas_lancedb)
export LANCEDB_URI="/path/to/your/database"

# Default embedding model (default: all-MiniLM-L6-v2)
export EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Device for embedding models (default: cpu)
export EMBEDDING_DEVICE="cuda"  # or "cpu"
```

### Claude Desktop Configuration

Add to your Claude Desktop configuration:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "aas-lancedb": {
      "command": "aas-lancedb-mcp",
      "env": {
        "LANCEDB_URI": "/path/to/your/database",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

## 💡 Usage Examples

### Creating a Table

```python
# Via MCP tool call
{
  "name": "create_table",
  "arguments": {
    "config": {
      "name": "documents", 
      "dimension": 384,
      "embedding_model": "all-MiniLM-L6-v2",
      "description": "Document embeddings for semantic search"
    }
  }
}
```

### Adding Text Data

```python
# Text is automatically embedded
{
  "name": "add_text",
  "arguments": {
    "table_name": "documents",
    "data": {
      "text": "The quick brown fox jumps over the lazy dog.",
      "source": "example_docs",
      "metadata": {"category": "test", "priority": "high"}
    }
  }
}
```

### Semantic Search

```python
# Natural language query
{
  "name": "search_semantic", 
  "arguments": {
    "table_name": "documents",
    "query": {
      "text": "animals jumping",
      "limit": 5,
      "source_filter": "example_docs"
    }
  }
}
```

## 🏗️ Architecture

```
AAS LanceDB MCP Server
├── models.py          # Pydantic data models
├── embedding.py       # Sentence transformers integration  
├── server.py          # MCP server implementation
└── __main__.py        # CLI entry point
```

### Key Components

- **EmbeddingManager**: Handles sentence transformer models and caching
- **VectorData/TextData**: Flexible data models for different input types
- **SearchQuery**: Advanced search with filtering and pagination
- **TableConfig**: Configurable table creation with embedding models

## 🧪 Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_server.py -v
```

## 🚀 Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Format code
black src tests

# Lint code  
ruff check src tests

# Type checking
mypy src
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on [RyanLisse/lancedb_mcp](https://github.com/RyanLisse/lancedb_mcp) - Clean Python Lance MCP implementation
- [LanceDB](https://lancedb.com/) - High-performance vector database
- [Sentence Transformers](https://huggingface.co/sentence-transformers) - State-of-the-art text embeddings
- [Model Context Protocol](https://modelcontextprotocol.io/) - Standardized AI tool integration

## 📚 Related Projects

- [IBM MCP Context Forge](https://github.com/IBM/mcp-context-forge) - MCP Gateway & Registry
- [DesktopCommanderMCP](https://github.com/wonderwhy-er/DesktopCommanderMCP) - Desktop automation MCP
- [FastMCP](https://github.com/jlowin/fastmcp) - Fast, Pythonic MCP framework