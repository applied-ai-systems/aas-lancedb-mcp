# AAS LanceDB MCP Server

A comprehensive **Model Context Protocol (MCP) server** that provides AI agents with database-like operations over LanceDB with automatic embedding generation using state-of-the-art **BGE-M3 multilingual embeddings**.

## ✨ Why This MCP Server?

- **🎯 Database-like Interface**: Works like SQLite MCP - create tables, CRUD operations, migrations
- **🤖 Automatic Embeddings**: BGE-M3 generates 1024D multilingual embeddings for searchable text  
- **🔍 Semantic Search**: Natural language search across your data using vector similarity
- **📊 Rich Resources**: Dynamic database inspection (schemas, samples, statistics)
- **💡 Intelligent Prompts**: AI guidance for schema design, optimization, troubleshooting
- **🛡️ Safe Migrations**: Built-in table migration with validation and automatic backups
- **🌍 Multilingual**: BGE-M3 provides excellent performance across 100+ languages

## 🚀 Quick Start

### Install & Run with uvx (Recommended)

```bash
# Run directly without installation
uvx aas-lancedb-mcp --help

# Or install globally
uv tool install aas-lancedb-mcp
aas-lancedb-mcp --version
```

### Install from Source

```bash
git clone https://github.com/applied-ai-systems/aas-lancedb-mcp.git
cd aas-lancedb-mcp
uv tool install .
```

## 🛠️ MCP Capabilities Overview

### 🔧 **10 Database Tools**

| Tool | Purpose | Example |
|------|---------|---------|
| `create_table` | Create tables with schema | Create products table with searchable descriptions |
| `list_tables` | Show all tables | Get overview of database contents |
| `describe_table` | Get table schema & info | Understand table structure and metadata |
| `drop_table` | Delete tables | Remove unused tables |
| `insert` | Add data (auto-embeddings) | Insert product with searchable description |
| `select` | Query with filtering/sorting | Find products by price range |
| `update` | Modify data (auto-embeddings) | Update product info with new description |
| `delete` | Remove rows by conditions | Delete discontinued products |
| `search` | Semantic text search | "Find sustainable products" → matches related items |
| `migrate_table` | Safe schema changes | Add columns or change structure safely |

### 📁 **Dynamic Resources**

Resources provide AI agents with real-time database insights:

- **`lancedb://overview`** - Complete database statistics and table summary
- **`lancedb://tables/{name}/schema`** - Table schema, columns, searchable fields
- **`lancedb://tables/{name}/sample`** - Sample data for understanding contents  
- **`lancedb://tables/{name}/stats`** - Column statistics, data quality metrics

### 💬 **5 Intelligent Prompts**

AI-powered guidance for database operations:

- **`analyze_table`** - Generate insights about data patterns and quality
- **`design_schema`** - Help design optimal table schemas for use cases
- **`optimize_queries`** - Performance optimization recommendations
- **`troubleshoot_performance`** - Diagnose and solve database issues
- **`migration_planning`** - Plan safe schema migrations step-by-step

## 📋 Usage Examples

### Creating a Product Catalog

```json
{
  "tool": "create_table",
  "arguments": {
    "schema": {
      "name": "products", 
      "columns": [
        {"name": "title", "type": "text", "required": true, "searchable": true},
        {"name": "description", "type": "text", "searchable": true},
        {"name": "price", "type": "float", "required": true},
        {"name": "category", "type": "text", "required": true},
        {"name": "metadata", "type": "json"}
      ],
      "description": "E-commerce product catalog with semantic search"
    }
  }
}
```

### Adding Products (Embeddings Generated Automatically)

```json
{
  "tool": "insert", 
  "arguments": {
    "data": {
      "table_name": "products",
      "data": {
        "title": "Eco-Friendly Water Bottle", 
        "description": "Sustainable stainless steel water bottle with insulation",
        "price": 24.99,
        "category": "sustainability",
        "metadata": {"brand": "EcoLife", "material": "stainless_steel"}
      }
    }
  }
}
```

### Semantic Search (Natural Language)

```json
{
  "tool": "search",
  "arguments": {
    "query": {
      "table_name": "products",
      "query": "environmentally friendly drinking containers",
      "limit": 5
    }
  }
}
```

### Database Inspection (Resources)

```json
{
  "resource": "lancedb://tables/products/sample"
}
```

Returns sample product data for AI agents to understand the table structure.

### AI Guidance (Prompts)

```json
{
  "prompt": "design_schema",
  "arguments": {
    "use_case": "Customer support ticket system",
    "data_types": "ticket text, priority levels, timestamps", 
    "search_requirements": "semantic search across ticket descriptions"
  }
}
```

Returns AI-generated recommendations for optimal table design.

## ⚙️ Configuration

### Claude Desktop Setup

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "aas-lancedb": {
      "command": "aas-lancedb-mcp",
      "args": ["--db-uri", "~/my_database"],
      "env": {
        "EMBEDDING_MODEL": "BAAI/bge-m3"
      }
    }
  }
}
```

### Environment Variables

```bash
export LANCEDB_URI="./my_database"      # Database location
export EMBEDDING_MODEL="BAAI/bge-m3"    # Embedding model (default)
export EMBEDDING_DEVICE="cpu"           # cpu or cuda
```

### Command Line Options

```bash
aas-lancedb-mcp --help                   # Show help
aas-lancedb-mcp --version                # Show version  
aas-lancedb-mcp --db-uri ./my_db         # Custom database path
```

## 🏗️ Architecture

```
Enhanced MCP Server Architecture
├── 🔧 Tools (10)           - Database operations (CRUD, search, migrate)
├── 📁 Resources (dynamic)   - Real-time database introspection  
├── 💬 Prompts (5)          - AI guidance for database tasks
├── 🤖 BGE-M3 Embeddings    - Automatic 1024D multilingual vectors
├── 🛡️ Safe Migrations      - Schema evolution with validation
└── 📊 Rich Metadata        - Column types, constraints, statistics
```

### Key Technical Features

- **🎯 Database-like Interface**: Familiar SQL-style operations hiding vector complexity
- **🤖 Automatic Embedding Generation**: BGE-M3 creates vectors for searchable text columns  
- **🔍 Hybrid Search**: Combine semantic similarity with traditional filtering
- **📊 Dynamic Resources**: Real-time database inspection for AI agents
- **💡 Contextual Prompts**: AI guidance using actual database state
- **🛡️ Migration Safety**: Backup, validate, and rollback capabilities
- **🌍 Multilingual**: BGE-M3 excels across 100+ languages

## 🧪 Development & Testing

```bash
# Clone and setup
git clone https://github.com/applied-ai-systems/aas-lancedb-mcp.git
cd aas-lancedb-mcp

# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run tests with coverage  
uv run pytest --cov=src --cov-report=term-missing

# Format and lint
uv run ruff format .
uv run ruff check .

# Test CLI
uv run aas-lancedb-mcp --help
```

## 🚀 Performance & Scalability

- **BGE-M3 Embeddings**: 1024 dimensions, excellent multilingual performance
- **LanceDB Backend**: Columnar vector database optimized for scale
- **Efficient Operations**: Automatic embedding caching and batch processing
- **Memory Management**: Lazy loading and streaming for large datasets
- **Search Performance**: HNSW indexing for fast vector similarity search

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests (`pytest tests/`)
4. Format code (`uv run ruff format .`)
5. Submit Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[LanceDB](https://lancedb.com/)** - High-performance columnar vector database
- **[BGE-M3](https://huggingface.co/BAAI/bge-m3)** - State-of-the-art multilingual embeddings  
- **[Model Context Protocol](https://modelcontextprotocol.io/)** - Standardized AI tool integration
- **[Sentence Transformers](https://sbert.net/)** - Easy-to-use embedding framework

## 📚 Related MCP Projects

- **[MCP Servers](https://github.com/modelcontextprotocol/servers)** - Official MCP server collection
- **[FastMCP](https://github.com/jlowin/fastmcp)** - Fast Pythonic MCP framework  
- **[SQLite MCP](https://github.com/modelcontextprotocol/servers/tree/main/src/sqlite)** - Database MCP inspiration

---

**Ready to supercharge your AI agents with powerful database capabilities?** 🚀

```bash
uvx aas-lancedb-mcp --help
```