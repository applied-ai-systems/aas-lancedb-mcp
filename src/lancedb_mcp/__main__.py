"""Main entry point for AAS LanceDB MCP server."""

import argparse
import asyncio
import sys

from .server import run, set_db_uri


def main():
    """Run the AAS LanceDB MCP server."""
    parser = argparse.ArgumentParser(
        description="Enhanced LanceDB MCP Server with BGE-M3 embeddings"
    )
    parser.add_argument(
        "--db-uri",
        default=".aas_lancedb",
        help="Database URI/path (default: .aas_lancedb)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    # Set database URI if provided
    if args.db_uri:
        set_db_uri(args.db_uri)

    print("🚀 Starting AAS LanceDB MCP Server...")
    print(f"📁 Database: {args.db_uri}")
    print("🤖 Embedding Model: BAAI/bge-m3 (1024 dimensions)")
    print("🔧 Features: Tools (10), Resources (dynamic), Prompts (5)")
    print("📡 Protocol: Model Context Protocol (MCP)")
    print("⚡ Ready for AI agent connections!")
    print()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
