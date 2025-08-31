"""Main entry point for AAS LanceDB MCP server."""

import asyncio
import sys

from .server import run


def main():
    """Run the AAS LanceDB MCP server."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
