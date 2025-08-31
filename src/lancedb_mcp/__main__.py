"""Main entry point for AAS LanceDB MCP server."""

import sys
import asyncio
from .server import run


def main():
    """Main entry point."""
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
