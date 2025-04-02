import json
from typing import Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool


def get_bitcoin_price(currency: str) -> dict:
    if currency == "USD":
        return {"price": 83829, "currency": "USD"}
    elif currency == "CNY":
        return {"price": 83829 * 7, "currency": "CNY"}

    return {"error": "Unsupported currency"}


async def bitcoin_price_mcp_server() -> None:
    server = Server("mcp-bitcoin-price")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available time tools."""
        return [
            Tool(
                name="get_bitcoin_price",
                description="Get current price of Bitcoin",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "currency": {
                            "type": "string",
                            "description": "currency (e.g., 'USD', 'CNY').",
                        }
                    },
                    "required": ["currency"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Handle tool calls for queries."""
        try:
            match name:
                case "get_bitcoin_price":
                    currency = arguments.get("currency", None)
                    if not currency:
                        raise ValueError("Missing required argument: currency")
                    result = get_bitcoin_price(currency)
                case _:
                    raise ValueError(f"Unknown tool: {name}")
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            raise ValueError(f"Error processing query: {str(e)}")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)


def main():
    import asyncio

    asyncio.run(bitcoin_price_mcp_server())


if __name__ == "__main__":
    main()
