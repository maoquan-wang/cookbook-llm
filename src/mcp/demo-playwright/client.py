"""
# install
npm install -g @executeautomation/playwright-mcp-server

# start the server
npx -y @executeautomation/playwright-mcp-server
"""

import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from mcp.client.stdio import stdio_client

from mcp import ClientSession, StdioServerParameters


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self):
        server_params = StdioServerParameters(
            command="npx", args=["-y", "@executeautomation/playwright-mcp-server"], env=None
        )
        print("========")
        print(server_params)
        print("========")
        stdio, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        for tool in tools:
            print(tool)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    # python client.py
    asyncio.run(main())
