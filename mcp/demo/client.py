import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from mcp.client.stdio import stdio_client

from mcp import ClientSession, StdioServerParameters


class MCPClient:
    def __init__(self, server_script_path: str):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.server_script_path = server_script_path

    async def connect_to_server(self):
        server_params = StdioServerParameters(
            command="python", args=[self.server_script_path], env=None
        )
        stdio, write = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def test_tool(self, currency: str):
        result = await self.session.call_tool(
            "get_bitcoin_price", {"currency": currency}
        )
        print("Tool call result: \n")
        print(result.content[0].text)
        return result

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                else:
                    await self.test_tool(query)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    client = MCPClient("./my_server.py")
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    # python client.py
    asyncio.run(main())
