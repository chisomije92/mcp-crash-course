import asyncio
from pathlib import Path
from dotenv import load_dotenv
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()

# get path to server directory and file


llm = ChatOpenAI(model="gpt-5.1")


BASE_DIR = Path(__file__).resolve().parent

stdio_server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(BASE_DIR / "servers/math_server.py")],
    cwd=str(BASE_DIR),
)


async def main():
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("session initialized")
            tools = await load_mcp_tools(session)
            # print(tools)
            agent = create_agent(llm, tools)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 54 + 2 * 3?")]}
            )
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
