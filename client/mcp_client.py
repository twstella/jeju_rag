import asyncio
import re
import json
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        model_name = "Qwen/Qwen3-1.7B"
        # self.generator = pipeline("text-generation", model=model_name, tokenizer=model_name, max_new_tokens=512)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="bfloat16", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def connect_to_server(self, server_script_path: str):
        command = "python"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.read, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )
        await self.session.initialize()
        print(f"✅ Connected to server: {server_script_path}")

    async def process_query(self, query: str) -> str:
        tool_list = await self.session.list_tools()
        try:
            result = await self.session.call_tool("get_jeju_information",{"query":query})
            return result.content[0].text.strip()
        except Exception as e:
            return f"error occured: {e}"
        
    async def cleanup(self):
        await self.exit_stack.aclose()
