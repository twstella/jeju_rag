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
        model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
        #self.generator = pipeline("text-generation", model=model_name, tokenizer=model_name, max_new_tokens=512)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16",device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def connect_to_server(self, server_script_path: str):
        command = "python"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.read, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.session.initialize()
        print(f"✅ Connected to server: {server_script_path}")

    async def process_query(self, query: str) -> str:
        tool_list = await self.session.list_tools()
        tool_descriptions = "\n".join(f"{tool.name}: {tool.description}" for tool in tool_list.tools)

        prompt = f"""당신은 AI 어시스턴트입니다. 사용할 수 있는 도구는 다음과 같습니다:
{tool_descriptions}

적절할 경우 다음 형식으로 응답하세요: TOOL_CALL: <tool_name> <json_parameters>
그 외에는 자연스럽게 대화형으로 응답하세요.

{query}
어시스턴트:"""

        messages = [{"role":"user","content":prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        output = self.model.generate(input_ids.to(self.model.device),max_new_tokens=512, do_sample=False)
        generated = self.tokenizer.decode(output[0])

        if generated.startswith("TOOL_CALL:"):
            try:
                match = re.match(r"TOOL_CALL:\s*(\w+)\s*(\{.*\})", generated, re.DOTALL)
                if not match:
                    return f"Wrong TOOL_CALL format: {generated}"
                tool_name, tool_args_json = match.groups()
                tool_args = json.loads(tool_args_json)
                result = await self.session.call_tool(tool_name, tool_args)
                return f"🛠 Tool `{tool_name}` called.\n\n📄 Result:\n{result.content}"
            except Exception as e:
                return f"❗ Error calling tool: {str(e)}"
        else:
            return generated

    async def cleanup(self):
        await self.exit_stack.aclose()
