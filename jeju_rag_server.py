from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import re

model_name = "Qwen/Qwen3-1.7B"
# pipe = pipeline("text-generation", model=model_name, tokenizer=model_name, max_new_tokens=512)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="bfloat16", device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

client = chromadb.PersistentClient(path="chroma_jeju")
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)
collection = client.get_collection(name="jeju_info", embedding_function=emb)

mcp = FastMCP("get jeju tourism information")


@mcp.tool()
async def get_jeju_information(query: str):
    results = collection.query(query_texts=[query], n_results=3)
    ref = " ".join(results["documents"][0])
    full_prompt = f"[reference]: {ref} [query]: {query}"
    messages = [{"role":"system","content": "You are a helpful agent. Answer in Korean."},{"role": "user", "content": full_prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
    )
    model_inputs = tokenizer([text],return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return response


if __name__ == "__main__":
    mcp.run(transport="stdio")
