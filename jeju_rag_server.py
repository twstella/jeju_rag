from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-1.7B"
# pipe = pipeline("text-generation", model=model_name, tokenizer=model_name, max_new_tokens=512)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="bfloat16", device_map="auto"
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
    full_prompt = f"[reference]: {ref} [query]: {query} [answer]:"
    messages = [{"role":"system","content": "You are a helpful agent. Answer in Korean."},{"role": "user", "content": full_prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
    )
    output = model.generate(
        input_ids.to(model.device), max_new_tokens=512, do_sample=False
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    mcp.run(transport="stdio")
