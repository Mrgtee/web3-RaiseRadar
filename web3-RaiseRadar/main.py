import os
import requests
import uuid
import json
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain & Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="Web3 RaiseRadar Agent")

# --- MANDATORY CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-vercel-ai-ui-message-stream"] # Crucial for some UI versions
)

# 2. Define Custom CryptoPanic Tool
@tool
def fetch_crypto_news(query: str) -> str:
    """
    Fetches trending crypto news and market sentiment from CryptoPanic.
    Use this when the user asks for news, trends, or sentiment about a specific coin.
    """
    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    params = {"auth_token": api_key, "public": "true", "kind": "news", "regions": "en", "filter": "hot"}

    if any(x in query.lower() for x in ["bitcoin", "btc"]): params["currencies"] = "BTC"
    elif any(x in query.lower() for x in ["ethereum", "eth"]): params["currencies"] = "ETH"

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])[:3]
        if not results: return "No recent hot news found."
        return "\n".join([f"- **{p['title']}**\n Source: {p['url']}" for p in results])
    except Exception as e:
        return f"CryptoPanic Error: {str(e)}"

# 3. Setup Agent
search_tool = TavilySearch(max_results=5, search_depth="advanced", include_answer=True)
tools = [search_tool, fetch_crypto_news]

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Use standard stable identifier
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

system_msg = (
    "You are the Web3 RaiseRadar Agent. "
    "1. For funding deep dives, ALWAYS use 'tavily_search_results_json'. "
    "2. For high-level news, use 'fetch_crypto_news'. "
    "3. Format responses in a Markdown Table: Project | Event | Date | Source/Link."
)

agent_app = create_react_agent(llm, tools, prompt=system_msg)

# 4. Models
class ThreadRequest(BaseModel):
    metadata: Optional[dict] = {}

# 5. Core Endpoints
@app.get("/info")
async def info():
    return {"assistant_id": "Web3-RaiseRadar", "graph_id": "Web3-RaiseRadar", "config": {}}

@app.post("/threads")
async def create_thread(request: ThreadRequest):
    return {"thread_id": str(uuid.uuid4()), "metadata": request.metadata}

@app.post("/threads/search")
async def threads_search(): return []

# CRITICAL: FIXED VERCEL DATA STREAM PROTOCOL (v1)
@app.post("/threads/{thread_id}/runs/stream")
async def runs_stream(thread_id: str, request: Request):
    body = await request.json()
    messages = body.get("input", {}).get("messages", [])
    user_input = messages[-1].get("content", "") if messages else ""

    async def event_generator():
        # Vercel Protocol expects chunks prefixed by type codes.
        # 0: Text Part
        # b: Message Annotations / Metadata
        # d: Data Part
        
        # Initialize the UI message bubble
        yield f'0:""\n' 

        async for chunk in agent_app.astream(
            {"messages": [("user", user_input)]},
            stream_mode="updates" # 'updates' catches node completion
        ):
            for node_name, data in chunk.items():
                if "messages" in data:
                    msg = data["messages"][-1]
                    # Only emit final AI content to the main text stream
                    if hasattr(msg, "content") and msg.content and msg.type == "ai":
                        # Format: 0:"the text"\n
                        # Using json.dumps handles the required escaping of newlines/quotes
                        content = json.dumps(msg.content)
                        yield f'0:{content}\n'

        # Signal completion with a finish reason
        yield 'd:{"finishReason":"stop"}\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1" # This tells the UI to use the v1 protocol
        }
    )

@app.post("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    # History must return an array of messages
    return [{"role": "assistant", "type": "ai", "content": "RaiseRadar online. How can I help?", "metadata": {}}]

@app.get("/.well-known/agent.json")
async def get_agent_manifest():
    return {
        "name": "Web3 RaiseRadar",
        "url": "https://web3-raiseradar-production-1f6f.up.railway.app",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
