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

# --- ADDED: CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define Custom CryptoPanic Tool
@tool
def fetch_crypto_news(query: str) -> str:
    print(f"DEBUG: Accessing CryptoPanic for: {query}")

    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    url = "https://cryptopanic.com/api/developer/v2/posts/"

    params = {
        "auth_token": api_key,
        "public": "true",
        "kind": "news",
        "regions": "en",
        "filter": "hot"
    }

    if "bitcoin" in query.lower() or "btc" in query.lower():
        params["currencies"] = "BTC"
    elif "ethereum" in query.lower() or "eth" in query.lower():
        params["currencies"] = "ETH"

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])[:3]

        if not results:
            return "No recent hot news found on CryptoPanic."

        news_list = [
            f"- **{p['title']}**\n  Source: {p['url']}"
            for p in results
        ]
        return "\n".join(news_list)
    except Exception as e:
        return f"CryptoPanic Error: {str(e)}"

# 3. Enhanced Search Tool
search_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True
)

# 4. Setup Tools and Gemini LLM
tools = [search_tool, fetch_crypto_news]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)

# 5. Strengthened System Prompt
system_msg = (
    "You are the Web3 RaiseRadar Agent. "
    "1. For funding research or project deep dives, ALWAYS use 'tavily_search_results_json'. "
    "2. For high-level trending headlines, use 'fetch_crypto_news'. "
    "3. IMPORTANT: When users ask about dates (ICOs, Sales, Auctions), cross-reference "
    "carefully. If a date changed recently (like Zama's), state the NEW date clearly. "
    "4. Format responses in a Markdown Table: Project | Event | Date | Source/Link."
)

agent_app = create_react_agent(llm, tools, prompt=system_msg)

# 6. API Models
class ChatQuery(BaseModel):
    message: str

class ThreadRequest(BaseModel):
    metadata: Optional[dict] = {}

class HistoryRequest(BaseModel):
    limit: Optional[int] = 10
    before: Optional[str] = None

# 7. Endpoints
@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    inputs = {"messages": [("user", query.message)]}
    result = await agent_app.ainvoke(inputs)
    final_content = result["messages"][-1].content
    return {
        "response": [
            {
                "type": "text",
                "text": final_content
            }
        ]
    }

@app.get("/")
async def health_check():
    return {"status": "active", "agent": "RaiseRadar-v1"}

@app.get("/info")
async def info():
    return {
        "assistant_id": "raiseradar",
        "graph_id": "agent",
        "config": {},
    }

@app.post("/threads/search")
async def threads_search():
    return []

@app.post("/assistants/search")
async def assistants_search():
    return [
        {
            "assistant_id": "raiseradar",
            "name": "RaiseRadar Agent"
        }
    ]

@app.post("/threads")
async def create_thread(request: ThreadRequest):
    new_thread_id = str(uuid.uuid4())
    print(f"DEBUG: Created new thread: {new_thread_id}")
    return {
        "thread_id": new_thread_id,
        "metadata": request.metadata,
        "created_at": "2026-01-11T00:00:00Z",
        "updated_at": "2026-01-11T00:00:00Z"
    }

@app.post("/runs/wait")
async def runs_wait(request: Request):
    body = await request.json()
    user_input = body.get("input", {}).get("messages", [])[-1].get("content", "")
    result = await agent_app.ainvoke({"messages": [("user", user_input)]})
    final_content = result["messages"][-1].content
    return {
        "messages": [
            {
                "role": "assistant",
                "type": "ai",
                "content": final_content,
                "metadata": {}
            }
        ]
    }

@app.post("/threads/{thread_id}/runs/stream")
async def runs_stream(thread_id: str, request: Request):
    body = await request.json()
    user_input = body.get("input", {}).get("messages", [])[-1].get("content", "")

    async def event_generator():
        yield f"event: metadata\ndata: {json.dumps({'run_id': str(uuid.uuid4())})}\n\n"

        async for chunk in agent_app.astream(
            {"messages": [("user", user_input)]},
            stream_mode="values"
        ):
            if "messages" in chunk:
                msg = chunk["messages"][-1]
                if getattr(msg, "content", None):
                    yield f"data: {json.dumps({'event': 'values','data': {'messages':[{'role':'assistant','type':'ai','content': msg.content,'metadata': {}}]}})}\n\n"

        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str, request: Optional[HistoryRequest] = None):
    print(f"DEBUG: Warden UI requesting history for thread: {thread_id}")
    return [
        {
            "role": "assistant",
            "type": "ai",
            "content": "I'm ready! I've analyzed the latest Web3 raises. What would you like to know?",
            "metadata": {}
        }
    ]

# --- Updated Agent Card (Warden / Vercel Discovery) ---
@app.get("/.well-known/agent.json")
async def get_agent_manifest():
    return {
        "name": "Web3 RaiseRadar",
        "description": "Real-time funding research and Web3 project tracking.",
        "version": "1.0.0",
        "url": "https://web3-raiseradar-production-1f6f.up.railway.app",
        "skills": ["funding-research", "sentiment-analysis"],
        "author": "YourGitHubUsername"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
