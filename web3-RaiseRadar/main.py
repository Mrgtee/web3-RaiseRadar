import os
import requests
import uuid
from typing import List, Optional, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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

# --- CORS Middleware (Crucial for Vercel/Warden) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Tools & Agent Logic (Keeping your existing logic)
@tool
def fetch_crypto_news(query: str) -> str:
    """Fetches trending crypto news and market sentiment from CryptoPanic."""
    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    params = {"auth_token": api_key, "public": "true", "kind": "news", "regions": "en", "filter": "hot"}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        results = data.get('results', [])[:3]
        return "\n".join([f"- **{p['title']}**\n Source: {p['url']}" for p in results]) if results else "No news."
    except Exception as e: return f"Error: {str(e)}"

search_tool = TavilySearch(max_results=5, search_depth="advanced", include_answer=True)
tools = [search_tool, fetch_crypto_news]
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)

system_msg = (
    "You are the Web3 RaiseRadar Agent. "
    "Format responses in a Markdown Table: Project | Event | Date | Source/Link."
)
agent_app = create_react_agent(llm, tools, prompt=system_msg)

# 3. WARDEN COMPATIBILITY ENDPOINTS (The Fix)
# These endpoints satisfy the Vercel tester's discovery process

@app.get("/info")
async def info():
    """Tells the tester who this agent is."""
    return {
        "assistant_id": "raiseradar",
        "graph_id": "agent",
        "config": {}
    }

@app.post("/assistants/search")
async def assistants_search():
    """Allows the UI to find this assistant."""
    return [{"assistant_id": "raiseradar", "name": "RaiseRadar Agent"}]

@app.post("/threads/search")
async def threads_search():
    """Prevents 404 when the UI checks for history."""
    return []

# 4. Standard Chat & Run Endpoints
class ChatQuery(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    """Your custom endpoint for manual testing."""
    inputs = {"messages": [("user", query.message)]}
    result = await agent_app.ainvoke(inputs)
    return {"response": [{"type": "text", "text": result["messages"][-1].content}]}

@app.post("/runs/wait")
async def runs_wait(request: Request):
    """The official LangGraph endpoint the Vercel tester actually uses to chat."""
    body = await request.json()
    user_input = body.get("input", {}).get("messages", [])[-1].get("content", "")
    
    inputs = {"messages": [("user", user_input)]}
    result = await agent_app.ainvoke(inputs)
    
    # Format according to LangGraph spec
    return {
        "messages": [
            {
                "role": "assistant",
                "content": result["messages"][-1].content
            }
        ]
    }

@app.get("/")
async def health_check():
    return {"status": "active", "agent": "RaiseRadar-v1"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
