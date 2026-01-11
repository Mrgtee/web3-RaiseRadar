import os
import requests
import uuid
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # Added for Vercel/Warden compatibility
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
# This allows the Warden Vercel tester to call your Railway URL
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
    """
    Fetches trending crypto news and market sentiment from CryptoPanic.
    """
    # DEBUG: Crucial for monitoring Railway logs
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

# --- NEW: ThreadRequest Model ---
class ThreadRequest(BaseModel):
    metadata: Optional[dict] = {}

# 7. Endpoints
@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    try:
        inputs = {"messages": [("user", query.message)]}
        result = await agent_app.ainvoke(inputs)

        final_answer = result["messages"][-1].content

        return {
            "response": [
                {
                    "type": "text",
                    "text": final_answer
                }
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def health_check():
    return {"status": "active", "agent": "RaiseRadar-v1"}

# --- ADDED: Warden / LangGraph compatibility endpoints ---
@app.get("/info")
async def info():
    """Returns agent metadata expected by the Warden/LangGraph tester."""
    return {
        "assistant_id": "raiseradar",
        "graph_id": "agent",
        "config": {},
    }

@app.post("/threads/search")
async def threads_search():
    """Simulates thread search to prevent 404s in the Warden UI."""
    return []

@app.post("/assistants/search")
async def assistants_search():
    """Declares available assistants to the tester."""
    return [
        {
            "assistant_id": "raiseradar",
            "name": "RaiseRadar Agent"
        }
    ]

# --- NEW: /threads endpoint ---
@app.post("/threads")
async def create_thread(request: ThreadRequest):
    """
    Creates a unique thread ID for the conversation.
    The Warden tester needs this to start the chat.
    """
    new_thread_id = str(uuid.uuid4())
    print(f"DEBUG: Created new thread: {new_thread_id}")
    return {
        "thread_id": new_thread_id,
        "metadata": request.metadata,
        "created_at": "2026-01-11T00:00:00Z",  # Current date
        "updated_at": "2026-01-11T00:00:00Z"
    }

# --- NEW: /runs/wait endpoint ---
@app.post("/runs/wait")
async def runs_wait(request: Request):
    body = await request.json()
    # Extract the message from the standard LangGraph body
    user_input = body.get("input", {}).get("messages", [])[-1].get("content", "")
    
    # Optional: You can extract thread_id if you want to save history later
    # thread_id = body.get("thread_id") 

    inputs = {"messages": [("user", user_input)]}
    result = await agent_app.ainvoke(inputs)
    
    return {
        "messages": [
            {
                "role": "assistant",
                "content": result["messages"][-1].content
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    # Railway sets the PORT environment variable automatically
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

