import os, requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# --- 1. DEFINE REAL-TIME TOOLS ---
@tool
def get_breaking_news():
    """Fetches the latest Web3 news from CryptoPanic V2 API."""
    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    url = f"https://cryptopanic.com/api/v2/posts/?auth_token={api_key}&public=true"
    try:
        response = requests.get(url).json()
        news = [f"- {p['title']} (Source: {p['domain']})" for p in response['results'][:5]]
        return "\n".join(news) if news else "No news found."
    except:
        return "Error connecting to news feed."

search_tool = TavilySearchResults(max_results=3)
tools = [search_tool, get_breaking_news]

# --- 2. CONFIGURE THE AGENT ---
llm = ChatOpenAI(model="gpt-4o")
system_msg = (
    "You are a Web3 Sales Tracker Agent. Today is January 10, 2026. "
    "Always use tools to find live funding and ICO data. Format output in Markdown tables."
)

# This creates the Graph workflow automatically
agent_app = create_react_agent(llm, tools, state_modifier=system_msg)

# --- 3. API ENDPOINT FOR WARDEN ---
app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: Query):
    result = await agent_app.ainvoke({"messages": [("user", query.message)]})
    return {"response": result["messages"][-1].content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
