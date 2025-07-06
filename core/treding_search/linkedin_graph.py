import os
import json
import logging
import re
import asyncio
from datetime import datetime
from typing import List, TypedDict

import aiohttp
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from duckduckgo_search import DDGS

# --------------------------------
# ✅ Shared Helpers
# --------------------------------
def clean_content(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)
    return text.strip()

def extract_all_text(soup):
    for script in soup(["script", "style", "noscript"]):
        script.extract()
    return clean_content(soup.get_text(separator=' '))

# --------------------------------
# ✅ Tools
# --------------------------------

@tool
async def scrape_url_tool(url: str) -> dict:
    """Scrapes the given URL and returns the cleaned title and text content."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as response:
                response.raise_for_status()
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.string if soup.title else ""
                content = extract_all_text(soup)
                return {
                    "url": url,
                    "title": clean_content(title),
                    "content": content
                }
    except Exception as e:
        return {"url": url, "error": str(e)}

@tool
def duckduckgo_search_tool(query: str) -> str:
    """Search using DuckDuckGo and return a brief snippet."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=1)
            for r in results:
                return f"{r['title']} - {r['body']} ({r['href']})"
    except Exception as e:
        return f"Search failed: {str(e)}"

def generate_post_tool_factory(model_name: str = "llama3.1") -> Runnable:
    """Returns a Runnable that generates a LinkedIn post from title + content."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a friendly Indian professional who writes simple and engaging LinkedIn posts about business and technology news.\n\n"
        "Your tone should be warm, relatable, and positive — like you're sharing with your professional network in India.\n\n"
        "Style:\n"
        "- Simple language\n"
        "- Short sentences\n"
        "- Friendly and inspiring tone\n"
        "- Emojis at beginning/end of lines\n"
        "- 3-5 hashtags\n\n"
        "DO NOT:\n"
        "- Say 'Here is the post'\n"
        "- Add any explanation\n"
        "- Use markdown or formatting tags\n\n"
        "Just return the post."
        ),
        ("human", "News Title: {title}\n\nNews Content: {content}\n\nWrite the LinkedIn post:")
    ])
    llm = OllamaLLM(model=model_name)
    return (prompt | llm | StrOutputParser()).with_config({"verbose": True})

# --------------------------------
# ✅ LangGraph State Definition
# --------------------------------
class State(TypedDict):
    news_all: List[dict]
    news_chunk: List[dict]
    posts: List[dict]
    idx: int
    messages: List[HumanMessage | AIMessage]

# --------------------------------
# ✅ Logger Node
# --------------------------------
def setup_logger(state: State) -> State:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("linkedinagent.log")]
    )
    logging.info("Logger initialized")
    return state

# --------------------------------
# ✅ Scrape URLs Node
# --------------------------------
def scrape_urls(state: State) -> State:
    with open("DATA/json/tech_website.json", "r") as f:
        urls = [item.get("url", item) for item in json.load(f)]

    async def fetch_all():
        tasks = [scrape_url_tool.ainvoke(url) for url in urls]
        return await asyncio.gather(*tasks)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(fetch_all())
    loop.close()

    state["news_all"] = results
    state["idx"] = 0
    state["posts"] = []
    return state

# --------------------------------
# ✅ Chunk Selector
# --------------------------------
def select_chunk(state: State) -> State:
    chunk_size = 5
    start = state["idx"] * chunk_size
    state["news_chunk"] = state["news_all"][start:start + chunk_size]
    return state

# --------------------------------
# ✅ Post Generator
# --------------------------------
def generate_posts(state: State) -> State:
    model_name = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
    agent = generate_post_tool_factory(model_name)
    posts = []

    for item in state["news_chunk"]:
        title = item.get("title", "").strip()
        content = item.get("content", "").strip()

        if not title:
            logging.warning(f"⚠️ Missing title: {item}")
            continue

        if len(content) < 50:
            snippet = duckduckgo_search_tool.invoke(title)
            content += f"\n\nExtra Info from DuckDuckGo:\n{snippet}"

        try:
            result = agent.invoke({"title": title, "content": content})
            posts.append({
                "title": title,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "linkedin_post": result.strip()
            })
            logging.info(f"✅ Post generated: {title}")
        except Exception as e:
            logging.error(f"❌ Failed for {title}: {e}")

    state["posts"].extend(posts)
    return state

# --------------------------------
# ✅ Append to File
# --------------------------------
def append_md(state: State) -> State:
    with open("linkedin_posts_output.md", "a", encoding="utf-8") as f:
        for p in state["posts"]:
            f.write(f"## 📰 {p['title']}\n")
            f.write(f"**⏰ {p['timestamp']}**\n\n")
            f.write(f"{p['linkedin_post']}\n\n---\n\n")
    return state

# --------------------------------
# ✅ Conditional Step
# --------------------------------
def next_or_end(state: State) -> str:
    total = len(state["news_all"])
    if (state["idx"] + 1) * 5 < total:
        state["idx"] += 1
        return "select_chunk"
    return END

# --------------------------------
# ✅ Graph Execution
# --------------------------------
def main():
    graph = StateGraph(State)

    graph.add_node("logger", setup_logger)
    graph.add_node("scrape", scrape_urls)
    graph.add_node("select_chunk", select_chunk)
    graph.add_node("generate", generate_posts)
    graph.add_node("append", append_md)

    graph.add_edge(START, "logger")
    graph.add_edge("logger", "scrape")
    graph.add_edge("scrape", "select_chunk")
    graph.add_edge("select_chunk", "generate")
    graph.add_edge("generate", "append")
    graph.add_conditional_edges("append", next_or_end, {"select_chunk": "select_chunk"})
    graph.add_edge("append", END)

    agent = graph.compile()
    agent.invoke({"messages": [], "idx": 0})

    print("✅ Done. Check linkedin_posts_output.md")

# --------------------------------
# ✅ Run
# --------------------------------
if __name__ == "__main__":
    main()
