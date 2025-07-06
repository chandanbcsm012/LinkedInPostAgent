# tools.py
import os
import re
import json
import logging
import aiohttp
import asyncio
from datetime import datetime
from typing import List
from bs4 import BeautifulSoup

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import END
# ----------------------
#  Utility Functions
# ----------------------
def get_urls_from_json(path: str) -> List[str]:
    with open(path, 'r') as f:
        data = json.load(f)
    return [entry['url'] if isinstance(entry, dict) else entry for entry in data]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)
    return text.strip()


def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return clean_text(soup.get_text())

# ----------------------
#  Logger
# ----------------------
def init_logger(state):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("linkedin_agent.log")
        ]
    )
    logging.info("Logger initialized.")
    return state

# ----------------------
#  Scrape URLs
# ----------------------
async def fetch(session, url):
    try:
        async with session.get(url, timeout=15) as resp:
            resp.raise_for_status()
            html = await resp.text()
            return {
                "url": url,
                "title": clean_text(BeautifulSoup(html, "html.parser").title.string or ""),
                "content": extract_text(html)
            }
    except Exception as e:
        logging.error(f"❌ Failed to fetch {url}: {e}")
        return {"url": url, "error": str(e)}

async def scrape_urls_async(urls):
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*(fetch(session, url) for url in urls))

def scrape_urls(state):
    urls = state["urls"]
    results = asyncio.run(scrape_urls_async(urls))
    state["articles"] = results
    return state

# ----------------------
#  Filter Content
# ----------------------
def filter_content(state):
    state["articles"] = [a for a in state["articles"] if a.get("title") and len(a.get("content", "")) > 100]
    return state

# ----------------------
#  Select Chunk
# ----------------------
def select_chunk(state):
    chunk_size = 5
    start = state["idx"] * chunk_size
    end = start + chunk_size
    state["chunk"] = state["articles"][start:end]
    return state

# ----------------------
#  Search Tool
# ----------------------
@tool
def duckduckgo_search_tool(query: str) -> str:
    """Search latest info using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=1)
            for r in results:
                return f"{r['title']} - {r['body']} ({r['href']})"
    except Exception as e:
        return f"DuckDuckGo error: {str(e)}"

# ----------------------
#  Enrich Content
# ----------------------
def enrich_content(state):
    for article in state["chunk"]:
        if len(article.get("content", "")) < 150:
            logging.info(f"🟡 Enriching: {article['title']}")
            snippet = duckduckgo_search_tool.invoke(article['title'])
            article['content'] += f"\n\n🔎 Extra Info: {snippet}"
    return state

# ----------------------
#  Post Generation Tool
# ----------------------
def generate_post_tool_factory(model_name="llama3.1") -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a friendly Indian professional writing LinkedIn posts about tech/business news.
        - Use simple language
        - Friendly tone with emojis
        - Add 3-5 relevant hashtags
        - Mention the source at the end
        - Do NOT include headers or formatting or explanations
        """),
        ("human", "News Title: {title}\n\nNews Content: {content}\n\nWrite a simple LinkedIn post:")
    ])
    llm = OllamaLLM(model=model_name)
    return prompt | llm | StrOutputParser()

def generate_posts(state):
    agent = generate_post_tool_factory()
    results = []

    for item in state["chunk"]:
        try:
            post = agent.invoke({"title": item["title"], "content": item["content"]})
            results.append({
                "title": item["title"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "linkedin_post": post.strip()
            })
            logging.info(f"✅ Generated post for: {item['title']}")
        except Exception as e:
            logging.error(f"❌ Post generation failed: {e}")

    state["posts"].extend(results)
    return state

# ----------------------
#  Write Output
# ----------------------
def write_output(state):
    with open("linkedin_posts_output.md", "a", encoding="utf-8") as f:
        for p in state["posts"]:
            f.write(f"## 📰 {p['title']}\n")
            f.write(f"**⏰ {p['timestamp']}**\n\n")
            f.write(f"{p['linkedin_post']}\n\n---\n\n")
    return state

# ----------------------
#  Next or End
# ----------------------
def next_or_end(state):
    # Make sure the key exists
    articles = state.get("articles")

    if not articles:
        logging.warning("⚠️ No articles found in state. Ending graph.")
        return END

    total = len(articles)
    current_chunk_end = (state["idx"] + 1) * 5

    logging.info(f"[🔁] Chunk index: {state['idx']} | Processed: {current_chunk_end}/{total}")

    if current_chunk_end < total:
        state["idx"] += 1
        return "select_chunk"

    logging.info("✅ All articles processed. Ending.")
    return END

