# main.py
import os
import json
import logging
import asyncio
from datetime import datetime
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from tools import (
    init_logger,
    scrape_urls,
    filter_content,
    enrich_content,
    select_chunk,
    generate_posts,
    write_output,
    next_or_end,
    get_urls_from_json
)

class State(TypedDict):
    urls: List[str]
    articles: List[dict]
    chunk: List[dict]
    idx: int
    posts: List[dict]
    messages: List[HumanMessage | AIMessage]

def main():
    urls = get_urls_from_json("DATA/json/tech_website.json")

    graph = StateGraph(State)
    graph.add_node("logger", init_logger)
    graph.add_node("scrape", scrape_urls)
    graph.add_node("filter", filter_content)
    graph.add_node("select_chunk", select_chunk)
    graph.add_node("enrich", enrich_content)
    graph.add_node("generate", generate_posts)
    graph.add_node("write", write_output)

    graph.add_edge(START, "logger")
    graph.add_edge("logger", "scrape")
    graph.add_edge("scrape", "filter")
    graph.add_edge("filter", "select_chunk")
    graph.add_edge("select_chunk", "enrich")
    graph.add_edge("enrich", "generate")
    graph.add_edge("generate", "write")
    graph.add_conditional_edges("write", next_or_end, {"select_chunk": "select_chunk"})
    graph.add_edge("write", END)

    app = graph.compile()
    initial_state: State = {"urls": urls, "articles": [], "idx": 0, "posts": [], "messages": []}
    app.invoke(initial_state)

    print("\n✅ Completed. Check 'linkedin_posts_output.md' for results.")

if __name__ == "__main__":
    main()
