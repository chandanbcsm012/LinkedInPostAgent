import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

# -----------------------------
# Logger Setup
# -----------------------------
def setup_logger(log_file: str = "linkedin_post_generator.log") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# -----------------------------
# Load News Data
# -----------------------------
def load_news_data(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Create Prompt Chain
# -----------------------------
def create_linkedin_chain(model_name: str = "llama3.1") -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional LinkedIn content writer. "
         "Given a news title and its content, generate a concise, insightful LinkedIn post suitable for tech or business professionals. "
         "Keep it clear, engaging, and aligned with a professional tone.\n\n"
         "**Important**: Do not include any intro like 'Here is the post' or 'Sure!'. "
         "Just return the LinkedIn post itself—no extra comments, no markdown formatting, no explanations.\n\n"
         "The post should be clean and ready for direct copy-paste."
        ),
        ("human",
         "News Title: {title}\n\n"
         "News Content: {content}\n\n"
         "Write a professional LinkedIn post based on the above.")
    ])

    model = OllamaLLM(model=model_name)
    output_parser = StrOutputParser()
    return prompt | model | output_parser


# -----------------------------
# Generate LinkedIn Posts
# -----------------------------
async def generate_linkedin_posts(chain: Runnable, news_items: List[Dict[str, Any]], logger: logging.Logger) -> List[Dict[str, Any]]:
    logger.info("Starting LinkedIn post generation...")
    tasks = [chain.ainvoke(item) for item in news_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    enriched_posts = []
    for idx, (item, result) in enumerate(zip(news_items, results)):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content"),
            "timestamp": timestamp
        }

        if isinstance(result, Exception):
            logger.error(f"Error processing item #{idx + 1}: {result}")
            entry["linkedin_post"] = f"Error: {result}"
        else:
            logger.info(f"Generated post for item #{idx + 1}")
            entry["linkedin_post"] = result

        enriched_posts.append(entry)

    return enriched_posts


# -----------------------------
# Utility: Chunking
# -----------------------------
def chunk_list(data: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


# -----------------------------
# Save to Markdown File
# -----------------------------
def append_posts_to_markdown_file(posts: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        for post in posts:
            f.write(f"## 📰 {post['title']}\n")
            f.write(f"**⏰ {post['timestamp']}**\n\n")
            f.write(f"{post['linkedin_post'].strip()}\n\n")
            f.write("---\n\n")


# -----------------------------
# Main Entry Point
# -----------------------------
def main():
    logger = setup_logger()
    json_path = "DATA/json/organized_website_data.json"
    output_md_path = "linkedin_posts_output.md"
    model_name = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
    chunk_size = 5

    try:
        # Clean previous output if exists
        if os.path.exists(output_md_path):
            os.remove(output_md_path)
            logger.info(f"Old output file '{output_md_path}' removed.")

        news_data = load_news_data(json_path)
        logger.info(f"Loaded {len(news_data)} news items from '{json_path}'")

        news_chunks = chunk_list(news_data, chunk_size)
        chain = create_linkedin_chain(model_name)

        for idx, chunk in enumerate(news_chunks):
            logger.info(f"Processing chunk {idx + 1} of {len(news_chunks)}...")
            linkedin_posts = asyncio.run(generate_linkedin_posts(chain, chunk, logger))

            append_posts_to_markdown_file(linkedin_posts, output_md_path)
            logger.info(f"Appended chunk {idx + 1} to '{output_md_path}'")

            # Print preview
            for i, post in enumerate(linkedin_posts):
                print(f"\n--- Chunk {idx + 1} - Post #{i + 1} ---")
                print(post["linkedin_post"].strip())

        logger.info(f"✅ All posts written to '{output_md_path}'")

    except Exception as e:
        logger.exception(f"Unhandled exception occurred: {e}")


if __name__ == "__main__":
    main()
