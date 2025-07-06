import json
import logging
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def read_tech_websites(filename="DATA/json/tech_website.json"):
    logger.info(f"Reading website list from {filename}")
    with open(filename, "r") as f:
        return json.load(f)

def clean_content(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]+', '', text)
    return text.strip()

def extract_all_text(soup):
    # Extract all visible text from the page, excluding scripts/styles
    for script in soup(["script", "style", "noscript"]):
        script.extract()
    text = soup.get_text(separator=' ')
    return clean_content(text)

async def fetch_url_content(session, url):
    logger.info(f"Fetching content from: {url}")
    try:
        async with session.get(url, timeout=15) as response:
            response.raise_for_status()
            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.string if soup.title else ""
            all_content = extract_all_text(soup)
            logger.info(f"Fetched and cleaned content for: {url}")
            return {
                "url": url,
                "title": clean_content(title),
                "content": all_content
            }
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return {"url": url, "error": str(e)}

def save_organized_data(data, filename="DATA/json/organized_website_data.json"):
    logger.info(f"Saving organized data to {filename}")
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("Data saved successfully.")

async def main():
    websites = read_tech_websites()
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for entry in websites:
            url = entry.get("url") if isinstance(entry, dict) else entry
            tasks.append(fetch_url_content(session, url))
        results = await asyncio.gather(*tasks)
    save_organized_data(results)

if __name__ == "__main__":
    asyncio.run(main())