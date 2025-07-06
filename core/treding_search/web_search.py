import json
import logging
from langchain_community.tools import DuckDuckGoSearchResults

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------
# Read Organized Data
# -----------------------------
def read_organized_data(filename="organized_website_data.json"):
    logging.info(f"🔍 Reading data from {filename}")
    with open(filename, "r") as f:
        return json.load(f)

# -----------------------------
# Save Organized Data
# -----------------------------
def save_organized_data(data, filename="organized_website_data.json"):
    logging.info(f"💾 Saving organized data to {filename}")
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# -----------------------------
# Web Search via DuckDuckGo Tool
# -----------------------------
def get_web_search_results(query, max_results=3):
    logging.info(f"🌐 Performing web search for query: {query}")
    search = DuckDuckGoSearchResults(output_format="json")
    results = search.invoke(query)

    if isinstance(results, dict) and "results" in results:
        results = results["results"]

    organized = []
    for item in results[:max_results]:
        organized.append({
            "title": item.get("title", ""),
            "snippet": item.get("body", ""),
            "link": item.get("href", "")
        })
    logging.debug(f"📄 Search results: {organized}")
    return organized

# -----------------------------
# Append Web Search Results to Data
# -----------------------------
def append_web_search_to_data(filename="organized_website_data.json"):
    data = read_organized_data(filename)
    logging.info(f"📦 Total items loaded: {len(data)}")

    data_dict = {}
    skipped = 0
    for entry in data:
        url = entry.get("url")
        content = entry.get("content", "")

        if not url or not content:
            skipped += 1
            logging.warning(f"⚠️ Skipping entry due to missing URL/content: {entry}")
            continue

        query = content[:100]
        web_results = get_web_search_results(query)

        data_dict[url] = {
            "title": entry.get("title", ""),
            "content": content,
            "reference_link": url,
            "web_search": web_results
        }

    logging.info(f"✅ Processed: {len(data_dict)} items | Skipped: {skipped}")
    save_organized_data(data_dict, filename)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    append_web_search_to_data()
