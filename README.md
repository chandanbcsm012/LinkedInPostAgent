# 🧠 Trending Post LinkedIn Agent

This project uses LangGraph + LangChain + Ollama to generate professional, friendly LinkedIn posts in an Indian tone based on trending news articles. It enhances the content using DuckDuckGo web search and supports step-wise generation in a structured graph.

---

## 🚀 Features

- ✅ Loads structured news data from `organized_website_data.json`
- 🤖 Uses LLM (e.g., LLaMA 3 via Ollama) to generate high-quality LinkedIn posts
- 🔍 Adds context using DuckDuckGoSearch tool for enriched content
- 🧩 Built as a graph using `langgraph` for modularity and easy debugging
- 📄 Outputs clean, copy-ready posts in `linkedin_posts_output.md`
- 🇮🇳 Friendly, simple tone with emojis and hashtags
- 🪵 Logging enabled (`agent.log`) with optional verbose mode

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies:
- langgraph
- langchain
- langchain_community
- langchain_ollama
- duckduckgo_search
- ollama



## 🧪 Run the Project

```bash
python core/linkedin_graph.py
```

---

## ✍️ Prompt Design

We instruct the agent:
- Write like an Indian professional
- Use simple words, emojis, and relevant hashtags
- Keep it friendly and human

---

## 📌 Example Output

```markdown
## 📰 India Launches New AI Policy
**⏰ 2025-07-03 10:15:00**

India takes a bold step in AI governance with its new national policy focused on innovation and ethics. 🇮🇳🤖

Exciting times ahead for the tech ecosystem. Let's innovate responsibly! 💡

#AI #IndiaTech #Innovation #Policy
```

---

## 📬 Author

Made with ❤️ by Chandan Kumar