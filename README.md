# Deep Research Agent

A modular research assistant built with LangGraph, Azure OpenAI, and multiple search/scraping backends. It can answer questions using live web data, JavaScript-rendered pages, HTML tables, and NSE India market data.

## What it does

- Runs a multi-step research workflow with search, scraping, and synthesis
- Supports Tavily, DuckDuckGo, Playwright-based scraping, and NSE market data queries
- Exposes the workflow through a CLI, a FastAPI backend, and a Streamlit UI
- Keeps the provider layer loosely coupled so different search/scraping implementations can be swapped in

## Main entry points

- CLI: [main.py](main.py)
- FastAPI app: [api/main.py](api/main.py)
- Streamlit UI: [streamlit_app.py](streamlit_app.py)
- Quick API smoke test: [quick_start.py](quick_start.py)

## Features

- LangGraph-based research workflow
- Azure OpenAI integration
- Tavily search and DuckDuckGo fallback
- Playwright scraper for JavaScript-heavy sites
- HTML table extraction and formatting
- NSE India live market data support
- Conversation-aware research flow and API endpoints

## Prerequisites

- Python 3.10+
- A working internet connection for live web research
- Azure OpenAI credentials and a Tavily API key (or DuckDuckGo fallback)

## Setup

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

3. Create a local environment file
   ```bash
   copy NUL .env
   ```

4. Add the required environment variables to [.env](.env)
   ```env
   AZURE_OPENAI_API_KEY=your_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
   TAVILY_API_KEY=your_tavily_key
   ```

   Optional variables:
   ```env
   SEARCH_PROVIDER=tavily
   LLM_PROVIDER=azure_openai
   ENABLE_SCRAPER=true
   SCRAPER_MAX_URLS=3
   RESEARCH_DEPTH=deep
   MAX_SEARCH_ITERATIONS=3
   ```

## Run the CLI

```bash
python main.py "What are the latest RBI monetary policy decisions?"
```

You can also target NSE-backed queries directly:

```bash
SEARCH_PROVIDER=nse python main.py "nifty 50"
SEARCH_PROVIDER=nse python main.py "quote RELIANCE"
SEARCH_PROVIDER=nse python main.py "option chain NIFTY"
SEARCH_PROVIDER=nse python main.py "gainers"
```

## Run the API

```bash
uvicorn api.main:app --reload
```

Then open:

- http://localhost:8000/docs
- http://localhost:8000/health

## Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Project layout

```text
agents/          Research workflow and agent nodes
api/             FastAPI routes and schemas
config/          Environment/configuration handling
database/       Persistence, scoring, and validation logic
llm/             LLM provider wrappers
tools/           Search, scraping, and NSE adapters
utils/           Logging, PDF generation, and helpers
```

## Notes

- The app can work with search-only flows, but Playwright-based scraping is enabled by default for JS-heavy sites.
- The FastAPI backend and Streamlit UI both depend on the same underlying research engine.
- If you are running the API locally, [quick_start.py](quick_start.py) can be used as a simple smoke test once the server is up.
- **Quick Help?** → `QUICK_REFERENCE.md`

---

**Version 2.0.0** | Production Ready | Last Updated: 2026-04-30
