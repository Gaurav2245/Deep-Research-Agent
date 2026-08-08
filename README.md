# Deep Research Agent

A modular research assistant built with LangGraph, Azure OpenAI, and multiple search/scraping backends. It can answer questions using live web data, JavaScript-rendered pages, HTML tables, and NSE India market data.

## What it does

- Runs a multi-step research workflow with search, scraping, and synthesis
- Supports Tavily, DuckDuckGo, Playwright-based scraping, and NSE market data queries
- Auto-detects Indian stock-market queries (nifty/sensex/gainers/losers/banknifty) and routes them to live NSE data instead of generic web search
- Runs each round's search queries and deep-scrapes concurrently to cut wall-clock latency
- Exposes the workflow through a CLI, a FastAPI backend, and a Streamlit UI
- Exports any completed research session as a PDF report via the API or the UI
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

3. Create a local environment file from the template and fill in your credentials
   ```bash
   cp .env.example .env      # Windows: copy .env.example .env
   ```

   [.env.example](.env.example) documents every variable the app reads (required and optional), including:
   - `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `TAVILY_API_KEY` — required
   - `ALLOWED_ORIGINS` — CORS allow-list for the API (defaults to local dev ports)
   - `API_KEY` — if set, all `/api/v1/*` requests must send it in the `X-API-Key` header; leave unset for local dev (no auth enforced)
   - `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` — per-client-IP rate limit (default: 120 req/60s); set `RATE_LIMIT_REQUESTS=0` to disable
   - `ENABLE_NSE_AUTO_ROUTING` — auto-route Indian market-data queries to NSE's live API (default: on)
   - `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` / `DB_NAME` — Postgres connection (defaults match `docker-compose.yml`)

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

Once a research session completes, download its PDF report with `GET /api/v1/research/{id}/pdf`.

## Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Project layout

```text
agents/          LangGraph workflow: query understanding, search, scraping,
                 scoring, synthesis, validation, and DB persistence nodes
api/             FastAPI routes, schemas, and optional API-key auth
config/          Environment/configuration handling
database/        Models, confidence/source scoring, embeddings, and
                 data validation logic
llm/             LLM provider wrappers
tools/           Search, scraping, and NSE adapters
utils/           Logging, PDF generation, and helpers
```

## Notes

- The app can work with search-only flows, but Playwright-based scraping is enabled by default for JS-heavy sites.
- The FastAPI backend and Streamlit UI both depend on the same underlying research engine.
- Source scoring, confidence scoring, and data-quality validation run as part of the LangGraph pipeline itself (`agents/enhanced_nodes.py`) and are persisted to the same `Research` row the API returns.
- NSE auto-routing (`tools/nse_router_tool.py`, `tools/query_router.py`) is deliberately conservative: it only fires for queries anchored to a recognizable Indian-market token (nifty/sensex/nse/bse/banknifty) and always falls back to normal web search if NSE itself fails.
- Rate limiting is in-memory and per-process — fine for a single API instance, but won't coordinate limits across multiple replicas. Swap `api/rate_limit.py` for a Redis-backed limiter before scaling horizontally.
- If you are running the API locally, [quick_start.py](quick_start.py) can be used as a simple smoke test once the server is up.

---

**Version 2.0.0** | Last Updated: 2026-08-07
