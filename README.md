# RAG AI — Advanced Multi-Agent System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-blueviolet)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-ff4b4b)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-1.7-green)](https://github.com/facebookresearch/faiss)
[![MCP](https://img.shields.io/badge/Protocol-MCP--1.0-orange)](https://modelcontextprotocol.io)
[![Model](https://img.shields.io/badge/LLM-Qwen--2.5--7B-darkblue)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, **multi-agent AI system** built with **LangGraph** and the **Model Context Protocol (MCP)**. Features a supervisor orchestrator that routes queries to specialized agents (Document, Research, Data), 7 integrated tools, and an advanced Streamlit interface with real-time agent activity visualization.

## Architecture

```
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    │  (Agent Selector)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Supervisor Agent   │
                    │  (Intent Classifier │
                    │   + Router)         │
                    └──┬────────┬────────┬┘
                       │        │        │
              ┌────────▼──┐ ┌───▼────┐ ┌─▼─────┐
              │ Document  │ │Research│ │  Data │
              │  Agent    │ │ Agent  │ │  Agent│
              └─────┬─────┘ └───┬────┘ └───┬───┘
                    │           │          │
         ┌──────────────────────────────────────────┐
         │          MCP Protocol Layer              │
         │  (Server → Registry → Client → Tools)    │
         └──────────────────────────────────────────┘
                    │           │           │
         ┌──────┐ ┌────┐ ┌─────┐ ┌────┐ ┌───────┐ ┌────┐ ┌─────┐
         │ RAG  │ │Web │ │Calc │ │Code│ │Weather│ │Web │ │Summ │
         │Search│ │Srch│ │     │ │Anls│ │       │ │Read│ │     │
         └──────┘ └────┘ └─────┘ └────┘ └───────┘ └────┘ └─────┘
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Supervisor** | Classifies intent, routes to specialists, handles weather | `fetch_weather`, `retrieve_context` |
| **Document Agent** | PDF Q&A with RAG retrieval | `retrieve_context`, `summarize_text` |
| **Research Agent** | Web search & information synthesis | `web_search`, `read_webpage`, `summarize_text` |
| **Data Agent** | Math computation & code analysis | `calculate`, `analyze_code` |

### Tools (7 total)

| Tool | Category | Description |
|------|----------|-------------|
| `retrieve_context` | Retrieval | Semantic search over uploaded PDFs |
| `fetch_weather` | API | Real-time weather via OpenWeatherMap |
| `web_search` | API | DuckDuckGo web search (no API key) |
| `read_webpage` | API | URL content extraction |
| `summarize_text` | Analysis | LLM-powered text summarization |
| `calculate` | Computation | Safe math expression evaluator |
| `analyze_code` | Analysis | Python code structure analysis |

### MCP Protocol

The **Model Context Protocol** layer decouples tools from agents:
- **`mcp/protocol.py`** — Typed message schemas (Pydantic)
- **`mcp/server.py`** — Singleton tool registry with async dispatch, timeouts, metrics
- **`mcp/client.py`** — Agent-facing client with LRU caching, parallel execution, traces

### Latency Optimizations

- **Async tool dispatch** — tools run in async event loop via `asyncio.to_thread`
- **Connection pooling** — shared `requests.Session` with retry logic for HTTP tools
- **TTL caching** — weather (5 min), RAG retrieval (2 min) cached to avoid redundant calls
- **LRU cache** — MCP client caches per-agent results (128 entries)
- **Lazy agent loading** — specialist agents instantiated on first use, not at startup
- **Module-level model caching** — LLM and embeddings cached with fallback for non-Streamlit

## Prerequisites

- Python 3.12+
- OpenWeatherMap API key ([openweathermap.org](https://openweathermap.org/api))
- HuggingFace token ([huggingface.co](https://huggingface.co/settings/tokens))
- LangSmith API key (optional)

## Installation

```bash
git clone <repository-url>
cd "multi-agent-rag"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```bash
OPENWEATHERMAP_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here
LANGSMITH_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true               
```

## Usage

```bash
streamlit run app.py
```

### Agent Modes

- **Auto** — Supervisor automatically routes to the best specialist
- **Document** — Force document agent for PDF questions
- **Research** — Force research agent for web search
- **Data** — Force data agent for math/code

### Example Queries

| Query | Agent | Tools Used |
|-------|-------|-----------|
| "What does the resume say about experience?" | Document | `retrieve_context` |
| "Weather in Mumbai?" | Supervisor | `fetch_weather` |
| "Latest news about AI" | Research | `web_search`, `read_webpage` |
| "Calculate sqrt(144) + sin(radians(30))" | Data | `calculate` |
| "Analyze this Python code: def foo(): pass" | Data | `analyze_code` |

## 📁 Project Structure

```
rag-ai/
├── app.py                  # Streamlit UI (agent selector, traces, glass design)
├── agent.py                # Entry point → delegates to SupervisorAgent
├── mcp/
│   ├── protocol.py         # MCP message types (Pydantic models)
│   ├── server.py           # Tool registry + async dispatch engine
│   └── client.py           # Agent-facing client (caching, parallel, traces)
├── agents/
│   ├── base.py             # Abstract base agent (LangGraph loop + MCP)
│   ├── supervisor.py       # Orchestrator (intent classifier + router)
│   ├── research_agent.py   # Web search specialist
│   ├── document_agent.py   # PDF Q&A specialist
│   └── data_agent.py       # Math + code specialist
├── tools/
│   ├── rag.py              # RAG retrieval (cached)
│   ├── weather.py          # Weather API (pooled + cached)
│   ├── web_search.py       # DuckDuckGo search
│   ├── web_reader.py       # URL content extraction
│   ├── summarizer.py       # LLM summarization
│   ├── calculator.py       # Safe math evaluator
│   └── code_analysis.py    # Python AST analysis
├── utils/
│   ├── llm.py              # Model init (cached, lazy)
│   ├── vector_store.py     # FAISS operations
│   └── cache.py            # TTL + LRU caching
├── tests/
│   ├── test_agent.py       # Entry point tests
│   ├── test_tools.py       # All 7 tools
│   ├── test_mcp.py         # MCP protocol + server + client
│   └── test_agents.py      # Routing + parsing tests
└── faiss_index/            # Auto-created
```

## Testing

```bash
python -m pytest tests/ -v
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Agent workflow orchestration |
| `langchain` | LLM framework |
| `streamlit` | Web UI |
| `faiss-cpu` | Vector search |
| `llama-cpp-python` | Local LLM inference |
| `pydantic` | MCP protocol schemas |
| `duckduckgo-search` | Web search |
| `trafilatura` | Web content extraction |

---


## NOTE:
### OpenWeatherMap API Key
1. Sign up at [openweathermap.org](https://openweathermap.org/api)
2. Get your free API key from the dashboard
3. Add it to `.env` as `OPENWEATHERMAP_API_KEY`

### HuggingFace Token
1. Create an account at [huggingface.co](https://huggingface.co)
2. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept the terms for `google/embeddinggemma-300m` model
4. Add it to `.env` as `HUGGINGFACE_TOKEN`

### LangSmith API Key (Optional)
LangSmith provides observability and tracing for LangChain applications. To enable tracing:

1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Create a new API key in your account settings
3. Get your workspace ID from the LangSmith dashboard