# Investor Intelligence System

An AI-powered platform for ingesting, analyzing, and querying investment-related data using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). The system combines structured financial data with LLM reasoning to surface actionable investor insights.

## Overview

Investor Intelligence System is built to help investors, analysts, and researchers make sense of large volumes of financial and market data. It ingests raw data, stores it in a structured database and vector store, and exposes an API for querying insights through a RAG-powered LLM pipeline.

## Features

- **Data Ingestion** — Automated pipelines for collecting and processing investment-related data from various sources.
- **Vector Search** — Semantic search over financial documents and data using a vector store.
- **RAG Pipeline** — Retrieval-Augmented Generation for grounding LLM responses in relevant, up-to-date data.
- **LLM Integration** — Configurable LLM layer for generating insights, summaries, and answers to investor queries.
- **REST API** — FastAPI-based routes and schemas for integrating the system into other applications.
- **Structured Storage** — Relational/database layer for persisting processed and structured data.

## Project Structure

```
Investor_Intelligence_System/
├── data/            # Raw and/or processed data files
├── database/        # Database models, connections, and migrations
├── Ingestion/        # Data ingestion pipelines and connectors
├── llm/             # LLM client configuration, prompts, and orchestration
├── rag/             # Retrieval-Augmented Generation logic
├── routes/          # API route/endpoint definitions
├── schemas/         # Request/response and data validation schemas
├── vector_store/    # Vector database integration for embeddings/search
├── main.py          # Application entry point
├── notebook.ipynb   # Exploratory analysis / experimentation notebook
├── docs.txt         # Additional project documentation/notes
├── pyproject.toml   # Project metadata and dependencies
├── uv.lock          # Locked dependency versions (uv package manager)
├── .python-version  # Pinned Python version
└── .env             # Environment variables (not committed to version control)
```

## Tech Stack

- **Language:** Python
- **Package Management:** [uv](https://github.com/astral-sh/uv)
- **API Framework:** FastAPI (based on `routes/` and `schemas/` structure)
- **LLM/RAG:** Custom orchestration in `llm/` and `rag/` modules
- **Vector Store:** Embedding-based semantic search (see `vector_store/`)
- **Database:** Structured storage layer (see `database/`)

## Getting Started

### Prerequisites

- Python (version pinned in `.python-version`)
- [uv](https://github.com/astral-sh/uv) installed for dependency management

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Investor_Intelligence_System
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with the required configuration (API keys, database connection strings, vector store credentials, etc.). See `docs.txt` for details on required variables.

### Running the Application

```bash
uv run main.py
```

The API will be available at the configured host/port (default FastAPI settings unless overridden).

## Usage

- Use the `Ingestion/` pipelines to load and process new investment data.
- Query the system through the API routes defined in `routes/` to retrieve LLM-generated insights grounded in your data via the RAG pipeline.
- Use `notebook.ipynb` for ad hoc exploration, testing, and prototyping.

## Configuration

Environment-specific settings (API keys, database URLs, vector store endpoints, LLM provider credentials, etc.) should be placed in a `.env` file. Refer to `docs.txt` for a full list of required and optional environment variables.

## Contributing

1. Create a feature branch from `main`.
2. Make your changes and ensure dependencies are tracked via `pyproject.toml`/`uv.lock`.
3. Submit a pull request with a clear description of the changes.