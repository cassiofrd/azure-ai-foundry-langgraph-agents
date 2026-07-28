# Azure AI Foundry + LangGraph Multi-Agent Sample

A reference project demonstrating how to build an AI agent using **Azure
AI Foundry**, **LangGraph**, **Azure AI Search**, and **Azure Monitor**.

## Features

-   Azure AI Foundry Responses API
-   LangGraph orchestration
-   Tool calling
-   Azure AI Search integration
-   Embedding-based retrieval
-   Conversation continuity (`previous_response_id`)
-   Structured execution metrics (`ExecutionContext`)
-   Azure Monitor / Application Insights telemetry
-   Automated unit tests

## Architecture

``` text
User
  │
  ▼
Supervisor Graph (LangGraph)
  │
  ├── Azure AI Foundry
  │
  ├── Tool Executor
  │      └── Local tools
  │
  ├── Azure AI Search
  │
  └── Telemetry
         ├── Console summary
         └── Azure Monitor / Application Insights
```

## Project structure

``` text
apps/
graphs/
shared/
scripts/
tests/
```

## Requirements

-   Python 3.12+
-   Azure AI Foundry Project
-   Azure OpenAI deployment
-   Azure AI Search (optional for retrieval)
-   Azure Application Insights (optional for telemetry)

## Installation

``` bash
git clone <repository-url>
cd azure-ai-foundry-langgraph-agents

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Copy the example configuration:

``` bash
cp .env.example .env
```

Fill in the Azure credentials.

## Running

``` bash
python -m apps.supervisor.main
```

## Running tests

``` bash
pytest -q
```

## Telemetry

When enabled, every execution generates:

-   Execution ID
-   Status
-   Duration
-   LLM calls
-   Tool calls
-   Token usage
-   Error count

The information is available both in the console and in Azure
Application Insights through `customEvents`.

## Main technologies

-   Python
-   LangGraph
-   Azure AI Foundry
-   Azure OpenAI
-   Azure AI Search
-   Azure Monitor
-   Application Insights
-   pytest

## Roadmap

-   [x] LangGraph orchestration
-   [x] Tool calling
-   [x] Azure AI Search integration
-   [x] Execution telemetry
-   [x] Azure Monitor integration
-   [ ] Dashboard / Workbook
-   [ ] Redis conversation memory
-   [ ] Multi-agent supervisor expansion

## License

MIT
