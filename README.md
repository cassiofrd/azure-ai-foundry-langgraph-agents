# Azure AI Foundry + LangGraph Agents

Hands-on reference project combining **Microsoft Foundry** and **LangGraph** to explore model inference, agent orchestration, tools, knowledge, evaluations, tracing and deployment.

## Sprint 1 — Minimal Foundry model inside LangGraph

```text
User
  |
  v
LangGraph
  |
  v
Microsoft Foundry Project
  |
  v
Model deployment
```

### Learning goals

- Authenticate with Microsoft Entra ID using `DefaultAzureCredential`.
- Connect to a Foundry project with `AIProjectClient`.
- Obtain an authenticated OpenAI client from the project.
- Call the Responses API using a model deployment.
- Wrap the model call in a LangGraph node.
- Test the graph without making real Azure calls.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
az login
```

Configure `.env`:

```env
FOUNDRY_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>
FOUNDRY_MODEL_DEPLOYMENT=<deployment-name>
```

Run tests:

```powershell
$env:PYTHONPATH = (Get-Location).Path
pytest -q
```

Run the first LangGraph agent:

```powershell
python -m apps.supervisor.main
```

## Roadmap

- [x] Sprint 1: Foundry model invocation inside LangGraph
- [ ] Sprint 2: Conversation state and checkpointing
- [ ] Sprint 3: LangGraph tools
- [ ] Sprint 4: Azure AI Search knowledge
- [ ] Sprint 5: Foundry Agent Service
- [ ] Sprint 6: Evaluations
- [ ] Sprint 7: Tracing and observability
- [ ] Sprint 8: API and deployment
