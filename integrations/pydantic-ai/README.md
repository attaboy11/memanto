# Pydantic AI + Memanto

Pydantic AI tools that give a Pydantic AI agent persistent, cross-session memory backed by Memanto.

## What it does

`pydantic-ai-memanto` wraps Memanto's `SdkClient` as three Pydantic AI tools:

- `memanto_remember` — store a structured memory (fact, preference, decision, and 10 other types)
- `memanto_recall` — search stored memories by natural-language query
- `memanto_answer` — get a synthesized, RAG-grounded answer from stored memories

Same pattern as the CrewAI and LangGraph integrations: memory lives in a real Memanto agent, so you can read, edit, and audit everything with the Memanto CLI or web UI.

## Install

```bash
pip install pydantic-ai-memanto
export MOORCHEH_API_KEY=your_key_xxxxxxxxxxxxxxxxxx
```

## Use it

```python
from pydantic_ai import Agent

from memanto.cli.client.sdk_client import SdkClient
from memanto_pydantic_ai import MemantoSetup, create_memanto_tools

setup = MemantoSetup(api_key="your_key")
client = setup.setup(agent_id="travel-agent", description="Travel planning assistant")

agent = Agent(
    "openai:gpt-4o",
    system_prompt=(
        "You have long-term memory. Use memanto_remember to save durable facts, "
        "preferences, and decisions. Use memanto_recall or memanto_answer before "
        "answering whenever the user refers to something from an earlier session."
    ),
    tools=create_memanto_tools(client, agent_id="travel-agent"),
)

result = agent.run_sync("I always fly out of LAX, remember that for future trips.")
print(result.output)
```

`create_memanto_tools(client, agent_id)` binds the client and agent id when the tools are created, so the model's only inputs are the memory content or query it's reasoning about — it never has to pass a client or agent id itself.

## API

`MemantoSetup(api_key)` — creates or reuses a Memanto agent and activates a session.
- `.setup(agent_id, pattern="tool", description=None, duration_hours=6)` → returns an `SdkClient` ready to use
- `.teardown(agent_id)` — deactivates the session

`create_memanto_tools(client, agent_id)` → `list[Tool]` — the three tools, ready to pass into `Agent(tools=...)`.

## Tests

```bash
pip install -e ".[dev]"
pytest tests
```

Tests use a mocked `SdkClient`, so no network calls are made.
