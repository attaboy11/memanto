"""
Memanto Tools for Pydantic AI

Wraps Memanto's SdkClient as Pydantic AI tools (recall/remember/answer) so a
Pydantic AI agent gets persistent, cross-session memory. Mirrors the pattern
already used for CrewAI, LangGraph, and the other Python integrations.
"""

from __future__ import annotations

import logging

from pydantic_ai import Tool

from memanto.app.utils.errors import AgentAlreadyExistsError
from memanto.cli.client.sdk_client import SdkClient

logger = logging.getLogger(__name__)


class MemantoSetup:
    """
    Manages Memanto agent lifecycle for the Pydantic AI integration.

    Handles agent creation, session activation, and teardown so a Pydantic AI
    script can focus on the agent's actual task.
    """

    def __init__(self, api_key: str) -> None:
        self.client = SdkClient(api_key=api_key)

    def setup(
        self,
        agent_id: str,
        pattern: str = "tool",
        description: str | None = None,
        duration_hours: int = 6,
    ) -> SdkClient:
        """Create agent (if needed) and activate a session."""
        try:
            self.client.create_agent(
                agent_id=agent_id,
                pattern=pattern,
                description=description,
            )
            logger.info("Created Memanto agent '%s'", agent_id)
        except AgentAlreadyExistsError:
            logger.info("Memanto agent '%s' already exists, reusing", agent_id)
        except Exception as e:
            logger.error("Failed to create agent '%s': %s", agent_id, e)
            raise

        self.client.activate_agent(agent_id, duration_hours=duration_hours)
        logger.info("Activated session for agent '%s'", agent_id)
        return self.client

    def teardown(self, agent_id: str) -> None:
        """Deactivate the agent session."""
        try:
            self.client.deactivate_agent(agent_id)
            logger.info("Deactivated session for agent '%s'", agent_id)
        except Exception as e:
            logger.warning("Failed to deactivate agent '%s': %s", agent_id, e)


def create_memanto_tools(client: SdkClient, agent_id: str) -> list[Tool]:
    """
    Create Memanto tools for a Pydantic AI agent, bound to a specific client
    and agent. Pass the result straight into ``Agent(tools=...)``:

    ```python
    from pydantic_ai import Agent
    from memanto.cli.client.sdk_client import SdkClient
    from memanto_pydantic_ai import create_memanto_tools

    client = SdkClient(api_key="...")
    client.activate_agent("my-agent")

    agent = Agent(
        "openai:gpt-4o",
        tools=create_memanto_tools(client, agent_id="my-agent"),
    )
    ```

    The client and agent id are captured by the tool functions (matching the
    CrewAI/LangGraph integrations), so the model never has to pass either one
    itself — its only inputs are the memory content/query it's reasoning about.
    """

    def memanto_remember(
        memory_type: str,
        title: str,
        content: str,
        confidence: float,
        tags: str = "",
    ) -> str:
        """Store a structured memory in Memanto for long-term persistence.

        Use this to save facts, observations, decisions, preferences, or any
        information that should be available in future sessions.

        Args:
            memory_type: The semantic type of memory to store. Must be exactly
                one of: fact (objective truths/data), preference (user
                likes/dislikes), goal (objectives/targets), decision (choices
                made/agreed upon), artifact (files/code/deliverables), learning
                (insights/lessons learned), event (occurrences/meetings),
                instruction (how-tos/directives), relationship (connections
                between entities), context (background info/state),
                observation (trends/patterns/notices), commitment (promises/
                next steps), or error (failures/mistakes).
            title: Short title for the memory (1-100 characters).
            content: The memory content to store (1-10000 characters). Be
                concise and atomic.
            confidence: Confidence score from 0.0 to 1.0. Use 1.0 for verified
                explicit facts, 0.7-0.85 for observations/estimates, and lower
                for unverified information.
            tags: Comma-separated tags for categorization (e.g. 'market,ai,trend').
                Use lowercase.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        result = client.remember(
            agent_id=agent_id,
            memory_type=memory_type,
            title=title,
            content=content,
            confidence=confidence,
            tags=tag_list,
            source="pydantic-ai-agent",
            provenance="explicit_statement",
        )

        return (
            f"Memory stored successfully.\n"
            f"  ID: {result['memory_id']}\n"
            f"  Type: {memory_type}\n"
            f"  Title: {title}\n"
            f"  Confidence: {confidence}"
        )

    def memanto_recall(
        query: str,
        limit: int = 10,
        memory_types: str = "",
        min_similarity: float | None = None,
    ) -> str:
        """Search Memanto's persistent memory database using natural language.

        Returns stored memories ranked by semantic relevance. Use this to
        retrieve facts, research findings, decisions, or any previously
        stored information.

        Args:
            query: Natural language search query to find relevant memories.
            limit: Maximum number of memories to retrieve.
            memory_types: Comma-separated memory types to filter by (e.g.
                'fact,observation'). Leave empty for all types.
            min_similarity: Minimum similarity score from 0.0 to 1.0 to filter
                low-relevance memories.
        """
        type_list = (
            [t.strip() for t in memory_types.split(",") if t.strip()]
            if memory_types
            else None
        )

        result = client.recall(
            agent_id=agent_id,
            query=query,
            limit=limit,
            type=type_list,
            min_similarity=min_similarity,
        )

        memories = result.get("memories", [])
        if not memories:
            return f"No memories found for query: '{query}'"

        lines = [f"Found {len(memories)} memories for '{query}':\n"]
        for i, mem in enumerate(memories, 1):
            title = mem.get("title", "Untitled")
            content = mem.get("content", "")
            mem_type = mem.get("type", "unknown")
            confidence = mem.get("confidence", "N/A")
            tags = mem.get("tags", [])
            tag_str = f" [tags: {', '.join(tags)}]" if tags else ""

            lines.append(
                f"  {i}. [{mem_type}] {title} (confidence: {confidence}){tag_str}\n"
                f"     {content}\n"
            )

        return "\n".join(lines)

    def memanto_answer(question: str) -> str:
        """Get an AI-generated answer grounded in stored memories (RAG).

        Use this to synthesize insights from multiple stored memories into a
        coherent answer, instead of returning raw search results.

        Args:
            question: The question to answer from memory.
        """
        result = client.answer(agent_id=agent_id, question=question)

        answer = result.get("answer", "No answer could be generated.")
        sources = result.get("sources", [])

        output = f"Answer: {answer}"
        if sources:
            output += f"\n\nBased on {len(sources)} memory source(s)."

        return output

    return [
        Tool(memanto_remember, name="memanto_remember", takes_ctx=False),
        Tool(memanto_recall, name="memanto_recall", takes_ctx=False),
        Tool(memanto_answer, name="memanto_answer", takes_ctx=False),
    ]
