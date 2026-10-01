"""
Memanto Python client.

A single-agent client that runs in-process — no server to start. It resolves
credentials from the environment or ``~/.memanto`` (or runs keyless on the
on-prem backend), creates the agent on first use, and reuses the agent's live
session instead of activating a new one.

    from memanto import Memanto

    memanto = Memanto(agent_id="my-agent")
    memanto.remember("Alex prefers oat milk.")
    memanto.recall("what does Alex drink?")
    memanto.answer("Does Alex drink dairy?")

Every method returns the same dict shape as the matching REST endpoint. For
operations not wrapped here, use the underlying ``SdkClient`` via
:attr:`Memanto.client`.
"""

from __future__ import annotations

import os
from typing import Any

from memanto.app.utils.errors import AgentAlreadyExistsError, AgentNotFoundError
from memanto.cli.client.sdk_client import SdkClient

__all__ = ["Memanto"]

# Placeholder key for the on-prem backend; OnPremClient ignores it.
_ON_PREM_API_KEY = "on-prem"


def _resolve_api_key(api_key: str | None) -> str:
    """Return the key to use: explicit arg, then env/``~/.memanto/.env``.

    The on-prem backend needs no key. On the cloud path the key is also
    exported as ``MOORCHEH_API_KEY`` because the app services read it from
    the environment (same as the CLI's ``get_client``).
    """
    from memanto.app.clients.backend import Backend
    from memanto.cli.config.manager import ConfigManager

    config = ConfigManager()
    if api_key is None and config.get_backend() == Backend.ON_PREM:
        return _ON_PREM_API_KEY

    key = api_key or os.environ.get("MOORCHEH_API_KEY") or config.get_api_key()
    if not key or not key.strip():
        raise ValueError(
            "No Moorcheh API key found. Pass api_key=..., set MOORCHEH_API_KEY, "
            "or run `memanto` once to configure a backend."
        )
    os.environ["MOORCHEH_API_KEY"] = key
    return key


class Memanto:
    """Memory client bound to one Memanto agent.

    Args:
        agent_id: Agent to read and write (alphanumeric, ``-`` and ``_``).
        api_key: Moorcheh API key. Defaults to ``MOORCHEH_API_KEY`` or the key
            saved by ``memanto`` setup; not needed on the on-prem backend.
        auto_create: Create the agent if it does not exist (default ``True``).
        pattern: Pattern for an auto-created agent — ``"tool"`` (default),
            ``"support"``, or ``"project"``.
        session_hours: Lifetime of a newly activated session. Defaults to the
            configured session duration.

    Raises:
        ValueError: If no API key can be resolved, or *agent_id* is invalid.
        AgentNotFoundError: If the agent does not exist and *auto_create* is
            ``False``.

    Memanto allows one session per agent: activating a session signs out every
    other client of that agent. So the constructor reuses the agent's live
    session when there is one and only activates when there is none. Use one
    instance per agent; an instance is not thread-safe.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        api_key: str | None = None,
        auto_create: bool = True,
        pattern: str = "tool",
        session_hours: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.client = SdkClient(api_key=_resolve_api_key(api_key))
        self._ensure_agent(auto_create, pattern)
        self._ensure_session(session_hours)

    def _ensure_agent(self, auto_create: bool, pattern: str) -> None:
        try:
            self.client.get_agent(self.agent_id)
            return
        except AgentNotFoundError:
            if not auto_create:
                raise
        try:
            self.client.create_agent(self.agent_id, pattern=pattern)
        except AgentAlreadyExistsError:
            pass  # created concurrently by another client

    def _ensure_session(self, session_hours: int | None) -> None:
        from memanto.app.services.session_service import get_session_service

        session = get_session_service().get_session(self.agent_id)
        if session is not None and session.is_active():
            self.client.session_token = session.session_token
            self.client.agent_id = self.agent_id
        else:
            self.client.activate_agent(self.agent_id, duration_hours=session_hours)

    # Write

    def remember(
        self,
        content: str,
        *,
        type: str | None = None,
        title: str | None = None,
        confidence: float = 0.8,
        tags: list[str] | None = None,
        source: str = "user",
        provenance: str | None = None,
    ) -> dict[str, Any]:
        """Store one memory. *title* defaults to the start of *content*.

        Returns a dict with ``memory_id``, ``status``, ``type``, etc.
        """
        if title is None:
            title = f"{content[:50]}..." if len(content) > 50 else content
        return self.client.remember(
            self.agent_id,
            memory_type=type,
            title=title,
            content=content,
            confidence=confidence,
            tags=tags,
            source=source,
            provenance=provenance,
        )

    def batch_remember(self, memories: list[dict[str, Any]]) -> dict[str, Any]:
        """Store up to 100 memories. Each item takes the same keys as
        :meth:`remember` (``content`` required)."""
        return self.client.batch_remember(self.agent_id, memories)

    def update_memory(self, memory_id: str, **updates: Any) -> dict[str, Any]:
        """Update fields of a memory, e.g. ``content=...`` or ``tags=[...]``."""
        return self.client.update_memory(self.agent_id, memory_id, updates)

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory by id."""
        return self.client.delete_memory(self.agent_id, memory_id)

    # Read

    def recall(
        self,
        query: str,
        *,
        limit: int | None = None,
        type: list[str] | None = None,
        tags: list[str] | None = None,
        min_similarity: float | None = None,
    ) -> dict[str, Any]:
        """Semantic search over this agent's memories.

        Returns a dict whose ``memories`` list is ranked by relevance.
        """
        return self.client.recall(
            self.agent_id,
            query,
            limit=limit,
            type=type,
            tags=tags,
            min_similarity=min_similarity,
        )

    def recall_as_of(
        self,
        as_of: str,
        *,
        limit: int | None = None,
        type: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """What the agent believed at *as_of* (ISO date or datetime)."""
        return self.client.recall_as_of(
            self.agent_id, as_of, limit=limit, type=type, tags=tags
        )

    def recall_changed_since(
        self,
        since: str,
        *,
        limit: int | None = None,
        type: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Memories created or updated after *since* (ISO date or datetime)."""
        return self.client.recall_changed_since(
            self.agent_id, since, limit=limit, type=type, tags=tags
        )

    def recall_recent(
        self,
        *,
        limit: int | None = None,
        type: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Most recently created memories, newest first."""
        return self.client.recall_recent(
            self.agent_id, limit=limit, type=type, tags=tags
        )

    def answer(
        self,
        question: str,
        *,
        limit: int | None = None,
        threshold: float | None = None,
        temperature: float | None = None,
        ai_model: str | None = None,
        kiosk_mode: bool | None = None,
    ) -> dict[str, Any]:
        """Answer *question* grounded in this agent's memories.

        Returns a dict with ``answer`` and the supporting memories.
        """
        return self.client.answer(
            self.agent_id,
            question,
            limit=limit,
            threshold=threshold,
            temperature=temperature,
            ai_model=ai_model,
            kiosk_mode=kiosk_mode,
        )
