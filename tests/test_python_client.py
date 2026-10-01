"""Tests for the public ``memanto.Memanto`` client wrapper."""

from unittest.mock import MagicMock, patch

import pytest

from memanto.app.utils.errors import AgentAlreadyExistsError, AgentNotFoundError


@pytest.fixture
def sdk():
    """Patch SdkClient, key resolution, and the session store."""
    client = MagicMock()
    session_service = MagicMock()
    session_service.get_session.return_value = None
    with (
        patch("memanto.client.SdkClient", return_value=client) as cls,
        patch("memanto.client._resolve_api_key", return_value="key"),
        patch(
            "memanto.app.services.session_service.get_session_service",
            return_value=session_service,
        ),
    ):
        client.session_service = session_service
        client.cls = cls
        yield client


def test_top_level_import():
    import memanto
    from memanto.client import Memanto

    assert memanto.Memanto is Memanto
    with pytest.raises(AttributeError):
        _ = memanto.NotAThing


def test_existing_agent_without_session_activates(sdk):
    from memanto import Memanto

    Memanto("bot", session_hours=2)

    sdk.cls.assert_called_once_with(api_key="key")
    sdk.create_agent.assert_not_called()
    sdk.activate_agent.assert_called_once_with("bot", duration_hours=2)


def test_live_session_is_reused_not_reactivated(sdk):
    from memanto import Memanto

    session = MagicMock(session_token="tok")
    session.is_active.return_value = True
    sdk.session_service.get_session.return_value = session

    Memanto("bot")

    sdk.activate_agent.assert_not_called()
    assert sdk.session_token == "tok"
    assert sdk.agent_id == "bot"


def test_missing_agent_is_auto_created(sdk):
    from memanto import Memanto

    sdk.get_agent.side_effect = AgentNotFoundError("nope")
    Memanto("bot", pattern="support")

    sdk.create_agent.assert_called_once_with("bot", pattern="support")
    sdk.activate_agent.assert_called_once()


def test_concurrent_create_is_tolerated(sdk):
    from memanto import Memanto

    sdk.get_agent.side_effect = AgentNotFoundError("nope")
    sdk.create_agent.side_effect = AgentAlreadyExistsError("exists")
    Memanto("bot")

    sdk.activate_agent.assert_called_once()


def test_missing_agent_without_auto_create_raises(sdk):
    from memanto import Memanto

    sdk.get_agent.side_effect = AgentNotFoundError("nope")
    with pytest.raises(AgentNotFoundError):
        Memanto("bot", auto_create=False)
    sdk.create_agent.assert_not_called()


def test_remember_defaults_title_from_content(sdk):
    from memanto import Memanto

    m = Memanto("bot")
    m.remember("short")
    assert sdk.remember.call_args.kwargs["title"] == "short"

    long = "x" * 60
    m.remember(long, type="fact", tags=["a"])
    kwargs = sdk.remember.call_args.kwargs
    assert kwargs["title"] == "x" * 50 + "..."
    assert kwargs["memory_type"] == "fact"
    assert kwargs["tags"] == ["a"]
    assert sdk.remember.call_args.args == ("bot",)


def test_read_methods_are_scoped_to_agent(sdk):
    from memanto import Memanto

    m = Memanto("bot")
    m.recall("q", limit=3, type=["fact"])
    sdk.recall.assert_called_once_with(
        "bot", "q", limit=3, type=["fact"], tags=None, min_similarity=None
    )
    m.answer("why?", temperature=0.1)
    assert sdk.answer.call_args.args == ("bot", "why?")
    assert sdk.answer.call_args.kwargs["temperature"] == 0.1
    m.recall_recent(limit=5)
    sdk.recall_recent.assert_called_once_with("bot", limit=5, type=None, tags=None)
    m.update_memory("m1", content="new")
    sdk.update_memory.assert_called_once_with("bot", "m1", {"content": "new"})
    m.delete_memory("m1")
    sdk.delete_memory.assert_called_once_with("bot", "m1")


def test_resolve_api_key_on_prem_needs_no_key(monkeypatch):
    from memanto.app.clients.backend import Backend
    from memanto.client import _resolve_api_key

    monkeypatch.delenv("MOORCHEH_API_KEY", raising=False)
    with patch("memanto.cli.config.manager.ConfigManager") as cm:
        cm.return_value.get_backend.return_value = Backend.ON_PREM
        assert _resolve_api_key(None) == "on-prem"


def test_resolve_api_key_cloud(monkeypatch):
    import os

    from memanto.app.clients.backend import Backend
    from memanto.client import _resolve_api_key

    monkeypatch.delenv("MOORCHEH_API_KEY", raising=False)
    with patch("memanto.cli.config.manager.ConfigManager") as cm:
        cm.return_value.get_backend.return_value = Backend.CLOUD
        cm.return_value.get_api_key.return_value = None
        with pytest.raises(ValueError, match="No Moorcheh API key"):
            _resolve_api_key(None)

        assert _resolve_api_key("explicit") == "explicit"
        assert os.environ["MOORCHEH_API_KEY"] == "explicit"

        cm.return_value.get_api_key.return_value = "saved"
        monkeypatch.delenv("MOORCHEH_API_KEY")
        assert _resolve_api_key(None) == "saved"
