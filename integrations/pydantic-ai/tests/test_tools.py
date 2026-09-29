from unittest.mock import MagicMock

from memanto_pydantic_ai.tools import MemantoSetup, create_memanto_tools

from memanto.app.utils.errors import AgentAlreadyExistsError


def test_memanto_setup_create_new_agent():
    setup = MemantoSetup(api_key="test_key")
    setup.client = MagicMock()

    client = setup.setup(agent_id="test-agent", pattern="tool", duration_hours=4)

    assert client == setup.client
    setup.client.create_agent.assert_called_once_with(
        agent_id="test-agent", pattern="tool", description=None
    )
    setup.client.activate_agent.assert_called_once_with("test-agent", duration_hours=4)


def test_memanto_setup_agent_exists():
    setup = MemantoSetup(api_key="test_key")
    setup.client = MagicMock()
    setup.client.create_agent.side_effect = AgentAlreadyExistsError("test-agent")

    # Should catch the error and still activate
    client = setup.setup(agent_id="test-agent")

    assert client == setup.client
    setup.client.create_agent.assert_called_once()
    setup.client.activate_agent.assert_called_once_with("test-agent", duration_hours=6)


def test_memanto_setup_teardown():
    setup = MemantoSetup(api_key="test_key")
    setup.client = MagicMock()

    setup.teardown("test-agent")
    setup.client.deactivate_agent.assert_called_once_with("test-agent")


def _tools_by_name(client, agent_id="test-agent"):
    return {t.name: t for t in create_memanto_tools(client, agent_id=agent_id)}


def test_create_memanto_tools_returns_all_three():
    tools = _tools_by_name(MagicMock())
    assert set(tools.keys()) == {"memanto_remember", "memanto_recall", "memanto_answer"}


def test_memanto_remember_tool():
    client = MagicMock()
    client.remember.return_value = {"memory_id": "mem-123"}

    tools = _tools_by_name(client)
    result = tools["memanto_remember"].function(
        memory_type="fact",
        title="Test Fact",
        content="The content",
        confidence=0.9,
        tags="tag1, tag2 ",
    )

    assert "Memory stored successfully" in result
    assert "mem-123" in result
    assert "fact" in result

    client.remember.assert_called_once_with(
        agent_id="test-agent",
        memory_type="fact",
        title="Test Fact",
        content="The content",
        confidence=0.9,
        tags=["tag1", "tag2"],
        source="pydantic-ai-agent",
        provenance="explicit_statement",
    )


def test_memanto_recall_tool_success():
    client = MagicMock()
    client.recall.return_value = {
        "memories": [
            {
                "id": "mem-123",
                "type": "fact",
                "title": "Fact 1",
                "content": "Content 1",
                "confidence": 0.8,
                "tags": ["test"],
            }
        ]
    }

    tools = _tools_by_name(client)
    result = tools["memanto_recall"].function(
        query="test query", limit=5, memory_types="fact", min_similarity=0.7
    )

    assert "Found 1 memories for 'test query'" in result
    assert "Fact 1" in result
    assert "Content 1" in result

    client.recall.assert_called_once_with(
        agent_id="test-agent",
        query="test query",
        limit=5,
        type=["fact"],
        min_similarity=0.7,
    )


def test_memanto_recall_tool_empty():
    client = MagicMock()
    client.recall.return_value = {"memories": []}

    tools = _tools_by_name(client)
    result = tools["memanto_recall"].function(query="test query")

    assert result == "No memories found for query: 'test query'"


def test_memanto_answer_tool_success():
    client = MagicMock()
    client.answer.return_value = {
        "answer": "This is the answer.",
        "sources": [{"id": "mem-1"}],
    }

    tools = _tools_by_name(client)
    result = tools["memanto_answer"].function(question="What is this?")

    assert "Answer: This is the answer." in result
    assert "Based on 1 memory source(s)." in result

    client.answer.assert_called_once_with(agent_id="test-agent", question="What is this?")


def test_memanto_answer_tool_no_answer():
    client = MagicMock()
    client.answer.return_value = {}

    tools = _tools_by_name(client)
    result = tools["memanto_answer"].function(question="What is this?")

    assert "Answer: No answer could be generated." in result
