from shared.tools import TOOL_REGISTRY
from shared.tools import TOOLS


def test_search_tool_registered():

    assert "search_documents" in TOOL_REGISTRY


def test_search_tool_declared():

    names = {
        tool["name"]
        for tool in TOOLS
    }

    assert "search_documents" in names