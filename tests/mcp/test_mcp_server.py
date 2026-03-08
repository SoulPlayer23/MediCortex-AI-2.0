"""
Tests for the MCP Server: tool listing and tool invocation.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


class TestMCPToolListing:
    """Tests for the MCP server handle_list_tools."""

    @pytest.mark.asyncio
    async def test_lists_all_tools(self):
        from tools.mcp_server import handle_list_tools
        tools = await handle_list_tools()
        assert len(tools) >= 10  # We have 12+ tools
        tool_names = [t.name for t in tools]
        # Verify key tools are listed
        assert "search_pubmed" in tool_names
        assert "check_drug_interactions" in tool_names
        assert "recommend_drugs" in tool_names
        assert "retrieve_patient_records" in tool_names
        assert "analyze_patient_history" in tool_names
        assert "extract_document_text" in tool_names
        assert "extract_image_findings" in tool_names
        assert "analyze_report" in tool_names

    @pytest.mark.asyncio
    async def test_tools_have_schemas(self):
        from tools.mcp_server import handle_list_tools
        tools = await handle_list_tools()
        for tool in tools:
            assert tool.name, "Tool must have a name"
            assert tool.description, f"Tool '{tool.name}' must have a description"
            assert tool.inputSchema, f"Tool '{tool.name}' must have an input schema"

    @pytest.mark.asyncio
    async def test_tools_have_required_fields(self):
        from tools.mcp_server import handle_list_tools
        tools = await handle_list_tools()
        for tool in tools:
            schema = tool.inputSchema
            assert "type" in schema, f"Tool '{tool.name}' schema missing 'type'"
            assert "properties" in schema, f"Tool '{tool.name}' schema missing 'properties'"


class TestMCPToolInvocation:
    """Tests for the MCP server handle_call_tool."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        from tools.mcp_server import handle_call_tool
        results = await handle_call_tool("nonexistent_tool", {"dummy": "value"})
        assert len(results) == 1
        assert "Unknown tool" in results[0].text or "Error" in results[0].text

    @pytest.mark.asyncio
    async def test_drug_interaction_invocation(self):
        from tools.mcp_server import handle_call_tool

        with patch("tools.drug_interaction_tools.check_drug_interactions") as mock_tool:
            mock_tool.invoke.return_value = "No interactions found."
            results = await handle_call_tool("check_drug_interactions", {
                "medications": "Aspirin",
                "patient_conditions": ""
            })

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_recommend_drugs_invocation(self):
        from tools.mcp_server import handle_call_tool

        with patch("tools.drug_recommendation_tools.recommend_drugs") as mock_tool:
            mock_tool.invoke.return_value = "Metformin recommended."
            results = await handle_call_tool("recommend_drugs", {
                "condition": "Diabetes",
                "query_type": "recommendation"
            })

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_analyze_report_invocation(self):
        from tools.mcp_server import handle_call_tool

        with patch("tools.report_analysis_tools.analyze_report") as mock_tool:
            mock_tool.invoke.return_value = "Hemoglobin is low."
            results = await handle_call_tool("analyze_report", {
                "extracted_content": "Hemoglobin: 10.2",
                "report_type": "lab_report"
            })

        assert len(results) == 1


class TestMCPResources:
    """Tests for MCP server resource listing."""

    @pytest.mark.asyncio
    async def test_lists_agent_cards(self):
        from tools.mcp_server import handle_list_resources
        resources = await handle_list_resources()
        assert len(resources) >= 5
        uris = [r.uri for r in resources]
        # Each agent should have a card resource
        uri_strings = [str(u) for u in uris]
        assert any("pubmed" in u for u in uri_strings)
        assert any("pharmacology" in u for u in uri_strings)
