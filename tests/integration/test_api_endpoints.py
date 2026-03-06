"""
Tests for FastAPI endpoints: /health, /.well-known/agent-cards
"""
import pytest


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.fixture
    def client(self):
        from orchestrator import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_health_lists_agents(self, client):
        response = client.get("/health")
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)
        assert len(data["agents"]) > 0


class TestAgentCardsEndpoint:
    """Tests for the /.well-known/agent-cards endpoint."""

    @pytest.fixture
    def client(self):
        from orchestrator import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_returns_all_cards(self, client):
        response = client.get("/.well-known/agent-cards")
        assert response.status_code == 200
        data = response.json()
        # API returns dict keyed by agent name
        assert isinstance(data, (list, dict))
        if isinstance(data, dict):
            assert len(data) == 5
        else:
            assert len(data) == 5

    def test_cards_have_required_fields(self, client):
        response = client.get("/.well-known/agent-cards")
        data = response.json()
        if isinstance(data, dict):
            for name, card in data.items():
                assert "name" in card
                assert "description" in card
                assert "capabilities" in card
        else:
            for card in data:
                assert "name" in card
                assert "description" in card
                assert "capabilities" in card
