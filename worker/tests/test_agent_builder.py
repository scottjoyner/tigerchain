from __future__ import annotations

from sqlmodel import Field, Session, SQLModel, create_engine

class _User(SQLModel, table=True):  # pragma: no cover - minimal user table for FK
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)

from tigerchain_app.agents.registry import AgentRegistryLoader
from tigerchain_app.agents.service import AgentBuilderService
from tigerchain_app.config import Settings


def create_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_agent_builder_registry_snapshot_contains_metadata():
    settings = Settings()
    with create_session() as session:
        service = AgentBuilderService(session)
        kb = service.create_knowledge_base(
            {
                "name": "Product Architecture",
                "description": "Core diagrams",
                "tags": ["architecture"],
                "document_ids": ["doc::alpha"],
            },
            created_by=None,
        )
        agent = service.create_agent_profile(
            {
                "name": "architecture_planner",
                "description": "Plans product evolution",
                "role_type": "team_member",
                "knowledge_base_id": kb.id,
                "requires_human_validation": True,
            },
            created_by=None,
        )
        tool = service.create_tool_template(
            {
                "name": "roadmap-template",
                "description": "Generates roadmap drafts",
                "specification": {"entrypoint": "roadmap"},
            },
            created_by=None,
        )
        service.assign_tool_to_agent(agent.id, tool.id, {"scope": "global"})
        service.upsert_validator(
            {
                "name": "Human Validator",
                "contact": "validator@example.com",
                "is_active": True,
            },
            created_by=None,
        )

        loader = AgentRegistryLoader(session, settings)
        snapshot = loader.build_snapshot()

        assert agent.name in snapshot.registry
        agent_config = snapshot.registry[agent.name]
        assert agent_config["knowledge_base"]["id"] == kb.id
        assert agent_config["requires_human_validation"] is True
        assert snapshot.validator["name"] == "Human Validator"
        assert any(tool_item["name"] == "roadmap-template" for tool_item in snapshot.tools)
