from __future__ import annotations

"""Database models supporting the configurable agent builder."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


class KnowledgeBase(SQLModel, table=True):
    """Curated collections of documents or embeddings for an agent domain."""

    __tablename__ = "knowledge_bases"
    __table_args__ = (UniqueConstraint("name", name="uq_knowledge_bases_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    document_ids: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    extra_metadata: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    created_by: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class AgentProfile(SQLModel, table=True):
    """Configurable RAG agent definition exposed through the builder."""

    __tablename__ = "agent_profiles"
    __table_args__ = (UniqueConstraint("name", name="uq_agent_profiles_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    display_name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    role: Optional[str] = Field(default=None)
    role_type: str = Field(default="chat", index=True)
    system_prompt: Optional[str] = Field(default=None)
    knowledge_base_id: Optional[int] = Field(default=None, foreign_key="knowledge_bases.id")
    embedding_documents: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    extra_metadata: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    provider: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    available_tools: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    requires_human_validation: bool = Field(default=True)
    validator_instructions: Optional[str] = Field(default=None)
    created_by: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class AgentTeam(SQLModel, table=True):
    """Container for multi-agent teams orchestrated together."""

    __tablename__ = "agent_teams"
    __table_args__ = (UniqueConstraint("name", name="uq_agent_teams_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    orchestrator_agent: Optional[str] = Field(default=None, index=True)
    extra_metadata: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_by: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class AgentTeamMember(SQLModel, table=True):
    """Links an agent profile to a team with optional responsibilities."""

    __tablename__ = "agent_team_members"
    __table_args__ = (UniqueConstraint("team_id", "agent_id", name="uq_agent_team_membership"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    team_id: int = Field(foreign_key="agent_teams.id", index=True)
    agent_id: int = Field(foreign_key="agent_profiles.id", index=True)
    priority: int = Field(default=0, index=True)
    responsibilities: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    extra_metadata: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class ToolTemplate(SQLModel, table=True):
    """Reusable MCP tool templates shared across agents."""

    __tablename__ = "tool_templates"
    __table_args__ = (UniqueConstraint("name", name="uq_tool_templates_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    specification: dict = Field(default_factory=dict, sa_column=Column(JSON))
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    is_mcp: bool = Field(default=True, index=True)
    extra_metadata: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    created_by: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class AgentToolAssignment(SQLModel, table=True):
    """Associates tool templates with agents and captures usage scope."""

    __tablename__ = "agent_tool_assignments"
    __table_args__ = (UniqueConstraint("agent_id", "tool_id", name="uq_agent_tool_assignment"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: int = Field(foreign_key="agent_profiles.id", index=True)
    tool_id: int = Field(foreign_key="tool_templates.id", index=True)
    scope: str = Field(default="global")
    extra_metadata: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class ValidatorProfile(SQLModel, table=True):
    """Represents a human validator integrated into the orchestration flow."""

    __tablename__ = "validator_profiles"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    role_description: Optional[str] = Field(default=None)
    contact: Optional[str] = Field(default=None)
    instructions: Optional[str] = Field(default=None)
    extra_metadata: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    is_active: bool = Field(default=True, index=True)
    created_by: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
