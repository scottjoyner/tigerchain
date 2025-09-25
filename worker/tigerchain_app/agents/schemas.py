from __future__ import annotations

"""Pydantic schemas for the agent builder API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class KnowledgeBaseUpdate(BaseModel):
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
    metadata: Optional[dict] = None


class KnowledgeBaseRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    name: str
    description: Optional[str]
    tags: List[str]
    document_ids: List[str]
    metadata: dict = Field(default_factory=dict, alias="extra_metadata")


class AgentProfileCreate(BaseModel):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    role_type: str = Field(default="chat")
    system_prompt: Optional[str] = None
    knowledge_base_id: Optional[int] = None
    embedding_documents: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    available_tools: List[str] = Field(default_factory=list)
    requires_human_validation: bool = Field(default=True)
    validator_instructions: Optional[str] = None


class AgentProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    role_type: Optional[str] = None
    system_prompt: Optional[str] = None
    knowledge_base_id: Optional[int] = None
    embedding_documents: Optional[List[str]] = None
    metadata: Optional[dict] = None
    tags: Optional[List[str]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    available_tools: Optional[List[str]] = None
    requires_human_validation: Optional[bool] = None
    validator_instructions: Optional[str] = None


class AgentProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    name: str
    display_name: Optional[str]
    description: Optional[str]
    role: Optional[str]
    role_type: str
    system_prompt: Optional[str]
    knowledge_base_id: Optional[int]
    embedding_documents: List[str]
    metadata: dict = Field(default_factory=dict, alias="extra_metadata")
    tags: List[str]
    provider: Optional[str]
    model: Optional[str]
    temperature: Optional[float]
    available_tools: List[str]
    requires_human_validation: bool
    validator_instructions: Optional[str]


class AgentTeamMemberCreate(BaseModel):
    agent_id: int
    priority: int = 0
    responsibilities: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class AgentTeamCreate(BaseModel):
    name: str
    description: Optional[str] = None
    orchestrator_agent: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    members: List[AgentTeamMemberCreate] = Field(default_factory=list)


class AgentTeamUpdate(BaseModel):
    description: Optional[str] = None
    orchestrator_agent: Optional[str] = None
    metadata: Optional[dict] = None
    tags: Optional[List[str]] = None
    members: Optional[List[AgentTeamMemberCreate]] = None


class AgentTeamMemberRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    agent_id: int
    priority: int
    responsibilities: List[str]
    metadata: dict = Field(default_factory=dict, alias="extra_metadata")


class AgentTeamRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    name: str
    description: Optional[str]
    orchestrator_agent: Optional[str]
    metadata: dict = Field(default_factory=dict, alias="extra_metadata")
    tags: List[str]
    members: List[AgentTeamMemberRead]


class ToolTemplateCreate(BaseModel):
    name: str
    description: Optional[str] = None
    specification: dict = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_mcp: bool = True
    metadata: dict = Field(default_factory=dict)


class ToolTemplateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    name: str
    description: Optional[str]
    specification: dict
    tags: List[str]
    is_mcp: bool
    metadata: dict = Field(default_factory=dict, alias="extra_metadata")


class ToolAssignmentRequest(BaseModel):
    tool_id: int
    scope: str = Field(default="global")
    metadata: dict = Field(default_factory=dict)


class ValidatorProfileCreate(BaseModel):
    name: str
    role_description: Optional[str] = None
    contact: Optional[str] = None
    instructions: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    is_active: bool = True


class ValidatorProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    name: str
    role_description: Optional[str]
    contact: Optional[str]
    instructions: Optional[str]
    metadata: dict = Field(default_factory=dict, alias="extra_metadata")
    is_active: bool


class RegistrySnapshotRead(BaseModel):
    registry: Dict[str, Any]
    knowledge_bases: List[Dict[str, Any]]
    teams: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    validator: Optional[Dict[str, Any]]
    orchestrator_agent: Optional[str]
