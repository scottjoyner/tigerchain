from __future__ import annotations

"""Utilities for loading agent builder configuration from persistent storage."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from ..config import Settings
from .models import (
    AgentProfile,
    AgentTeam,
    AgentTeamMember,
    AgentToolAssignment,
    KnowledgeBase,
    ToolTemplate,
    ValidatorProfile,
)


@dataclass
class AgentRegistrySnapshot:
    """Aggregated view of the agent builder state."""

    registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    knowledge_bases: List[Dict[str, Any]] = field(default_factory=list)
    teams: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    validator: Optional[Dict[str, Any]] = None
    orchestrator_agent: Optional[str] = None


class AgentRegistryLoader:
    """Builds registry snapshots from SQLModel storage."""

    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    def build_snapshot(self) -> AgentRegistrySnapshot:
        knowledge_bases = self._load_knowledge_bases()
        tools = self._load_tools()
        tool_map = {tool["id"]: tool for tool in tools}
        validator = self._load_validator()
        profiles = self._load_agent_profiles()
        assignments = self._load_assignments()
        teams = self._load_teams()

        registry: Dict[str, Dict[str, Any]] = {}
        orchestrator_agent: Optional[str] = None

        kb_map = {kb["id"]: kb for kb in knowledge_bases}
        agent_tools: Dict[int, List[Dict[str, Any]]] = {}
        for assignment in assignments:
            agent_tools.setdefault(assignment["agent_id"], []).append(
                {
                    "tool_id": assignment["tool_id"],
                    "tool": tool_map.get(assignment["tool_id"]),
                    "scope": assignment["scope"],
                    "metadata": assignment["metadata"],
                }
            )

        for profile in profiles:
            kb_details = kb_map.get(profile["knowledge_base_id"]) if profile["knowledge_base_id"] else None
            config: Dict[str, Any] = {
                "provider": profile["provider"] or self.settings.llm_provider,
                "model": profile["model"] or self.settings.llm_model,
                "temperature": profile["temperature"] if profile["temperature"] is not None else self.settings.llm_temperature,
                "role": profile["role"],
                "role_type": profile["role_type"],
                "system_prompt": profile["system_prompt"],
                "knowledge_base": kb_details,
                "metadata": profile["metadata"],
                "tags": profile["tags"],
                "embedding_documents": profile["embedding_documents"],
                "available_tools": profile["available_tools"],
                "requires_human_validation": profile["requires_human_validation"],
                "validator_instructions": profile["validator_instructions"],
            }
            if profile["id"] in agent_tools:
                config["tools"] = agent_tools[profile["id"]]
            registry[profile["name"]] = config
            if profile["role_type"] == "orchestrator":
                orchestrator_agent = profile["name"]

        snapshot = AgentRegistrySnapshot(
            registry=registry,
            knowledge_bases=knowledge_bases,
            teams=teams,
            tools=tools,
            validator=validator,
            orchestrator_agent=orchestrator_agent,
        )
        return snapshot

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------
    def _load_knowledge_bases(self) -> List[Dict[str, Any]]:
        results = self.session.exec(select(KnowledgeBase)).all()
        payload: List[Dict[str, Any]] = []
        for kb in results:
            payload.append(
                {
                    "id": kb.id,
                    "name": kb.name,
                    "description": kb.description,
                    "tags": kb.tags,
                    "document_ids": kb.document_ids,
                    "metadata": kb.extra_metadata,
                }
            )
        return payload

    def _load_tools(self) -> List[Dict[str, Any]]:
        results = self.session.exec(select(ToolTemplate)).all()
        payload: List[Dict[str, Any]] = []
        for tool in results:
            payload.append(
                {
                    "id": tool.id,
                    "name": tool.name,
                    "description": tool.description,
                    "specification": tool.specification,
                    "tags": tool.tags,
                    "is_mcp": tool.is_mcp,
                    "metadata": tool.extra_metadata,
                }
            )
        return payload

    def _load_validator(self) -> Optional[Dict[str, Any]]:
        result = self.session.exec(
            select(ValidatorProfile).where(ValidatorProfile.is_active == True).order_by(ValidatorProfile.updated_at.desc())
        ).first()
        if not result:
            return None
        return {
            "id": result.id,
            "name": result.name,
            "role_description": result.role_description,
            "contact": result.contact,
            "instructions": result.instructions,
            "metadata": result.extra_metadata,
        }

    def _load_agent_profiles(self) -> List[Dict[str, Any]]:
        results = self.session.exec(select(AgentProfile)).all()
        payload: List[Dict[str, Any]] = []
        for profile in results:
            payload.append(
                {
                    "id": profile.id,
                    "name": profile.name,
                    "display_name": profile.display_name,
                    "description": profile.description,
                    "role": profile.role,
                    "role_type": profile.role_type,
                    "system_prompt": profile.system_prompt,
                    "knowledge_base_id": profile.knowledge_base_id,
                    "embedding_documents": profile.embedding_documents,
                    "metadata": profile.extra_metadata,
                    "tags": profile.tags,
                    "provider": profile.provider,
                    "model": profile.model,
                    "temperature": profile.temperature,
                    "available_tools": profile.available_tools,
                    "requires_human_validation": profile.requires_human_validation,
                    "validator_instructions": profile.validator_instructions,
                }
            )
        return payload

    def _load_assignments(self) -> List[Dict[str, Any]]:
        results = self.session.exec(select(AgentToolAssignment)).all()
        payload: List[Dict[str, Any]] = []
        for assignment in results:
            payload.append(
                {
                    "id": assignment.id,
                    "agent_id": assignment.agent_id,
                    "tool_id": assignment.tool_id,
                    "scope": assignment.scope,
                    "metadata": assignment.extra_metadata,
                }
            )
        return payload

    def _load_teams(self) -> List[Dict[str, Any]]:
        teams = self.session.exec(select(AgentTeam)).all()
        team_payload: List[Dict[str, Any]] = []
        members = self.session.exec(select(AgentTeamMember)).all()
        members_by_team: Dict[int, List[Dict[str, Any]]] = {}
        for member in members:
            members_by_team.setdefault(member.team_id, []).append(
                {
                    "id": member.id,
                    "agent_id": member.agent_id,
                    "priority": member.priority,
                    "responsibilities": member.responsibilities,
                    "metadata": member.extra_metadata,
                }
            )
        for team in teams:
            team_payload.append(
                {
                    "id": team.id,
                    "name": team.name,
                    "description": team.description,
                    "orchestrator_agent": team.orchestrator_agent,
                    "metadata": team.extra_metadata,
                    "tags": team.tags,
                    "members": members_by_team.get(team.id, []),
                }
            )
        return team_payload
