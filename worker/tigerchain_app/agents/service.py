from __future__ import annotations

"""Service layer for the agent builder endpoints."""

from datetime import datetime
from typing import Iterable, List

from sqlmodel import Session, select

from .models import (
    AgentProfile,
    AgentTeam,
    AgentTeamMember,
    AgentToolAssignment,
    KnowledgeBase,
    ToolTemplate,
    ValidatorProfile,
)


def _now() -> datetime:
    return datetime.utcnow()


class AgentBuilderService:
    """Encapsulates persistence logic for agent builder resources."""

    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _normalise_metadata(payload: dict) -> dict:
        if "metadata" in payload and "extra_metadata" not in payload:
            payload["extra_metadata"] = payload.pop("metadata")
        return payload

    # ------------------------------------------------------------------
    # Knowledge bases
    # ------------------------------------------------------------------
    def create_knowledge_base(self, payload: dict, created_by: int | None) -> KnowledgeBase:
        payload = self._normalise_metadata(dict(payload))
        kb = KnowledgeBase(**payload, created_by=created_by)
        self.session.add(kb)
        self.session.commit()
        self.session.refresh(kb)
        return kb

    def update_knowledge_base(self, kb_id: int, payload: dict) -> KnowledgeBase:
        kb = self.session.get(KnowledgeBase, kb_id)
        if kb is None:
            raise ValueError(f"Knowledge base {kb_id} not found")
        payload = self._normalise_metadata(dict(payload))
        for key, value in payload.items():
            if value is not None:
                setattr(kb, key, value)
        kb.updated_at = _now()
        self.session.add(kb)
        self.session.commit()
        self.session.refresh(kb)
        return kb

    def list_knowledge_bases(self) -> List[KnowledgeBase]:
        return self.session.exec(select(KnowledgeBase).order_by(KnowledgeBase.name)).all()

    # ------------------------------------------------------------------
    # Agent profiles
    # ------------------------------------------------------------------
    def create_agent_profile(self, payload: dict, created_by: int | None) -> AgentProfile:
        payload = self._normalise_metadata(dict(payload))
        profile = AgentProfile(**payload, created_by=created_by)
        self.session.add(profile)
        self.session.commit()
        self.session.refresh(profile)
        return profile

    def update_agent_profile(self, agent_id: int, payload: dict) -> AgentProfile:
        profile = self.session.get(AgentProfile, agent_id)
        if profile is None:
            raise ValueError(f"Agent profile {agent_id} not found")
        payload = self._normalise_metadata(dict(payload))
        for key, value in payload.items():
            if value is not None:
                setattr(profile, key, value)
        profile.updated_at = _now()
        self.session.add(profile)
        self.session.commit()
        self.session.refresh(profile)
        return profile

    def list_agent_profiles(self) -> List[AgentProfile]:
        statement = select(AgentProfile).order_by(AgentProfile.name)
        return self.session.exec(statement).all()

    def get_agent_profile(self, agent_id: int) -> AgentProfile | None:
        return self.session.get(AgentProfile, agent_id)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def create_tool_template(self, payload: dict, created_by: int | None) -> ToolTemplate:
        payload = self._normalise_metadata(dict(payload))
        template = ToolTemplate(**payload, created_by=created_by)
        self.session.add(template)
        self.session.commit()
        self.session.refresh(template)
        return template

    def list_tool_templates(self) -> List[ToolTemplate]:
        return self.session.exec(select(ToolTemplate).order_by(ToolTemplate.name)).all()

    def assign_tool_to_agent(self, agent_id: int, tool_id: int, payload: dict) -> AgentToolAssignment:
        payload = self._normalise_metadata(dict(payload))
        assignment = self.session.exec(
            select(AgentToolAssignment).where(
                AgentToolAssignment.agent_id == agent_id,
                AgentToolAssignment.tool_id == tool_id,
            )
        ).first()
        if assignment:
            for key, value in payload.items():
                if value is not None:
                    setattr(assignment, key, value)
            assignment.updated_at = _now()
        else:
            assignment = AgentToolAssignment(agent_id=agent_id, tool_id=tool_id, **payload)
        self.session.add(assignment)
        self.session.commit()
        self.session.refresh(assignment)
        return assignment

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------
    def create_agent_team(self, payload: dict, created_by: int | None) -> AgentTeam:
        members_payload = payload.pop("members", [])
        payload = self._normalise_metadata(dict(payload))
        team = AgentTeam(**payload, created_by=created_by)
        self.session.add(team)
        self.session.commit()
        self.session.refresh(team)
        self._sync_team_members(team, members_payload)
        return team

    def update_agent_team(self, team_id: int, payload: dict) -> AgentTeam:
        team = self.session.get(AgentTeam, team_id)
        if team is None:
            raise ValueError(f"Agent team {team_id} not found")
        members_payload = payload.pop("members", None)
        payload = self._normalise_metadata(dict(payload))
        for key, value in payload.items():
            if value is not None:
                setattr(team, key, value)
        team.updated_at = _now()
        self.session.add(team)
        self.session.commit()
        self.session.refresh(team)
        if members_payload is not None:
            self._sync_team_members(team, members_payload)
        return team

    def list_agent_teams(self) -> List[AgentTeam]:
        return self.session.exec(select(AgentTeam).order_by(AgentTeam.name)).all()

    def list_team_members(self, team_id: int) -> List[AgentTeamMember]:
        statement = select(AgentTeamMember).where(AgentTeamMember.team_id == team_id).order_by(AgentTeamMember.priority)
        return self.session.exec(statement).all()

    def _sync_team_members(self, team: AgentTeam, members_payload: Iterable[dict]) -> None:
        existing_members = {member.id: member for member in self.list_team_members(team.id)}
        seen_ids: set[int] = set()
        order = 0
        for member_payload in members_payload:
            member_payload = self._normalise_metadata(dict(member_payload))
            member_id = member_payload.get("id")
            if member_id and member_id in existing_members:
                member = existing_members[member_id]
                for key, value in member_payload.items():
                    if key == "id":
                        continue
                    setattr(member, key, value)
                member.priority = member_payload.get("priority", order)
                member.updated_at = _now()
                seen_ids.add(member_id)
                self.session.add(member)
            else:
                payload = {key: value for key, value in member_payload.items() if key not in {"id", "priority"}}
                priority = member_payload.get("priority", order)
                member = AgentTeamMember(team_id=team.id, priority=priority, **payload)
                self.session.add(member)
            order += 1
        for member_id, member in existing_members.items():
            if member_id not in seen_ids and member_id is not None:
                self.session.delete(member)
        self.session.commit()

    # ------------------------------------------------------------------
    # Validator
    # ------------------------------------------------------------------
    def upsert_validator(self, payload: dict, created_by: int | None) -> ValidatorProfile:
        payload = self._normalise_metadata(dict(payload))
        validator = self.session.exec(select(ValidatorProfile).order_by(ValidatorProfile.updated_at.desc())).first()
        if validator:
            for key, value in payload.items():
                setattr(validator, key, value)
            validator.updated_at = _now()
        else:
            validator = ValidatorProfile(**payload, created_by=created_by)
        self.session.add(validator)
        self.session.commit()
        self.session.refresh(validator)
        return validator

    def list_validators(self) -> List[ValidatorProfile]:
        return self.session.exec(select(ValidatorProfile).order_by(ValidatorProfile.updated_at.desc())).all()
