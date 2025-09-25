from __future__ import annotations

"""FastAPI router implementing the agent builder API."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ..auth.database import get_session
from ..auth.models import User
from ..auth.security import get_current_active_user
from ..context import build_context
from ..utils.logging import get_logger
from .registry import AgentRegistryLoader
from .schemas import (
    AgentProfileCreate,
    AgentProfileRead,
    AgentProfileUpdate,
    AgentTeamCreate,
    AgentTeamRead,
    AgentTeamUpdate,
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    KnowledgeBaseUpdate,
    RegistrySnapshotRead,
    ToolAssignmentRequest,
    ToolTemplateCreate,
    ToolTemplateRead,
    ValidatorProfileCreate,
    ValidatorProfileRead,
)
from .service import AgentBuilderService

logger = get_logger(__name__)

router = APIRouter(prefix="/builder", tags=["agent-builder"])


def _refresh_registry(context, session) -> RegistrySnapshotRead:
    loader = AgentRegistryLoader(session, context.settings)
    snapshot = loader.build_snapshot()
    registry_data = dict(context.settings.model_registry)
    registry_data.update(snapshot.registry)
    context.settings.model_registry = registry_data
    context.agent_orchestrator.refresh(snapshot)
    return RegistrySnapshotRead(
        registry=registry_data,
        knowledge_bases=snapshot.knowledge_bases,
        teams=snapshot.teams,
        tools=snapshot.tools,
        validator=snapshot.validator,
        orchestrator_agent=snapshot.orchestrator_agent,
    )


@router.get("/knowledge-bases", response_model=List[KnowledgeBaseRead])
def list_knowledge_bases(session=Depends(get_session), current_user: User = Depends(get_current_active_user)):
    service = AgentBuilderService(session)
    return [KnowledgeBaseRead.model_validate(kb) for kb in service.list_knowledge_bases()]


@router.post("/knowledge-bases", response_model=KnowledgeBaseRead)
def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    session=Depends(get_session),
    context=Depends(build_context),
    current_user: User = Depends(get_current_active_user),
):
    service = AgentBuilderService(session)
    kb = service.create_knowledge_base(payload.model_dump(), created_by=current_user.id)
    _refresh_registry(context, session)
    return KnowledgeBaseRead.model_validate(kb)


@router.patch("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseRead)
def update_knowledge_base(
    kb_id: int,
    payload: KnowledgeBaseUpdate,
    session=Depends(get_session),
    context=Depends(build_context),
    current_user: User = Depends(get_current_active_user),
):
    service = AgentBuilderService(session)
    kb = service.update_knowledge_base(kb_id, payload.model_dump(exclude_unset=True))
    _refresh_registry(context, session)
    return KnowledgeBaseRead.model_validate(kb)


@router.get("/agents", response_model=List[AgentProfileRead])
def list_agents(session=Depends(get_session), current_user: User = Depends(get_current_active_user)):
    service = AgentBuilderService(session)
    return [AgentProfileRead.model_validate(agent) for agent in service.list_agent_profiles()]


@router.post("/agents", response_model=AgentProfileRead)
def create_agent(
    payload: AgentProfileCreate,
    session=Depends(get_session),
    context=Depends(build_context),
    current_user: User = Depends(get_current_active_user),
):
    service = AgentBuilderService(session)
    agent = service.create_agent_profile(payload.model_dump(), created_by=current_user.id)
    _refresh_registry(context, session)
    return AgentProfileRead.model_validate(agent)


@router.patch("/agents/{agent_id}", response_model=AgentProfileRead)
def update_agent(
    agent_id: int,
    payload: AgentProfileUpdate,
    session=Depends(get_session),
    context=Depends(build_context),
    current_user: User = Depends(get_current_active_user),
):
    service = AgentBuilderService(session)
    agent = service.update_agent_profile(agent_id, payload.model_dump(exclude_unset=True))
    _refresh_registry(context, session)
    return AgentProfileRead.model_validate(agent)


@router.post("/agents/{agent_id}/tools", response_model=AgentProfileRead)
def assign_tools_to_agent(
    agent_id: int,
    payload: ToolAssignmentRequest,
    session=Depends(get_session),
    context=Depends(build_context),
    current_user: User = Depends(get_current_active_user),
):
    service = AgentBuilderService(session)
    agent = service.get_agent_profile(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    assignment = service.assign_tool_to_agent(agent_id, payload.tool_id, payload.model_dump(exclude={"tool_id"}))
    logger.info("Assigned tool %s to agent %s", assignment.tool_id, agent_id)
    _refresh_registry(context, session)
    refreshed = service.get_agent_profile(agent_id)
    if refreshed is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentProfileRead.model_validate(refreshed)


@router.get("/tools", response_model=List[ToolTemplateRead])
def list_tools(session=Depends(get_session), current_user: User = Depends(get_current_active_user)):
    service = AgentBuilderService(session)
    return [ToolTemplateRead.model_validate(tool) for tool in service.list_tool_templates()]


@router.post("/tools", response_model=ToolTemplateRead)
def create_tool(
    payload: ToolTemplateCreate,
    session=Depends(get_session),
    context=Depends(build_context),
    current_user: User = Depends(get_current_active_user),
):
    service = AgentBuilderService(session)
    tool = service.create_tool_template(payload.model_dump(), created_by=current_user.id)
    _refresh_registry(context, session)
    return ToolTemplateRead.model_validate(tool)


@router.get("/teams", response_model=List[AgentTeamRead])
def list_teams(session=Depends(get_session), current_user: User = Depends(get_current_active_user)):
    service = AgentBuilderService(session)
    teams = []
    for team in service.list_agent_teams():
        members = service.list_team_members(team.id)
        team_dict = team.model_dump()
        team_dict["members"] = [member.model_dump() for member in members]
        teams.append(team_dict)
    return [AgentTeamRead.model_validate(team) for team in teams]


@router.post("/teams", response_model=AgentTeamRead)
def create_team(
    payload: AgentTeamCreate,
    session=Depends(get_session),
    context=Depends(build_context),
    current_user: User = Depends(get_current_active_user),
):
    service = AgentBuilderService(session)
    team = service.create_agent_team(payload.model_dump(), created_by=current_user.id)
    members = service.list_team_members(team.id)
    _refresh_registry(context, session)
    data = team.model_dump()
    data["members"] = [member.model_dump() for member in members]
    return AgentTeamRead.model_validate(data)


@router.patch("/teams/{team_id}", response_model=AgentTeamRead)
def update_team(
    team_id: int,
    payload: AgentTeamUpdate,
    session=Depends(get_session),
    context=Depends(build_context),
    current_user: User = Depends(get_current_active_user),
):
    service = AgentBuilderService(session)
    team = service.update_agent_team(team_id, payload.model_dump(exclude_unset=True))
    members = service.list_team_members(team.id)
    _refresh_registry(context, session)
    data = team.model_dump()
    data["members"] = [member.model_dump() for member in members]
    return AgentTeamRead.model_validate(data)


@router.get("/validator", response_model=List[ValidatorProfileRead])
def list_validator_profiles(session=Depends(get_session), current_user: User = Depends(get_current_active_user)):
    service = AgentBuilderService(session)
    return [ValidatorProfileRead.model_validate(v) for v in service.list_validators()]


@router.post("/validator", response_model=ValidatorProfileRead)
def upsert_validator(
    payload: ValidatorProfileCreate,
    session=Depends(get_session),
    context=Depends(build_context),
    current_user: User = Depends(get_current_active_user),
):
    service = AgentBuilderService(session)
    validator = service.upsert_validator(payload.model_dump(), created_by=current_user.id)
    _refresh_registry(context, session)
    return ValidatorProfileRead.model_validate(validator)


@router.get("/snapshot", response_model=RegistrySnapshotRead)
def get_registry_snapshot(session=Depends(get_session), context=Depends(build_context), current_user: User = Depends(get_current_active_user)):
    snapshot = AgentRegistryLoader(session, context.settings).build_snapshot()
    registry_data = dict(context.settings.model_registry)
    registry_data.update(snapshot.registry)
    return RegistrySnapshotRead(
        registry=registry_data,
        knowledge_bases=snapshot.knowledge_bases,
        teams=snapshot.teams,
        tools=snapshot.tools,
        validator=snapshot.validator,
        orchestrator_agent=snapshot.orchestrator_agent,
    )
