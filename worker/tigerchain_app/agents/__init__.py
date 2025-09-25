"""Multi-agent orchestration and builder utilities."""

from importlib import import_module
from typing import Any

__all__ = [
    "AgentOrchestrator",
    "AgentQueryResult",
    "QueryContext",
    "AgentRegistryLoader",
    "AgentRegistrySnapshot",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy import indirection
    if name in {"AgentOrchestrator", "AgentQueryResult", "QueryContext"}:
        module = import_module(".orchestrator", __name__)
        return getattr(module, name)
    if name in {"AgentRegistryLoader", "AgentRegistrySnapshot"}:
        module = import_module(".registry", __name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
