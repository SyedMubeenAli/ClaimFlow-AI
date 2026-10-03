"""
Wires the app together: settings, logging, database, AI provider, agents and
orchestrator. The UI builds this once and reuses it. Tests build their own with
temporary folders.

Demo Mode and Live AI can be switched while the app runs (`set_demo_mode`), so a
presenter whose network drops can flip to the offline workflow without a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .agents import default_agents
from .config import KEY_VARS, Settings
from .db import Database
from .llm import LLMProvider, build_provider
from .logging_setup import get_logger, setup_logging
from .orchestrator import WorkflowOrchestrator
from .rag import PolicyIndexService, build_embedder


def _embedder_for(settings: Settings):
    # Demo Mode must not depend on a model download, so it uses the offline embedder
    # unless a backend was chosen explicitly.
    backend = "hashing" if settings.demo_mode and settings.embedding_backend == "auto" else settings.embedding_backend
    return build_embedder(backend, settings.embedding_model)


@dataclass
class AppContainer:
    settings: Settings
    db: Database
    llm: LLMProvider
    orchestrator: WorkflowOrchestrator
    rag: PolicyIndexService

    def set_demo_mode(self, enabled: bool) -> str | None:
        """
        Switch between Demo Mode (scripted offline model) and Live AI. Returns an explanation
        if the switch is not possible, and changes nothing in that case.
        """
        if enabled == self.settings.demo_mode:
            return None
        if not enabled:
            provider = self.settings.provider
            if provider == "fake":
                return "No live provider is configured. Set LLM_PROVIDER and its API key in .env, then restart."
            if not self.settings.has_api_key(provider):
                return f"{KEY_VARS[provider]} is not set, so Live AI is not available. Add it to .env and restart."
        settings = replace(self.settings, demo_mode=enabled)
        self.settings = settings
        self.llm = build_provider(settings)
        self.rag.embedder = _embedder_for(settings)
        self.orchestrator.llm, self.orchestrator.settings = self.llm, settings
        get_logger("container").info("Switched to %s", "Demo Mode" if enabled else "Live AI")
        return None


def build_container(env: Mapping[str, str] | None = None) -> AppContainer:
    settings = Settings.from_env(env)
    setup_logging(settings.log_level, settings.log_dir, settings.secrets())
    db = Database(settings.database_path)
    db.init()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    llm = build_provider(settings)
    rag = PolicyIndexService(settings.upload_dir, _embedder_for(settings), settings.index_dir)
    orchestrator = WorkflowOrchestrator(db, llm, settings, default_agents(), rag=rag)
    get_logger("container").info("ClaimFlow ready: provider=%s model=%s demo=%s",
                                 settings.provider, settings.model, settings.demo_mode)
    return AppContainer(settings, db, llm, orchestrator, rag)
