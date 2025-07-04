from __future__ import annotations

"""High-level Crew wrapper used by the new multi-agent predictor."""

from pathlib import Path
from typing import Dict, Any
import logging

try:
    from crewai import Crew, Process
except ImportError as exc:  # pragma: no cover
    raise ImportError("CrewAI must be installed to use ConvFinQACrew") from exc

from .agents import build_agents
from .tasks import Context, build_tasks
from .tracer import build_tracer, JsonlTracer
from ...utils.config import Config

_logger = logging.getLogger(__name__)


class ConvFinQACrew:
    """Thin wrapper that owns the CrewAI `Crew` instance."""

    def __init__(self, config: Config, trace_dir: Path | None = None):
        self.config = config
        self.agents = build_agents(config)
        self.tracer: JsonlTracer | None = build_tracer(trace_dir)

        manager_agent = self.agents["supervisor"]
        worker_agents = [
            self.agents["extractor"],
            self.agents["calculator"],
            self.agents["validator"],
        ]

        # Create empty Crew; tasks are supplied per-run
        self.crew = Crew(
            agents=worker_agents,  # type: ignore[arg-type]
            tasks=[],
            process=Process.sequential,  # Use sequential process for pre-assigned task flow
            verbose=config.get("crewai", {}).get("verbose", True),
            callbacks=[self.tracer] if self.tracer else None,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, ctx: Context) -> str:
        """Execute the multi-agent workflow and return answer (string)."""
        self.crew.tasks = build_tasks(ctx, self.agents)
        result = self.crew.kickoff()
        # CrewAI returns last task result when hierarchical; cast to str
        return str(result)

    # ------------------------------------------------------------------
    def shutdown(self):
        if self.tracer:
            self.tracer.close() 