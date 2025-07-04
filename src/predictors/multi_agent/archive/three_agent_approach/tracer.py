from __future__ import annotations

"""Lightweight prompt/response tracer for CrewAI executions."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

import logging

# Use runtime import guard – typing only
try:
    from crewai import Agent  # type: ignore
except ImportError:  # pragma: no cover
    from typing import Any as Agent  # type: ignore

_logger = logging.getLogger(__name__)


class JsonlTracer:
    """Persist each agent interaction to a JSONL file for later inspection."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "a", encoding="utf-8")

    # ---------------------------------------------------------------------
    # CrewAI hooks
    # ---------------------------------------------------------------------
    def before_agent(self, agent: Agent, task: str, **kwargs):  # noqa: D401
        self._start_time = time.time()
        self._prompt = task

    def after_agent(self, agent: Agent, response: str, usage: Dict[str, int] | None = None, **kwargs):  # noqa: D401
        record = {
            "ts": time.time(),
            "agent_role": agent.role if hasattr(agent, "role") else str(agent),
            "prompt": self._prompt,
            "response": response,
            "latency": time.time() - self._start_time,
            "usage": usage or {},
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()

    # CrewAI may call __call__ on callback manager; provide noop for safety
    def __call__(self, *args, **kwargs):  # pragma: no cover
        pass


# Convenience helper

def build_tracer(results_dir: Path | None = None) -> JsonlTracer | None:
    if results_dir is None:
        return None
    tracer_path = Path(results_dir) / f"trace_{int(time.time())}.jsonl"
    _logger.info(f"Prompts will be traced to {tracer_path}")
    return JsonlTracer(tracer_path) 