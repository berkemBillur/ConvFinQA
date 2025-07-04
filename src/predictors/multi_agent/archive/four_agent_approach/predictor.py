from __future__ import annotations

"""Public predictor class that integrates the new Crew-based multi-agent system
with the existing ConvFinQA evaluator protocol.
"""

from typing import List, Dict, Union, Optional
import logging
import time

from ...evaluation.executor import execute_dsl_program
from ...utils.config import Config
from ...data.models import ConvFinQARecord
from ...utils.enhanced_tracker import get_enhanced_tracker

from .tasks import Context
from .orchestrator import ConvFinQACrew

logger = logging.getLogger(__name__)


class ConvFinQAMultiAgentPredictorV2:  # separate name to avoid immediate collision
    """CrewAI-powered multi-agent predictor (new architecture)."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.enhanced_tracker = get_enhanced_tracker()
        self.config_hash = f"multi_agent_v2_{hash(str(self.config))}"

        self.crew_wrapper = ConvFinQACrew(self.config)

    # ------------------------------------------------------------------
    def predict_turn(
        self,
        record: ConvFinQARecord,
        turn_index: int,
        conversation_history: List[Dict[str, str]],
    ) -> Union[float, str]:
        """Predict answer for a single conversation turn."""
        ctx = Context(record=record, turn_index=turn_index, conversation_history=conversation_history)

        start = time.time()
        success = False
        try:
            crew_output = self.crew_wrapper.run(ctx)

            processed = self._postprocess_output(ctx.question, crew_output)
            success = True
            return processed
        finally:
            # Individual execution tracking now handled by enhanced_benchmark_multi_agent.py
            pass

    # ------------------------------------------------------------------
    def _postprocess_output(self, question: str, raw: str) -> Union[float, str]:
        """Execute DSL when possible and fix percentage scale."""
        cleaned = raw.strip()

        # Attempt DSL execution
        if any(cleaned.startswith(op) for op in ("add(", "subtract(", "multiply(", "divide(")):
            try:
                result = execute_dsl_program(cleaned)
                if isinstance(result, (int, float)):
                    cleaned = result
            except Exception:
                # leave as-is
                pass

        # Percentage scaling: if expected decimal but model returned percentage
        if isinstance(cleaned, (int, float)):
            return cleaned

        # If the answer contains a percent sign, or the question refers to a percentage, decide whether to
        # convert to decimal form. We only divide by 100 when the raw value clearly represents a whole
        # percentage (e.g. 84.88%) rather than an already-decimal value (e.g. 0.8488).

        import re
        percent_match = re.match(r"-?\d+(?:\.\d+)?%?", cleaned)

        if percent_match and ("%" in cleaned or "percentage" in question.lower()):
            try:
                val = float(percent_match.group(0).replace("%", ""))

                # Heuristic: if the absolute numeric value is >= 1, treat it as a whole percentage and
                # convert to decimal; otherwise assume it is already in decimal form and leave unchanged.
                if abs(val) >= 1:
                    return val / 100.0
                else:
                    return val
            except Exception:
                return cleaned

        return cleaned 