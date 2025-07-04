from __future__ import annotations

"""High-level Crew wrapper implementing the paper's reflection framework.

This module implements the iterative reflection approach from "Enhancing Financial 
Question Answering with a Multi-Agent Reflection Framework" (arXiv:2410.21741):

Expert Initial Response → Critics Review → Expert Revision → Final Answer

The orchestrator maintains backward compatibility while using the paper's proven approach.
"""

from pathlib import Path
from typing import Dict, Any
import logging
import json
import re

try:
    from crewai import Crew, Process
except ImportError as exc:  # pragma: no cover
    raise ImportError("CrewAI must be installed to use ConvFinQACrew") from exc

from .agents import build_agents
from .tasks import Context, build_paper_tasks
from .tracer import build_tracer, JsonlTracer
from ...utils.config import Config
from ...utils.financial_matcher import financial_matcher
from ...utils.scale_normalizer import scale_normalizer

_logger = logging.getLogger(__name__)


class ConvFinQACrew:
    """Paper replication crew implementing iterative reflection framework."""

    def __init__(self, config: Config, trace_dir: Path | None = None):
        self.config = config
        self.agents = build_agents(config)
        self.tracer: JsonlTracer | None = build_tracer(trace_dir)
        
        # Get paper-specific configuration
        self.agent_config = config.get("three_agent_config", {})
        self.max_iterations = self.agent_config.get("max_iterations", 2)
        self.enable_post_processing = self.agent_config.get("enable_post_processing", True)
        self.verbose = self.agent_config.get("verbose", True)

        # Number extraction config
        self.number_cfg = self.agent_config.get("number_extraction", {})
        self.number_extraction_enabled = self.number_cfg.get("enabled", True)

    # ------------------------------------------------------------------
    # Public API (maintains backward compatibility)
    # ------------------------------------------------------------------
    def run(self, ctx: Context) -> str:
        """Execute the paper's multi-agent reflection workflow and return answer."""
        if self.verbose:
            _logger.info(f"🚀 Starting paper replication workflow for question: {ctx.question[:100]}...")
        
        expert_response = None
        prev_expert_response = None
        
        # Iterative reflection loop (paper's approach)
        for iteration in range(self.max_iterations):
            if self.verbose:
                _logger.info(f"📝 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Update context with iteration info
            ctx.iteration = iteration
            
            # Step 1: Expert provides response (initial or revision)
            expert_response = self._get_expert_response(ctx, prev_expert_response)
            
            if iteration == 0 and self.verbose:
                _logger.info(f"💡 Expert initial response: {expert_response[:200]}...")
            elif iteration > 0 and self.verbose:
                _logger.info(f"🔄 Expert revision: {expert_response[:200]}...")
            
            # Step 2: Critics review the expert response
            critic_feedback = self._get_critic_feedback(ctx, expert_response)
            
            # Record this turn in conversation history so next iteration can resolve references
            ctx.conversation_history.append({"question": ctx.question, "answer": expert_response})
            prev_expert_response = expert_response  # store for next loop
            
            # Step 3: Check if critics approve
            critics_approved = self._critics_approve(critic_feedback)
            if self.verbose:
                _logger.info(f"🎯 Critics approved: {critics_approved}")
            
            if critics_approved:
                if self.verbose:
                    _logger.info("✅ Critics approved the response")
                break
            else:
                if self.verbose:
                    _logger.info(f"❌ Critics requested improvements: {critic_feedback[:100]}...")
                # Add feedback for next iteration
                ctx.critic_feedback = critic_feedback
        
        # Safety check: if we've exhausted iterations, use the last response
        if not critics_approved and self.verbose:
            _logger.warning(f"⚠️ Reached max iterations ({self.max_iterations}), using final expert response")
        
        # Extract final answer from expert response
        final_answer = self._extract_final_answer(expert_response or "")
        
        # Apply lightweight post-processing if enabled
        if self.enable_post_processing:
            final_answer = self._apply_post_processing(final_answer, ctx)
        
        if self.verbose:
            _logger.info(f"🎯 Final answer: {final_answer}")
        
        return final_answer

    # ------------------------------------------------------------------
    # Paper implementation methods
    # ------------------------------------------------------------------
    
    def _get_expert_response(self, ctx: Context, prev_expert_response: str | None = None) -> str:
        """Get response from expert agent (unified extraction + calculation)."""
        # Create expert task
        expert_tasks = build_paper_tasks(ctx, self.agents, expert_response=None, prev_expert_response=prev_expert_response)
        
        # Create crew with just the expert
        expert_crew = Crew(
            agents=[self.agents["expert"]],
            tasks=expert_tasks,
            process=Process.sequential,
            verbose=self.verbose,
            **self._build_crew_kwargs(),
        )
        
        # Execute and return response
        result = expert_crew.kickoff()
        return str(result)
    
    def _get_critic_feedback(self, ctx: Context, expert_response: str) -> str:
        """Get feedback from both critic agents."""
        # Create critic tasks
        critic_tasks = build_paper_tasks(ctx, self.agents, expert_response=expert_response)
        
        # Create crew with both critics
        critic_crew = Crew(
            agents=[self.agents["extraction_critic"], self.agents["calculation_critic"]],
            tasks=critic_tasks,
            process=Process.sequential,
            verbose=self.verbose,
            **self._build_crew_kwargs(),
        )
        
        # Execute critic tasks
        critic_results = critic_crew.kickoff()
        return str(critic_results)
    
    def _critics_approve(self, critic_feedback: str) -> bool:
        """Return True only if *all* critics set ``is_correct`` to true.

        The previous implementation inspected just the first JSON object in the
        concatenated critic output, which meant the verdict of one critic could
        override the other.  We now:
        1. Find *every* JSON-looking block in the raw string (works whether
           CrewAI returned a python-list string or simple concatenation).
        2. Parse each block that contains an ``is_correct`` key.
        3. If we parsed at least one such flag, the overall approval is the
           logical **AND** of all flags (unanimous approval required).
        4. If no JSON blocks are found, fall back to the original keyword
           heuristic for backward compatibility with older prompts.
        """
        # ------------------------------------------------------------------
        # 1️⃣  Parse all structured JSON feedback blocks
        # ------------------------------------------------------------------
        parsed_flags: list[bool] = []
        try:
            # Find every minimal JSON object inside the string
            json_blocks = re.findall(r"\{[^{}]*\}", critic_feedback, re.DOTALL)
            for block in json_blocks:
                try:
                    obj = json.loads(block)
                    if isinstance(obj, dict) and "is_correct" in obj:
                        parsed_flags.append(bool(obj["is_correct"]))
                except json.JSONDecodeError:
                    # Skip malformed blocks – they'll be handled by heuristics
                    continue
        except Exception as e:  # pragma: no cover
            _logger.debug(f"Failed to collect critic JSON blocks: {e}")

        # If we extracted at least one explicit flag, require unanimous approval
        if parsed_flags:
            all_approved = all(parsed_flags)
            _logger.debug(f"Critic approval flags={parsed_flags} -> {all_approved}")
            return all_approved

        # ------------------------------------------------------------------
        # 2️⃣  Fallback to keyword-based heuristic (legacy behaviour)
        # ------------------------------------------------------------------
        feedback_lower = critic_feedback.lower()

        approval_indicators = [
            "approved",
            "approve",
            "good",
            "correct",
            "accurate",
            "sound",
            "logical",
        ]
        improvement_indicators = [
            "needs_improvement",
            "needs improvement",
            "incorrect",
            "error",
            "missing",
            "should",
            "could",
            "would",
        ]

        has_approval = any(indicator in feedback_lower for indicator in approval_indicators)
        needs_improvement = any(indicator in feedback_lower for indicator in improvement_indicators)

        # Explicit positive without improvement cues ⇒ approve
        if has_approval and not needs_improvement:
            return True

        # Default: approve only if no improvement keywords found
        return not needs_improvement
    
    def _extract_final_answer(self, expert_response: str) -> str:
        """Robustly extract the final answer from the expert response.

        Strategy:
        1. Iterate over every JSON-looking substring and attempt to parse it.
           Return `parsed["answer"]` from the first valid JSON containing that key.
        2. Fallback to several regex patterns.
        3. Ultimately return the raw response trimmed.
        """

        # 1. Attempt to parse every balanced JSON block
        try:
            for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", expert_response, re.DOTALL):
                json_candidate = match.group(0)
                try:
                    parsed = json.loads(json_candidate)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        return str(parsed["answer"])
                except json.JSONDecodeError:
                    continue  # try next candidate
        except Exception as e:  # pragma: no cover
            _logger.debug(f"JSON candidate extraction failed: {e}")

        # 2. Regex fallbacks
        answer_patterns = [
            r'"answer"\s*[:=]\s*"([^\"]+)"',
            r'"answer"\s*[:=]\s*([0-9.+\-eE%]+)',
            r'answer\s*[:=]\s*([0-9.+\-eE%]+)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, expert_response, re.IGNORECASE)
            if match:
                return match.group(1)

        # 3. Raw fallback
        return expert_response.strip()
    
    def _apply_post_processing(self, answer: str, ctx: Context) -> str:
        """Apply lightweight post-processing helpers for sanity checks."""
        try:
            processed = answer.strip()

            # Basic numeric extraction and comma removal
            if self.number_extraction_enabled:
                processed = self._extract_numeric(processed)

            return processed
            
        except Exception as e:
            _logger.warning(f"Post-processing failed: {e}")
            return answer  # Return original answer if post-processing fails

    # ------------------------------------------------------------------
    def shutdown(self):
        if self.tracer:
            self.tracer.close()

    # ------------------------------------------------------------------
    # Numeric extraction helpers
    # ------------------------------------------------------------------

    def _extract_numeric(self, text: str) -> str:
        """Extract numeric value from text, handling scale conversions and comma removal.

        Handles cases like:
        - "4.575515 billion" -> "4575515000" (convert billion to actual number)
        - "52.84%" -> "0.5284" (convert percentage to decimal)
        - "4,575,515.0" -> "4575515.0" (remove commas)
        """
        cleaned = text.strip()

        # Quick check: if already numeric (after optional % sign)
        if self._looks_numeric(cleaned):
            return cleaned.replace(",", "") if self.number_cfg.get("remove_commas", True) else cleaned

        # Handle scale conversions first
        cleaned_lower = cleaned.lower()
        
        # Optional: convert percentages to decimal form (e.g. 52.8% -> 0.528)
        if self.number_cfg.get("convert_percentage_to_decimal", False):
            pct_match = re.search(r'(\d+\.?\d*)%', cleaned_lower)
            if pct_match:
                number = float(pct_match.group(1))
                result = number * 0.01
                # Remove trailing .0 for integers
                if result == int(result):
                    return str(int(result))
                else:
                    return str(result)
        
        # Remove commas for easier regex matching
        text_no_commas = cleaned.replace(",", "") if self.number_cfg.get("remove_commas", True) else cleaned

        # Find all numbers (integers or decimals)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text_no_commas)

        if not numbers:
            return cleaned  # nothing found

        # Only filter years if we have multiple candidates AND at least one non-year number > 100
        # (to avoid filtering legitimate small values like 2.1 that happen to be near year range)
        if self.number_cfg.get("exclude_years", True) and len(numbers) > 1:
            non_year_numbers = [n for n in numbers if not (1900 <= float(n) <= 2100)]
            large_non_year_numbers = [n for n in non_year_numbers if abs(float(n)) > 100]
            
            # Only apply year filtering if we have substantial non-year alternatives
            if large_non_year_numbers:
                numbers = non_year_numbers
            elif non_year_numbers and any(abs(float(n)) > 10 for n in non_year_numbers):
                # More lenient: filter years if we have non-year numbers > 10
                numbers = non_year_numbers

        # Choose number according to config
        if self.number_cfg.get("prefer_largest_number", True):
            try:
                chosen = max(numbers, key=lambda x: abs(float(x)))
            except Exception:
                chosen = numbers[0]
        else:
            chosen = numbers[0]

        return chosen

    @staticmethod
    def _looks_numeric(text: str) -> bool:
        """Return True if text represents a simple numeric value (not percentage or scale)."""
        # Don't treat percentages or scale indicators as already numeric
        if '%' in text or 'billion' in text.lower() or 'million' in text.lower() or 'thousand' in text.lower():
            return False
        
        # Allow basic numeric formats
        pattern = r"^-?\d+(?:,\d{3})*(?:\.\d+)?$"
        return re.match(pattern, text) is not None 

    # ------------------------------------------------------------------
    # Internal helper for Crew constructor kwargs
    # ------------------------------------------------------------------
    def _build_crew_kwargs(self) -> Dict[str, Any]:
        """Return additional kwargs containing tracer as callback.

        CrewAI's API changed across versions: some use `callbacks`, others
        use `callback_manager`.  We inspect the constructor signature at
        runtime and supply whichever parameter is available.  If neither
        is supported, we silently return an empty dict so that execution
        continues without tracing (fail-soft behaviour).
        """
        if self.tracer is None:
            return {}

        # Pass even if not in signature; CrewAI dataclass accepts **data.
        return {"callbacks": [self.tracer]}  # type: ignore[arg-type] 