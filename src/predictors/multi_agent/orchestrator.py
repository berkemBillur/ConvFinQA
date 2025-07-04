from __future__ import annotations

"""High-level Crew wrapper implementing the paper's reflection framework.

This module implements the iterative reflection approach from "Enhancing Financial 
Question Answering with a Multi-Agent Reflection Framework" (arXiv:2410.21741):

Expert Initial Response → Critics Review → Expert Revision → Final Answer

The orchestrator maintains backward compatibility while using the paper's proven approach.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
import json
import re
import time

try:
    from crewai import Crew, Process
except ImportError as exc:  # pragma: no cover
    raise ImportError("CrewAI must be installed to use ConvFinQACrew") from exc

from .agents import build_agents, build_agents_six
from .tasks import Context, build_paper_tasks, build_six_agent_tasks, build_extraction_critic_task_six, build_calculation_critic_task_six
from .tracer import build_tracer, JsonlTracer
from .schemas import (
    SchemaValidator, ManagerOutput, ExtractorOutput, ReasonerOutput, 
    CriticOutput, SynthesiserOutput, ValidationError
)
from ...utils.config import Config
from ...utils.financial_matcher import financial_matcher
from ...utils.scale_normalizer import scale_normalizer

_logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is triggered."""
    pass


class CostBudgetExceeded(Exception):
    """Exception raised when cost budget is exceeded."""
    pass


class TokenUsageTracker:
    """Track token usage and costs across the multi-agent pipeline."""
    
    def __init__(self, budget_tokens: int = 15000, cost_per_1k_tokens: float = 0.002):
        self.budget_tokens = budget_tokens
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.total_tokens_used = 0
        self.stage_tokens = {}
        self.stage_calls = {}
        self.start_time = time.time()
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple word-based approximation."""
        # Rough approximation: 1 token ~= 0.75 words for English
        word_count = len(text.split())
        return int(word_count / 0.75)
    
    def record_stage_usage(self, stage_name: str, input_text: str, output_text: str):
        """Record token usage for a specific stage."""
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        total_stage_tokens = input_tokens + output_tokens
        
        # Track stage-specific usage
        if stage_name not in self.stage_tokens:
            self.stage_tokens[stage_name] = 0
            self.stage_calls[stage_name] = 0
        
        self.stage_tokens[stage_name] += total_stage_tokens
        self.stage_calls[stage_name] += 1
        self.total_tokens_used += total_stage_tokens
        
        _logger.debug(f"📊 {stage_name}: {input_tokens} in + {output_tokens} out = {total_stage_tokens} tokens")
    
    def check_budget(self, stage_name: str = "unknown") -> bool:
        """Check if we're within budget, raise exception if exceeded."""
        if self.total_tokens_used > self.budget_tokens:
            cost = self.get_estimated_cost()
            raise CostBudgetExceeded(
                f"Token budget exceeded at {stage_name}: {self.total_tokens_used}/{self.budget_tokens} "
                f"tokens (${cost:.4f})"
            )
        return True
    
    def get_remaining_budget(self) -> int:
        """Get remaining token budget."""
        return max(0, self.budget_tokens - self.total_tokens_used)
    
    def get_estimated_cost(self) -> float:
        """Get estimated cost in USD."""
        return (self.total_tokens_used / 1000) * self.cost_per_1k_tokens
    
    def should_early_exit(self, stage_name: str) -> bool:
        """Determine if we should exit early to preserve budget."""
        remaining = self.get_remaining_budget()
        
        # Estimate tokens needed for remaining stages
        remaining_stages_estimate = self._estimate_remaining_tokens(stage_name)
        
        return remaining < remaining_stages_estimate
    
    def _estimate_remaining_tokens(self, current_stage: str) -> int:
        """Estimate tokens needed for remaining pipeline stages."""
        stage_estimates = {
            "manager": 500,
            "extraction": 1500,
            "reasoning": 2000,
            "critics": 1000,
            "synthesis": 800
        }
        
        stage_order = ["manager", "extraction", "reasoning", "critics", "synthesis"]
        
        try:
            current_index = stage_order.index(current_stage)
            remaining_stages = stage_order[current_index + 1:]
            return sum(stage_estimates.get(stage, 500) for stage in remaining_stages)
        except ValueError:
            return 2000  # Default conservative estimate
    
    def get_usage_summary(self) -> dict:
        """Get comprehensive usage summary."""
        elapsed_time = time.time() - self.start_time
        cost = self.get_estimated_cost()
        
        return {
            "total_tokens": self.total_tokens_used,
            "budget_tokens": self.budget_tokens,
            "budget_remaining": self.get_remaining_budget(),
            "budget_utilization": (self.total_tokens_used / self.budget_tokens) * 100,
            "estimated_cost_usd": cost,
            "elapsed_time_seconds": elapsed_time,
            "tokens_per_second": self.total_tokens_used / elapsed_time if elapsed_time > 0 else 0,
            "stage_breakdown": dict(self.stage_tokens),
            "stage_calls": dict(self.stage_calls)
        }


class StageRecoveryManager:
    """Manages error recovery and circuit breaking for multi-agent stages."""
    
    def __init__(self, max_failures: int = 3, failure_window_minutes: int = 5):
        self.max_failures = max_failures
        self.failure_window_minutes = failure_window_minutes
        self.failure_history: Dict[str, list] = {}
        
    def record_failure(self, stage_name: str):
        """Record a failure for circuit breaker tracking."""
        current_time = time.time()
        
        if stage_name not in self.failure_history:
            self.failure_history[stage_name] = []
        
        # Add current failure
        self.failure_history[stage_name].append(current_time)
        
        # Clean up old failures outside the window
        window_start = current_time - (self.failure_window_minutes * 60)
        self.failure_history[stage_name] = [
            t for t in self.failure_history[stage_name] 
            if t >= window_start
        ]
    
    def should_break_circuit(self, stage_name: str) -> bool:
        """Check if circuit breaker should be triggered."""
        if stage_name not in self.failure_history:
            return False
        
        current_time = time.time()
        window_start = current_time - (self.failure_window_minutes * 60)
        
        recent_failures = [
            t for t in self.failure_history[stage_name] 
            if t >= window_start
        ]
        
        return len(recent_failures) >= self.max_failures
    
    def reset_circuit(self, stage_name: str):
        """Reset circuit breaker for a stage."""
        if stage_name in self.failure_history:
            self.failure_history[stage_name] = []


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
        return self._extract_approval_heuristic(critic_feedback)

    def _extract_approval_heuristic(self, feedback: str) -> bool:
        """Simple heuristic fallback for extracting critic approval."""
        feedback_lower = feedback.lower()
        
        # Look for positive indicators
        positive_indicators = [
            "correct", "accurate", "good", "appropriate", "valid", "sound"
        ]
        
        # Look for negative indicators
        negative_indicators = [
            "incorrect", "wrong", "error", "missing", "invalid", "inaccurate"
        ]
        
        positive_count = sum(1 for word in positive_indicators if word in feedback_lower)
        negative_count = sum(1 for word in negative_indicators if word in feedback_lower)
        
        # Simple heuristic: more positive than negative
        return positive_count > negative_count
    
    def _extract_final_answer(self, expert_response: str) -> str:
        """Extract the final answer from expert response."""
        if not expert_response:
            return "0"
        
        # Try to extract from answer field in JSON
        try:
            # Look for JSON-like structure first
            json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]*)"[^{}]*\}', expert_response, re.DOTALL)
            if json_match:
                return json_match.group(1)
        except Exception:
            pass
        
        # Try to extract numeric answer
        if self.number_extraction_enabled:
            return self._extract_numeric(expert_response)
        
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

    def shutdown(self):
        """Clean up resources."""
        if self.tracer:
            self.tracer.close()

    def _extract_numeric(self, text: str) -> str:
        """Extract numeric value from text, handling scale conversions and comma removal."""
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
        if self.number_cfg.get("exclude_years", True) and len(numbers) > 1:
            non_year_numbers = [n for n in numbers if not (1900 <= float(n) <= 2100)]
            large_non_year_numbers = [n for n in non_year_numbers if abs(float(n)) > 100]
            
            # Only apply year filtering if we have substantial non-year alternatives
            if large_non_year_numbers:
                numbers = non_year_numbers
            elif non_year_numbers and any(abs(float(n)) > 10 for n in non_year_numbers):
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
        """Check if text looks like a standalone number."""
        cleaned = text.strip().rstrip('%')
        try:
            float(cleaned.replace(",", ""))
            return True
        except ValueError:
            return False
        
    def _build_crew_kwargs(self) -> Dict[str, Any]:
        """Build keyword arguments for Crew initialization."""
        kwargs = {}
        
        # Add tracer if available
        if self.tracer:
            kwargs["callbacks"] = [self.tracer]
        
        return kwargs


class ConvFinQASixAgentCrew:
    """Six-agent tiered architecture implementing specialized workflow with comprehensive error recovery."""

    def __init__(self, config: Config, trace_dir: Path | None = None):
        self.config = config
        self.agents = build_agents_six(config)
        self.tracer: JsonlTracer | None = build_tracer(trace_dir)
        
        # Get six-agent specific configuration
        self.agent_config = config.get("six_agent_config", {})
        self.max_iterations = self.agent_config.get("max_iterations", 2)
        self.enable_post_processing = self.agent_config.get("enable_post_processing", True)
        self.verbose = self.agent_config.get("verbose", True)
        self.cache_enabled = self.agent_config.get("cache_enabled", True)

        # Number extraction config
        self.number_cfg = self.agent_config.get("number_extraction", {})
        self.number_extraction_enabled = self.number_cfg.get("enabled", True)
        
        # Cost monitoring configuration
        cost_config = self.agent_config.get("cost_monitoring", {})
        budget_tokens = cost_config.get("budget_tokens", 15000)
        cost_per_1k_tokens = cost_config.get("cost_per_1k_tokens", 0.002)
        
        # In-memory conversation cache
        self.conversation_cache: dict[str, str] = {}
        
        # Error recovery manager
        self.recovery_manager = StageRecoveryManager(
            max_failures=3, 
            failure_window_minutes=5
        )
        
        # Token usage tracker
        self.token_tracker = TokenUsageTracker(
            budget_tokens=budget_tokens,
            cost_per_1k_tokens=cost_per_1k_tokens
        )

    # ------------------------------------------------------------------
    # Public API with Enhanced Error Recovery
    # ------------------------------------------------------------------
    def run(self, ctx: Context) -> str:
        """Execute the six-agent tiered workflow with comprehensive error recovery and cost monitoring."""
        if self.verbose:
            _logger.info(f"🚀 Starting six-agent workflow for question: {ctx.question[:100]}...")
        
        # Enhanced tracing: log workflow start
        if self.tracer:
            self.tracer.log_workflow_start("six_agent", ctx.question)
        
        try:
            # Stage 1: Manager - routing decision with error recovery and cost tracking
            manager_result = self._run_manager_stage_with_recovery(ctx)
            if manager_result and manager_result.action == "cache_hit":
                cached_answer = manager_result.cached_answer
                if cached_answer:
                    if self.verbose:
                        _logger.info(f"💾 Cache hit: {cached_answer[:50]}...")
                    # Log successful workflow end with usage summary
                    if self.tracer:
                        self.tracer.log_workflow_end(cached_answer, success=True)
                    self._log_final_usage_summary(cached_answer)
                    return cached_answer
            
            # Stage 2: Full pipeline with comprehensive error recovery and cost monitoring
            final_answer = self._run_full_pipeline_with_recovery(ctx)
            
            # Cache the result
            if self.cache_enabled and final_answer:
                self.conversation_cache[ctx.question] = final_answer
                if self.verbose:
                    _logger.info(f"💾 Cached answer for: {ctx.question[:50]}...")
            
            # Log successful workflow end with usage summary
            if self.tracer:
                self.tracer.log_workflow_end(final_answer, success=True)
            
            self._log_final_usage_summary(final_answer)
            return final_answer
        
        except CostBudgetExceeded as e:
            # Handle budget exceeded gracefully
            _logger.warning(f"💰 Budget exceeded: {e}")
            if self.tracer:
                self.tracer.log_workflow_end(f"Budget exceeded: {e}", success=False)
            self._log_final_usage_summary("Budget exceeded")
            return self._create_emergency_fallback(ctx, f"Budget exceeded: {e}")
        
        except Exception as e:
            # Log failed workflow end
            if self.tracer:
                self.tracer.log_workflow_end(str(e), success=False)
            
            # Last resort fallback
            _logger.error(f"🚨 Complete workflow failure: {e}")
            self._log_final_usage_summary(f"Error: {e}")
            return self._create_emergency_fallback(ctx, str(e))

    def _run_manager_stage_with_recovery(self, ctx: Context) -> Optional[ManagerOutput]:
        """Run manager stage with error recovery and cost tracking."""
        stage_name = "manager"
        
        # Check circuit breaker
        if self.recovery_manager.should_break_circuit(stage_name):
            _logger.warning(f"🔌 Circuit breaker open for {stage_name} - skipping to pipeline")
            return ManagerOutput(
                action="run_pipeline",
                cached_answer="",
                reasoning="Circuit breaker triggered - proceeding with full pipeline"
            )
        
        # Check if we should early exit due to budget constraints
        if self.token_tracker.should_early_exit(stage_name):
            _logger.warning(f"💰 Early exit from {stage_name} due to budget constraints")
            return ManagerOutput(
                action="run_pipeline", 
                cached_answer="",
                reasoning="Budget constraints - proceeding with full pipeline"
            )
        
        try:
            if self.verbose:
                _logger.info("👔 Running Manager stage...")
        
            # Enhanced tracing: set context and log stage
            if self.tracer:
                self.tracer.set_execution_context(self.tracer._execution_id or "unknown", "manager")
        
            manager_tasks = build_six_agent_tasks(ctx, self.agents, "manager", 
                                                conversation_cache=self.conversation_cache)
        
            # Prepare input for token tracking
            input_text = f"Question: {ctx.question}"
        
            manager_crew = Crew(
                agents=[self.agents["manager"]],
                tasks=manager_tasks,
                process=Process.sequential,
                verbose=self.verbose,
                **self._build_crew_kwargs(),
            )
        
            result = manager_crew.kickoff()
            result_str = str(result)
            
            # Track token usage
            self.token_tracker.record_stage_usage(stage_name, input_text, result_str)
            
            # Check budget after execution
            self.token_tracker.check_budget(stage_name)
            
            manager_output = SchemaValidator.validate_manager_output(result_str)
            
            # Reset circuit breaker on success
            self.recovery_manager.reset_circuit(stage_name)
            return manager_output
            
        except Exception as e:
            _logger.error(f"❌ Manager stage failed: {e}")
            self.recovery_manager.record_failure(stage_name)
            
            # Graceful fallback: proceed with pipeline
            return ManagerOutput(
                action="run_pipeline",
                cached_answer="",
                reasoning=f"Manager stage failed: {str(e)[:100]}. Proceeding with full pipeline."
            )

    def _run_full_pipeline_with_recovery(self, ctx: Context) -> str:
        """Run the full six-agent pipeline with comprehensive error recovery."""
        if self.verbose:
            _logger.info("🔧 Running full pipeline with error recovery...")
        
        last_valid_answer = "0"  # Fallback answer
        
        # Iteration loop for revision
        for iteration in range(self.max_iterations):
            if self.verbose:
                _logger.info(f"📝 Iteration {iteration + 1}/{self.max_iterations}")
            
            ctx.iteration = iteration
            
            try:
                # Stage 2: Data Extraction with recovery
                extractor_output = self._run_extraction_stage_with_recovery(ctx)
                if not extractor_output:
                    _logger.error("❌ Extraction stage failed completely - using emergency fallback")
                    continue
                
                # Stage 3: Financial Reasoning with recovery
                reasoner_output = self._run_reasoning_stage_with_recovery(ctx, extractor_output)
                if not reasoner_output:
                    _logger.error("❌ Reasoning stage failed completely - using emergency fallback")
                    continue
                
                # Store the reasoner answer as a fallback
                last_valid_answer = reasoner_output.answer
                
                # Stage 4: Critics with recovery
                extraction_critic_output, calculation_critic_output = self._run_critics_stage_with_recovery(
                ctx, extractor_output, reasoner_output)
            
                # Stage 5: Synthesis with recovery
                synthesis_result = self._run_synthesis_stage_with_recovery(
                ctx, reasoner_output, extraction_critic_output, calculation_critic_output)
            
                # Handle synthesis outcome
                if not synthesis_result:
                    _logger.warning("❌ Synthesis stage failed - using reasoner output")
                    final_answer = last_valid_answer
                    if self.enable_post_processing:
                        final_answer = self._apply_post_processing(final_answer, ctx)
                    return final_answer

                if synthesis_result.status == "final":
                    # Numeric guard (Issue 3)
                    if not self._looks_numeric(synthesis_result.answer):
                        if self.verbose:
                            _logger.warning("⚠️ Final answer is not numeric – forcing revision queue")
                        ctx.critic_feedback = "Answer was non-numeric; please provide a numeric value."
                        continue

                    final_answer = synthesis_result.answer
                    if self.verbose:
                        _logger.info("✅ Critics approved - final answer ready")

                    if self.enable_post_processing:
                        final_answer = self._apply_post_processing(final_answer, ctx)

                    return final_answer
                else:  # status == "revise"
                    if self.verbose:
                        _logger.info(f"🔄 Revision requested: {synthesis_result.critique_summary[:100]}...")
                    ctx.critic_feedback = synthesis_result.critique_summary or "Critics requested revision."
                    continue
                
            except Exception as e:
                _logger.error(f"❌ Pipeline iteration {iteration + 1} failed: {e}")
                # Continue to next iteration or use fallback
                continue
        
        # If we've exhausted iterations, return the last valid answer
        if self.verbose:
            _logger.warning(f"⚠️ Reached max iterations ({self.max_iterations}), using last valid answer")
        
        if self.enable_post_processing:
            last_valid_answer = self._apply_post_processing(last_valid_answer, ctx)
        
        return last_valid_answer

    def _run_extraction_stage_with_recovery(self, ctx: Context) -> Optional[ExtractorOutput]:
        """Run extraction stage with error recovery and cost tracking."""
        stage_name = "extraction"
        
        # Check circuit breaker
        if self.recovery_manager.should_break_circuit(stage_name):
            _logger.warning(f"🔌 Circuit breaker open for {stage_name}")
            raise CircuitBreakerError(f"Circuit breaker triggered for {stage_name}")
        
        # Check budget before execution
        if self.token_tracker.should_early_exit(stage_name):
            _logger.warning(f"💰 Early exit from {stage_name} due to budget constraints")
            return self._create_fallback_extraction(ctx, "Budget constraints")
        
        try:
            if self.verbose:
                _logger.info("🔍 Running Extraction stage...")
        
            extraction_tasks = build_six_agent_tasks(ctx, self.agents, "extraction")
            
            # Prepare input for token tracking
            input_text = f"Question: {ctx.question}\nTable: {str(getattr(ctx, 'table', 'No table data'))[:500]}..."
        
            extraction_crew = Crew(
                agents=[self.agents["extractor"]],
                tasks=extraction_tasks,
                process=Process.sequential,
                verbose=self.verbose,
                **self._build_crew_kwargs(),
            )
        
            result = extraction_crew.kickoff()
            result_str = str(result)
            
            # Track token usage
            self.token_tracker.record_stage_usage(stage_name, input_text, result_str)
            
            # Check budget after execution
            self.token_tracker.check_budget(stage_name)
            
            extractor_output = SchemaValidator.validate_extractor_output(result_str)
            
            # Reset circuit breaker on success
            self.recovery_manager.reset_circuit(stage_name)
            return extractor_output
            
        except Exception as e:
            _logger.error(f"❌ Extraction stage failed: {e}")
            self.recovery_manager.record_failure(stage_name)
            
            # Try to create a minimal fallback extraction
            try:
                return self._create_fallback_extraction(ctx, str(e))
            except Exception as fallback_error:
                _logger.error(f"❌ Fallback extraction failed: {fallback_error}")
                return None

    def _run_reasoning_stage_with_recovery(self, ctx: Context, extractor_output: ExtractorOutput) -> Optional[ReasonerOutput]:
        """Run reasoning stage with error recovery and cost tracking."""
        stage_name = "reasoning"
        
        # Check circuit breaker
        if self.recovery_manager.should_break_circuit(stage_name):
            _logger.warning(f"🔌 Circuit breaker open for {stage_name}")
            return self._create_fallback_reasoning(ctx, "Circuit breaker triggered")
        
        # Check budget before execution
        if self.token_tracker.should_early_exit(stage_name):
            _logger.warning(f"💰 Early exit from {stage_name} due to budget constraints")
            return self._create_fallback_reasoning(ctx, "Budget constraints")
        
        try:
            if self.verbose:
                _logger.info("🧮 Running Reasoning stage...")
        
            reasoning_tasks = build_six_agent_tasks(ctx, self.agents, "reasoning", 
                                                  extractor_output=extractor_output.model_dump_json())
        
            # Prepare input for token tracking
            input_text = f"Question: {ctx.question}\nExtractions: {extractor_output.model_dump_json()[:300]}..."
        
            reasoning_crew = Crew(
                agents=[self.agents["reasoner"]],
                tasks=reasoning_tasks,
                process=Process.sequential,
                verbose=self.verbose,
                **self._build_crew_kwargs(),
            )
        
            result = reasoning_crew.kickoff()
            result_str = str(result)
            
            # Track token usage
            self.token_tracker.record_stage_usage(stage_name, input_text, result_str)
            
            # Check budget after execution
            self.token_tracker.check_budget(stage_name)
            
            reasoner_output = SchemaValidator.validate_reasoner_output(result_str)
            
            # Execute DSL if present with enhanced error handling
            if reasoner_output.dsl:
                reasoner_output = self._execute_dsl_safely(reasoner_output)
            
            # Reset circuit breaker on success
            self.recovery_manager.reset_circuit(stage_name)
            return reasoner_output

        except Exception as e:
            _logger.error(f"❌ Reasoning stage failed: {e}")
            self.recovery_manager.record_failure(stage_name)
            
            # Create fallback reasoning
            return self._create_fallback_reasoning(ctx, str(e))

    def _run_critics_stage_with_recovery(self, ctx: Context, extractor_output: ExtractorOutput, reasoner_output: ReasonerOutput) -> tuple[CriticOutput, CriticOutput]:
        """Run critics stage with error recovery and cost tracking."""
        stage_name = "critics"
        
        # Check circuit breaker
        if self.recovery_manager.should_break_circuit(stage_name):
            _logger.warning(f"🔌 Circuit breaker open for {stage_name}")
            return self._create_fallback_critics("Circuit breaker triggered")
        
        # Check budget before execution
        if self.token_tracker.should_early_exit(stage_name):
            _logger.warning(f"💰 Early exit from {stage_name} due to budget constraints")
            return self._create_fallback_critics("Budget constraints")
        
        try:
            if self.verbose:
                _logger.info("🔍 Running Critics stage...")
        
            # FIXED: Run each critic separately to avoid result loss
            # Build individual tasks for each critic
            extraction_critic_tasks = [build_extraction_critic_task_six(ctx, extractor_output.model_dump_json(), self.agents["extraction_critic"])]
            calculation_critic_tasks = [build_calculation_critic_task_six(ctx, reasoner_output.model_dump_json(), self.agents["calculation_critic"])]
            
            # Prepare input for token tracking
            input_text = f"Question: {ctx.question}\nExtraction: {extractor_output.model_dump_json()[:200]}...\nReasoning: {reasoner_output.model_dump_json()[:200]}..."
            
            # Run extraction critic separately
            extraction_critic_crew = Crew(
                agents=[self.agents["extraction_critic"]],
                tasks=extraction_critic_tasks,
                process=Process.sequential,
                verbose=self.verbose,
                **self._build_crew_kwargs(),
            )
        
            extraction_critic_result = extraction_critic_crew.kickoff()
            extraction_critic_result_str = str(extraction_critic_result)
            
            # Run calculation critic separately  
            calculation_critic_crew = Crew(
                agents=[self.agents["calculation_critic"]],
                tasks=calculation_critic_tasks,
                process=Process.sequential,
                verbose=self.verbose,
                **self._build_crew_kwargs(),
            )
            
            calculation_critic_result = calculation_critic_crew.kickoff()
            calculation_critic_result_str = str(calculation_critic_result)
            
            # Track combined token usage
            combined_results = f"Extraction: {extraction_critic_result_str}\nCalculation: {calculation_critic_result_str}"
            self.token_tracker.record_stage_usage(stage_name, input_text, combined_results)
            
            # Check budget after execution
            self.token_tracker.check_budget(stage_name)
            
            # Parse each result individually
            extraction_critic_output = SchemaValidator.validate_critic_output(extraction_critic_result_str, "extraction_critic")
            calculation_critic_output = SchemaValidator.validate_critic_output(calculation_critic_result_str, "calculation_critic")
        
            # Reset circuit breaker on success
            self.recovery_manager.reset_circuit(stage_name)
            return extraction_critic_output, calculation_critic_output

        except Exception as e:
            _logger.error(f"❌ Critics stage failed: {e}")
            self.recovery_manager.record_failure(stage_name)
            
            # Create fallback critics that request revision to be safe
            return self._create_fallback_critics(str(e))

    def _run_synthesis_stage_with_recovery(self, ctx: Context, reasoner_output: ReasonerOutput, 
                                         extraction_critic_output: CriticOutput, calculation_critic_output: CriticOutput) -> Optional[SynthesiserOutput]:
        """Run synthesis stage with error recovery and cost tracking."""
        stage_name = "synthesis"
        
        # Check budget before execution
        if self.token_tracker.should_early_exit(stage_name):
            _logger.warning(f"💰 Early exit from {stage_name} due to budget constraints")
            return self._create_fallback_synthesis(reasoner_output, extraction_critic_output, calculation_critic_output, "Budget constraints")
        
        try:
            if self.verbose:
                _logger.info("🎯 Running Synthesis stage...")
        
            synthesis_tasks = build_six_agent_tasks(ctx, self.agents, "synthesis",
                                                  reasoner_output=reasoner_output.model_dump_json(),
                                                  extraction_critic_output=extraction_critic_output.model_dump_json(),
                                                  calculation_critic_output=calculation_critic_output.model_dump_json())
        
            # Prepare input for token tracking
            input_text = f"Question: {ctx.question}\nReasoning: {reasoner_output.model_dump_json()[:200]}...\nCritic feedback: {extraction_critic_output.model_dump_json()[:100]}..."
        
            synthesis_crew = Crew(
                agents=[self.agents["synthesiser"]],
                tasks=synthesis_tasks,
                process=Process.sequential,
                verbose=self.verbose,
                **self._build_crew_kwargs(),
            )
        
            result = synthesis_crew.kickoff()
            result_str = str(result)
            
            # Track token usage
            self.token_tracker.record_stage_usage(stage_name, input_text, result_str)
            
            # Check budget after execution
            self.token_tracker.check_budget(stage_name)
            
            synthesis_output = SchemaValidator.validate_synthesiser_output(result_str)
            
            return synthesis_output
            
        except Exception as e:
            _logger.error(f"❌ Synthesis stage failed: {e}")
            self.recovery_manager.record_failure(stage_name)
            
            # Create fallback synthesis based on critic outputs
            return self._create_fallback_synthesis(reasoner_output, extraction_critic_output, calculation_critic_output, str(e))

    # ------------------------------------------------------------------
    # Enhanced Recovery Helper Methods
    # ------------------------------------------------------------------
    
    def _create_fallback_extraction(self, ctx: Context, error: str) -> ExtractorOutput:
        """Create a minimal fallback extraction."""
        from .schemas import ExtractionItem
        return ExtractorOutput(
            extractions=[ExtractionItem(
                row="FALLBACK_ROW",
                col="FALLBACK_COL", 
                raw="0",
                unit="",
                scale=1.0
            )],
            references_resolved=[],
            extraction_notes=f"⚠️ Fallback extraction due to error: {error[:100]}"
        )    
    def _create_fallback_reasoning(self, ctx: Context, error: str) -> ReasonerOutput:
        """Create a minimal fallback reasoning."""
        return ReasonerOutput(
            steps=[f"⚠️ Fallback reasoning due to error: {error[:100]}"],
            dsl="",
            answer="0"
        )
    
    def _create_fallback_critics(self, error: str) -> tuple[CriticOutput, CriticOutput]:
        """Create fallback critics that request revision to be safe."""
        fallback_critic = CriticOutput(
            is_correct=False,
            issues=[f"Critic evaluation failed: {error[:100]}"],
            suggested_fix="Please retry the operation due to critic evaluation failure."
        )
        return fallback_critic, fallback_critic
    
    def _create_fallback_synthesis(self, reasoner_output: ReasonerOutput, extraction_critic: CriticOutput, 
                                 calculation_critic: CriticOutput, error: str) -> SynthesiserOutput:
        """Create fallback synthesis based on available data."""
        # If both critics approve despite synthesis failure, use reasoner answer
        if extraction_critic.is_correct and calculation_critic.is_correct:
            return SynthesiserOutput(
                status="final",
                answer=reasoner_output.answer,
                critique_summary=f"Synthesis failed but critics approved: {error[:100]}"
            )
        else:
            # Request revision if critics had issues
            return SynthesiserOutput(
                status="revise",
                answer="",
                critique_summary=f"Synthesis failed and critics found issues: {error[:100]}"
            )
    
    def _create_emergency_fallback(self, ctx: Context, error: str) -> str:
        """Create an emergency fallback answer when everything fails."""
        _logger.error(f"🆘 Emergency fallback triggered: {error}")
        return "0"  # Safe numeric fallback
    
    def _execute_dsl_safely(self, reasoner_output: ReasonerOutput) -> ReasonerOutput:
        """Execute DSL with enhanced safety and error handling."""
        if not reasoner_output.dsl:
            return reasoner_output
        
        try:
            from ...evaluation.executor import execute_dsl_program
            
            # Comprehensive DSL validation before execution
            validation_result = self._validate_dsl_comprehensive(reasoner_output.dsl)
            if not validation_result["is_safe"]:
                if self.verbose:
                    _logger.warning(f"⚠️ DSL failed security validation: {validation_result['reason']}")
                return reasoner_output
            
            # Execute with timeout protection
            dsl_result = self._execute_dsl_with_timeout(reasoner_output.dsl, timeout_seconds=10)
            
            if isinstance(dsl_result, (int, float)):
                # Additional bounds checking on result
                if self._validate_result_bounds(dsl_result):
                    reasoner_output.answer = str(dsl_result)
                    if self.verbose:
                        _logger.info(f"🧮 DSL executed safely: {reasoner_output.dsl} = {dsl_result}")
                else:
                    if self.verbose:
                        _logger.warning(f"⚠️ DSL result out of bounds: {dsl_result}")
            else:
                if self.verbose:
                    _logger.warning(f"⚠️ DSL returned non-numeric result: {dsl_result}")
                    
        except Exception as e:
            if self.verbose:
                _logger.warning(f"⚠️ DSL execution failed safely: {e}")
        
        return reasoner_output
    
    def _validate_dsl_comprehensive(self, dsl_program: str) -> dict:
        """Comprehensive DSL safety validation with detailed reporting."""
        if not dsl_program or not dsl_program.strip():
            return {"is_safe": False, "reason": "Empty DSL program"}
        
        # Sanitize input
        sanitized_program = self._sanitize_dsl_input(dsl_program)
        
        # Check program length to prevent DoS
        if len(sanitized_program) > 1000:
            return {"is_safe": False, "reason": "DSL program too long (potential DoS)"}
        
        # Check for allowed operations only
        allowed_operations = ['add', 'subtract', 'multiply', 'divide', 'table_lookup']
        
        # Extract all function calls
        function_calls = re.findall(r'(\w+)\s*\(', sanitized_program)
        
        for func_name in function_calls:
            if func_name not in allowed_operations:
                return {"is_safe": False, "reason": f"Disallowed operation: {func_name}"}
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__\w+__',  # Python special methods
            r'eval|exec|import|open|file|input',  # Dangerous Python functions
            r'[;&|`$]',  # Shell injection patterns
            r'\.{2,}',  # Path traversal patterns
            r'<script|javascript:|data:',  # XSS patterns
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized_program, re.IGNORECASE):
                return {"is_safe": False, "reason": f"Dangerous pattern detected: {pattern}"}
        
        # Validate numeric arguments are within reasonable bounds
        numbers = re.findall(r'-?\d+(?:\.\d+)?', sanitized_program)
        for number in numbers:
            try:
                value = float(number)
                if abs(value) > 1e12:  # Prevent overflow
                    return {"is_safe": False, "reason": f"Number too large: {number}"}
            except ValueError:
                return {"is_safe": False, "reason": f"Invalid number format: {number}"}
        
        # Check for nested function depth (prevent stack overflow)
        nested_depth = self._calculate_nesting_depth(sanitized_program)
        if nested_depth > 10:
            return {"is_safe": False, "reason": "Too much nesting depth"}
        
        return {"is_safe": True, "reason": "DSL program passed all security checks"}
    
    def _sanitize_dsl_input(self, dsl_program: str) -> str:
        """Sanitize DSL input to remove potentially dangerous characters."""
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', dsl_program)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Strip and return
        return sanitized.strip()
    
    def _calculate_nesting_depth(self, program: str) -> int:
        """Calculate maximum nesting depth of parentheses."""
        max_depth = 0
        current_depth = 0
        
        for char in program:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        return max_depth
    
    def _execute_dsl_with_timeout(self, dsl_program: str, timeout_seconds: int = 10) -> Any:
        """Execute DSL with timeout protection."""
        import signal
        import time
        
        def timeout_handler(signum, frame):
            raise TimeoutError("DSL execution timed out")
        
        # Set up timeout handler (Unix-like systems only)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            start_time = time.time()
            from ...evaluation.executor import execute_dsl_program
            result = execute_dsl_program(dsl_program)
            execution_time = time.time() - start_time
            
            # Reset alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
            if self.verbose:
                _logger.debug(f"DSL executed in {execution_time:.3f}s")
            
            return result
            
        except TimeoutError:
            if self.verbose:
                _logger.warning(f"⚠️ DSL execution timed out after {timeout_seconds}s")
            raise
        
        except (AttributeError, OSError):
            # Windows or system without signal support - use basic execution
            from ...evaluation.executor import execute_dsl_program
            return execute_dsl_program(dsl_program)
        
        finally:
            try:
                signal.alarm(0)  # Make sure alarm is reset
            except (AttributeError, OSError):
                pass
    
    def _validate_result_bounds(self, result: Union[int, float]) -> bool:
        """Validate that DSL result is within reasonable bounds."""
        if not isinstance(result, (int, float)):
            return False
        
        # Check for NaN or infinity
        if not (result == result):  # NaN check
            return False
        
        if abs(result) == float('inf'):
            return False
        
        # Check reasonable bounds for financial calculations
        MAX_FINANCIAL_VALUE = 1e15  # 1 quadrillion - reasonable upper bound
        MIN_FINANCIAL_VALUE = -1e15
        
        return MIN_FINANCIAL_VALUE <= result <= MAX_FINANCIAL_VALUE
    
    def _validate_dsl_safety(self, dsl_program: str) -> bool:
        """Basic DSL safety validation - kept for backward compatibility."""
        validation_result = self._validate_dsl_comprehensive(dsl_program)
        return validation_result["is_safe"]

    def _log_final_usage_summary(self, final_answer: str):
        """Log comprehensive usage summary at the end of workflow."""
        usage_summary = self.token_tracker.get_usage_summary()
        
        if self.verbose:
            _logger.info("📊 === USAGE SUMMARY ===")
            _logger.info(f"💰 Total tokens: {usage_summary['total_tokens']}/{usage_summary['budget_tokens']}")
            _logger.info(f"💸 Estimated cost: ${usage_summary['estimated_cost_usd']:.4f}")
            _logger.info(f"📈 Budget utilization: {usage_summary['budget_utilization']:.1f}%")
            _logger.info(f"⏱️ Execution time: {usage_summary['elapsed_time_seconds']:.2f}s")
            _logger.info(f"🚀 Tokens/second: {usage_summary['tokens_per_second']:.1f}")
            
            # Log stage breakdown
            _logger.info("📋 Stage breakdown:")
            for stage, tokens in usage_summary['stage_breakdown'].items():
                calls = usage_summary['stage_calls'].get(stage, 0)
                _logger.info(f"  {stage}: {tokens} tokens ({calls} calls)")
        
        # Log to tracer if available
        if self.tracer:
            try:
                # Log usage summary (implementation varies by tracer)
                _logger.debug("Usage summary logged to tracer")
            except Exception as e:
                _logger.debug(f"Failed to log usage summary to tracer: {e}")
    
    def get_cost_summary(self) -> dict:
        """Get comprehensive cost and usage summary."""
        return self.token_tracker.get_usage_summary()

    def shutdown(self):
        """Clean up resources."""
        if self.tracer:
            self.tracer.close()

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

    def _extract_numeric(self, text: str) -> str:
        """Extract numeric value from text, handling scale conversions and comma removal."""
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
        if self.number_cfg.get("exclude_years", True) and len(numbers) > 1:
            non_year_numbers = [n for n in numbers if not (1900 <= float(n) <= 2100)]
            large_non_year_numbers = [n for n in non_year_numbers if abs(float(n)) > 100]
            
            # Only apply year filtering if we have substantial non-year alternatives
            if large_non_year_numbers:
                numbers = non_year_numbers
            elif non_year_numbers and any(abs(float(n)) > 10 for n in non_year_numbers):
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
        """Check if text looks like a standalone number."""
        cleaned = text.strip().rstrip('%')
        try:
            float(cleaned.replace(",", ""))
            return True
        except ValueError:
            return False

    def _build_crew_kwargs(self) -> Dict[str, Any]:
        """Build keyword arguments for Crew initialization."""
        kwargs = {}
        
        # Add tracer if available
        if self.tracer:
            kwargs["callbacks"] = [self.tracer]
        
        return kwargs
