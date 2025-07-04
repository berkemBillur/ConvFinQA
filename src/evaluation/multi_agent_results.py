"""Multi-agent results handling and storage for ConvFinQA benchmarks.

This module provides specialized result formatting and storage for multi-agent
predictions using CrewAI framework, maintaining consistency with the existing 
benchmark storage format while adding multi-agent-specific metadata.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrewAIQuestionResult:
    """Detailed result for a single CrewAI prediction."""
    record_id: str
    turn_index: int
    total_turns: int
    question: str
    ground_truth: str
    predicted_answer: str
    is_correct: bool
    dsl_program: str
    operation_type: str
    confidence: float
    reasoning: str
    
    # CrewAI-specific fields
    config_hash: str
    agent_flow: List[str]  # Which agents were involved
    execution_time: float
    estimated_cost: float
    fallback_used: bool
    error_message: Optional[str] = None


@dataclass
class CrewAICostBreakdown:
    """Cost breakdown for CrewAI execution."""
    total_cost: float
    cost_per_question: float
    cost_per_conversation: float
    supervisor_cost: float
    extractor_cost: float
    calculator_cost: float
    validator_cost: float


def save_crewai_timestamped_results(
    question_results: List[CrewAIQuestionResult],
    config_hash: str,
    agent_models: Dict[str, str],
    num_conversations: int,
    random_sample: bool,
    seed: Optional[int],
    cost_breakdown: CrewAICostBreakdown,
    timestamp: Optional[str] = None,
    results_base_dir: Optional[str] = None
) -> str:
    """Save CrewAI results to timestamped folder with all result files plus metadata.
    
    Args:
        question_results: List of CrewAI question results
        config_hash: Configuration hash for tracking
        agent_models: Dictionary of agent names to model names
        num_conversations: Number of conversations evaluated
        random_sample: Whether random sampling was used
        seed: Random seed if used
        cost_breakdown: Cost breakdown information
        
    Returns:
        Path to the created results directory
    """
    try:
        # Create timestamped directory
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if results_base_dir is None:
            results_base_dir = "results/multi_agent"
        
        results_dir = f"{results_base_dir}/{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Separate results by status
        all_results = question_results
        passed_results = [r for r in question_results if r.is_correct]
        failed_results = [r for r in question_results if not r.is_correct]
        
        # Save question-by-question results
        _save_crewai_questions_to_file(all_results, f"{results_dir}/all_results.txt", "All Results")
        _save_crewai_questions_to_file(failed_results, f"{results_dir}/failed_results.txt", "Failed Results")
        _save_crewai_questions_to_file(passed_results, f"{results_dir}/passed_results.txt", "Passed Results")
        
        # Save run metadata
        _save_crewai_run_metadata(
            f"{results_dir}/run_metadata.txt",
            config_hash,
            agent_models,
            num_conversations,
            random_sample,
            seed,
            len(all_results),
            len(passed_results),
            len(failed_results),
            cost_breakdown
        )
        
        logger.info(f"CrewAI results saved to: {results_dir}/")
        logger.info(f"  - all_results.txt ({len(all_results)} questions)")
        logger.info(f"  - failed_results.txt ({len(failed_results)} questions)")
        logger.info(f"  - passed_results.txt ({len(passed_results)} questions)")
        logger.info(f"  - run_metadata.txt")
        
        return results_dir
        
    except Exception as e:
        logger.error(f"Failed to save CrewAI results to timestamped folder: {e}")
        raise


def _save_crewai_questions_to_file(question_results: List[CrewAIQuestionResult], filename: str, title: str) -> None:
    """Save CrewAI question results to a specific file with enhanced details."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Add accuracy summary at the top for all_results.txt
            if title == "All Results" and question_results:
                correct_count = sum(1 for r in question_results if r.is_correct)
                total_count = len(question_results)
                accuracy_pct = (correct_count / total_count * 100) if total_count > 0 else 0
                
                f.write(f"{correct_count}/{total_count} questions correctly predicted\n")
                f.write(f"Accuracy rate: {accuracy_pct:.1f}%\n\n")
            
            f.write(f"ConvFinQA CrewAI Benchmark - {title}\n")
            f.write("=" * 80 + "\n\n")
            
            if not question_results:
                f.write("No questions in this category.\n")
                return
            
            for i, result in enumerate(question_results, 1):
                status = "✓ CORRECT" if result.is_correct else "✗ INCORRECT"
                
                f.write(f"{i}. Record: {result.record_id}, Turn {result.turn_index}/{result.total_turns}\n")
                f.write(f"Question: \"{result.question}\"\n")
                f.write(f"Expected: {result.ground_truth}\n")
                f.write(f"CrewAI Prediction: {result.predicted_answer}\n")
                f.write(f"Status: {status}\n")
                
                # CrewAI-specific information
                f.write(f"Agent Flow: {' → '.join(result.agent_flow)}\n")
                f.write(f"Execution Time: {result.execution_time:.1f}s\n")
                f.write(f"Estimated Cost: ${result.estimated_cost:.4f}\n")
                f.write(f"Configuration: {result.config_hash}\n")
                f.write(f"Confidence: {result.confidence:.3f}\n")
                
                if result.fallback_used:
                    f.write(f"Fallback: Used\n")
                
                if result.error_message:
                    f.write(f"Error: {result.error_message}\n")
                
                f.write(f"DSL Program: {result.dsl_program}\n")
                f.write(f"Operation: {result.operation_type}\n")
                
                if not result.is_correct and result.reasoning:
                    f.write(f"Error Analysis: {result.reasoning}\n")
                
                f.write("-" * 80 + "\n\n")
        
    except Exception as e:
        logger.error(f"Failed to save CrewAI {title} to {filename}: {e}")


def _save_crewai_run_metadata(
    filename: str,
    config_hash: str,
    agent_models: Dict[str, str],
    num_conversations: int,
    random_sample: bool,
    seed: Optional[int],
    total_questions: int,
    passed_questions: int,
    failed_questions: int,
    cost_breakdown: CrewAICostBreakdown
) -> None:
    """Save CrewAI run metadata to file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("CrewAI Benchmark - Run Metadata\n")
            f.write("=" * 80 + "\n\n")
            
            # Run parameters
            f.write("RUN PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration Hash: {config_hash}\n")
            f.write(f"Agent Models: ")
            model_str = ", ".join([f"{agent}({model})" for agent, model in agent_models.items()])
            f.write(f"{model_str}\n")
            f.write(f"Conversations evaluated: {num_conversations}\n")
            f.write(f"Sampling strategy: {'Random' if random_sample else 'Deterministic'}\n")
            if random_sample and seed is not None:
                f.write(f"Random seed: {seed}\n")
            f.write(f"Total questions: {total_questions}\n")
            f.write(f"Passed questions: {passed_questions}\n")
            f.write(f"Failed questions: {failed_questions}\n\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            accuracy_pct = (passed_questions / total_questions * 100) if total_questions > 0 else 0
            f.write(f"CrewAI Multi-Agent: {accuracy_pct:.1f}%\n")
            f.write(f"Target Achievement: {'PASS' if accuracy_pct >= 45.0 else 'BELOW TARGET'} (Target: 45.0%)\n\n")
            
            # Cost analysis
            f.write("COST ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total cost: ${cost_breakdown.total_cost:.2f}\n")
            f.write(f"Cost per question: ${cost_breakdown.cost_per_question:.4f}\n")
            f.write(f"Cost per conversation: ${cost_breakdown.cost_per_conversation:.4f}\n")
            f.write(f"Agent breakdown: ")
            f.write(f"Supervisor(${cost_breakdown.supervisor_cost:.2f}), ")
            f.write(f"Calculator(${cost_breakdown.calculator_cost:.2f}), ")
            f.write(f"Extractor(${cost_breakdown.extractor_cost:.2f}), ")
            f.write(f"Validator(${cost_breakdown.validator_cost:.2f})\n\n")
            
            # Configuration details
            f.write("CONFIGURATION DETAILS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Performance tracking enabled: Yes\n")
            f.write(f"Tracking data location: experiments/tracking/\n")
            f.write(f"Configuration fingerprint: {config_hash}\n")
            
    except Exception as e:
        logger.error(f"Failed to save CrewAI metadata to {filename}: {e}")


def convert_to_crewai_result(
    record_id: str,
    turn_index: int,
    total_turns: int,
    question: str,
    ground_truth: str,
    predicted_answer: str,
    is_correct: bool,
    dsl_program: str,
    operation_type: str,
    confidence: float,
    reasoning: str,
    config_hash: str,
    agent_flow: List[str],
    execution_time: float,
    estimated_cost: float,
    fallback_used: bool,
    error_message: Optional[str] = None
) -> CrewAIQuestionResult:
    """Convert individual prediction data to CrewAIQuestionResult."""
    return CrewAIQuestionResult(
        record_id=record_id,
        turn_index=turn_index,
        total_turns=total_turns,
        question=question,
        ground_truth=ground_truth,
        predicted_answer=predicted_answer,
        is_correct=is_correct,
        dsl_program=dsl_program,
        operation_type=operation_type,
        confidence=confidence,
        reasoning=reasoning,
        config_hash=config_hash,
        agent_flow=agent_flow,
        execution_time=execution_time,
        estimated_cost=estimated_cost,
        fallback_used=fallback_used,
        error_message=error_message
    )


def calculate_cost_breakdown(question_results: List[CrewAIQuestionResult]) -> CrewAICostBreakdown:
    """Calculate cost breakdown from question results."""
    total_cost = sum(r.estimated_cost for r in question_results)
    total_questions = len(question_results)
    
    # Estimate conversation count (assuming roughly 4 questions per conversation)
    total_conversations = max(1, total_questions // 4)
    
    cost_per_question = total_cost / total_questions if total_questions > 0 else 0
    cost_per_conversation = total_cost / total_conversations if total_conversations > 0 else 0
    
    # Estimate agent-specific costs (based on typical usage patterns)
    # These are estimates - actual tracking would require more detailed logging
    supervisor_cost = total_cost * 0.25   # Orchestration
    calculator_cost = total_cost * 0.40    # Most expensive operations
    extractor_cost = total_cost * 0.20     # Data retrieval
    validator_cost = total_cost * 0.15     # Validation
    
    return CrewAICostBreakdown(
        total_cost=total_cost,
        cost_per_question=cost_per_question,
        cost_per_conversation=cost_per_conversation,
        supervisor_cost=supervisor_cost,
        extractor_cost=extractor_cost,
        calculator_cost=calculator_cost,
        validator_cost=validator_cost
    ) 