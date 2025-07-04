#!/usr/bin/env python3
"""Benchmark script for Multi-Agent Predictor approach.

This script evaluates the CrewAI multi-agent predictor performance
on the ConvFinQA dataset with detailed cost tracking and agent analysis.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import signal
import atexit

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ConvFinQADataset
from src.evaluation.evaluator import ConvFinQAEvaluator
from src.evaluation.multi_agent_results import (
    CrewAIQuestionResult, 
    CrewAICostBreakdown,
    save_crewai_timestamped_results,
    convert_to_crewai_result,
    calculate_cost_breakdown
)
from src.utils.config import Config
from src.logger import get_logger
from src.predictors.multi_agent_predictor import ConvFinQAMultiAgentPredictor

logger = get_logger(__name__)

# Global variables for logging cleanup
_log_file_path = None
_log_file_handler = None
_original_stdout = None
_original_stderr = None


class TeeOutput:
    """Simple class to write to both file and original stream."""
    def __init__(self, file_obj, original_stream):
        self.file = file_obj
        self.original = original_stream
        
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.original.write(text)
        
    def flush(self):
        self.file.flush()
        self.original.flush()
        
    def __getattr__(self, name):
        return getattr(self.original, name)


def setup_logging_to_file(results_dir: Path) -> None:
    """Setup logging to capture all output to file in results directory."""
    global _log_file_path, _log_file_handler, _original_stdout, _original_stderr
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _log_file_path = results_dir / f"command_output_{timestamp}.txt"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    
    log_file = open(_log_file_path, 'w', encoding='utf-8')
    
    sys.stdout = TeeOutput(log_file, _original_stdout)
    sys.stderr = TeeOutput(log_file, _original_stderr)
    
    _log_file_handler = log_file
    
    print(f"📝 Command output will be saved to: {_log_file_path}")


def cleanup_logging() -> None:
    """Cleanup logging and ensure file is properly closed."""
    global _log_file_handler, _log_file_path, _original_stdout, _original_stderr
    
    if _log_file_handler:
        if _original_stdout:
            sys.stdout = _original_stdout
        if _original_stderr:
            sys.stderr = _original_stderr
            
        _log_file_handler.close()
        _log_file_handler = None
        
        if _log_file_path and _log_file_path.exists():
            print(f"✅ Command output saved to: {_log_file_path}")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully to save logs."""
    print(f"\n🛑 Interrupted by signal {signum}. Saving logs...")
    cleanup_logging()
    sys.exit(1)


# Register signal handlers and exit handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_logging)


def check_environment() -> bool:
    """Check if the environment is properly configured for multi-agent predictor."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'sk-your-openai-api-key-here':
        logger.error("❌ OpenAI API key not configured properly")
        logger.error("   Please set OPENAI_API_KEY in your .env file")
        logger.error("   Get your API key from: https://platform.openai.com/api-keys")
        return False
    
    logger.info("✅ Environment configured properly")
    return True


def create_multi_agent_predictor(config: Config) -> Optional[ConvFinQAMultiAgentPredictor]:
    """Create multi-agent predictor with proper error handling."""
    try:
        predictor = ConvFinQAMultiAgentPredictor(config)
        logger.info("✅ Multi-agent predictor created successfully")
        return predictor
    except ImportError as e:
        logger.error(f"❌ Multi-agent dependencies not available: {e}")
        logger.error("Please install dependencies: uv add crewai crewai-tools langchain-openai")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to create multi-agent predictor: {e}")
        return None


def collect_multi_agent_detailed_results(
    evaluator: ConvFinQAEvaluator,
    predictor: ConvFinQAMultiAgentPredictor,
    num_conversations: int,
    random_sample: bool = False,
    seed: Optional[int] = None
) -> List[CrewAIQuestionResult]:
    """Collect detailed multi-agent results for all questions."""
    dataset = evaluator.dataset
    
    # Get records based on sampling strategy
    if random_sample:
        import random
        if seed is not None:
            random.seed(seed)
        all_records = dataset.get_split('dev')
        records = random.sample(all_records, min(num_conversations, len(all_records)))
    else:
        records = dataset.get_split('dev')[:num_conversations]
    
    logger.info(f"Processing {len(records)} conversations...")
    
    question_results = []
    total_questions = 0
    
    for record_idx, record in enumerate(records):
        logger.info(f"Processing conversation {record_idx + 1}/{len(records)}: {record.id}")
        
        conversation_history = []
        questions = record.dialogue.conv_questions
        expected_answers = record.dialogue.executed_answers
        
        for turn_idx, (question, expected) in enumerate(zip(questions, expected_answers)):
            total_questions += 1
            logger.debug(f"  Turn {turn_idx + 1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Track execution time
                start_time = time.time()
                
                # Get prediction from multi-agent system
                predicted = predictor.predict_turn(record, turn_idx, conversation_history)
                
                execution_time = time.time() - start_time
                
                # Get multi-agent specific metadata
                config_hash = predictor.config_hash if hasattr(predictor, 'config_hash') else "unknown"
                
                # Estimate cost (this would ideally come from the predictor)
                estimated_cost = execution_time * 0.01  # Placeholder cost estimation
                
                # Check correctness
                from src.evaluation.metrics import is_answer_correct
                is_correct = is_answer_correct(predicted, expected)
                
                # Create detailed result
                result = convert_to_crewai_result(
                    record_id=record.id,
                    turn_index=turn_idx,
                    total_turns=len(questions),
                    question=question,
                    ground_truth=str(expected),
                    predicted_answer=str(predicted),
                    is_correct=is_correct,
                    dsl_program="",  # Would come from predictor if available
                    operation_type="multi_agent",
                    confidence=0.8,  # Default confidence for multi-agent
                    reasoning="",  # Would come from predictor if available
                    config_hash=config_hash,
                    agent_flow=["Supervisor", "Extractor", "Calculator", "Validator"],  # Default flow
                    execution_time=execution_time,
                    estimated_cost=estimated_cost,
                    fallback_used=False,
                    error_message=None
                )
                
                question_results.append(result)
                
                # Update conversation history
                conversation_history.append({
                    'question': question,
                    'answer': str(predicted)
                })
                
            except Exception as e:
                logger.error(f"Error processing question {turn_idx}: {e}")
                execution_time = time.time() - start_time
                
                # Create error result
                error_result = convert_to_crewai_result(
                    record_id=record.id,
                    turn_index=turn_idx,
                    total_turns=len(questions),
                    question=question,
                    ground_truth=str(expected),
                    predicted_answer="ERROR",
                    is_correct=False,
                    dsl_program="",
                    operation_type="error",
                    confidence=0.0,
                    reasoning=f"Error: {str(e)}",
                    config_hash="error",
                    agent_flow=["Error"],
                    execution_time=execution_time,
                    estimated_cost=0.0,
                    fallback_used=True,
                    error_message=str(e)
                )
                
                question_results.append(error_result)
    
    logger.info(f"Processed {total_questions} questions from {len(records)} conversations")
    return question_results


def run_multi_agent_benchmark(
    num_conversations: int = 10,
    random_sample: bool = False,
    seed: Optional[int] = None,
    save_results: bool = False,
    show_questions: bool = False,
    failures_only: bool = False
) -> Dict[str, Any]:
    """Run multi-agent benchmark evaluation with comprehensive tracking."""
    
    # Setup logging early if results will be saved
    results_dir_path = None
    timestamp = None
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir_path = Path("results") / "multi_agent" / timestamp
        setup_logging_to_file(results_dir_path)
    
    # Check environment
    if not check_environment():
        return {"error": "Environment not configured"}
    
    try:
        # Load configuration and create components
        config = Config("config/base.json")
        logger.info("✅ Configuration loaded")
        
        dataset = ConvFinQADataset()
        if not dataset.is_loaded:
            dataset.load()
        logger.info(f"✅ Dataset loaded: {len(dataset.get_split('dev'))} dev records")
        
        evaluator = ConvFinQAEvaluator(dataset)
        logger.info("✅ Evaluator created")
        
        # Create multi-agent predictor
        predictor = create_multi_agent_predictor(config)
        if not predictor:
            return {"error": "Failed to create multi-agent predictor"}
        
        # Collect detailed results
        logger.info("🤖 Starting multi-agent benchmark evaluation...")
        start_time = time.time()
        
        question_results = collect_multi_agent_detailed_results(
            evaluator, predictor, num_conversations, random_sample, seed
        )
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        total_questions = len(question_results)
        correct_questions = sum(1 for r in question_results if r.is_correct)
        accuracy = correct_questions / total_questions if total_questions > 0 else 0
        
        # Calculate cost breakdown
        cost_breakdown = calculate_cost_breakdown(question_results)
        
        # Get agent models for metadata
        crewai_config = config.get('models.crewai', {})
        agent_models = {
            'Supervisor': crewai_config.get('supervisor_model', 'gpt-4o'),
            'Extractor': crewai_config.get('extractor_model', 'gpt-4o-mini'),
            'Calculator': crewai_config.get('calculator_model', 'gpt-4o'),
            'Validator': crewai_config.get('validator_model', 'gpt-4o-mini')
        }
        
        # Analyze agent performance (simplified - based on agent_flow)
        agent_breakdown = {}
        for result in question_results:
            for agent in result.agent_flow:
                if agent not in agent_breakdown:
                    agent_breakdown[agent] = {'calls': 0, 'total_cost': 0.0, 'total_time': 0.0}
                agent_breakdown[agent]['calls'] += 1
                agent_breakdown[agent]['total_cost'] += result.estimated_cost / len(result.agent_flow)
                agent_breakdown[agent]['total_time'] += result.execution_time / len(result.agent_flow)
        
        # Display results
        logger.info(f"\n{'='*60}")
        logger.info("MULTI-AGENT BENCHMARK RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Conversations: {num_conversations}")
        logger.info(f"Total questions: {total_questions}")
        logger.info(f"Correct: {correct_questions}")
        logger.info(f"Accuracy: {accuracy*100:.1f}%")
        logger.info(f"Total cost: ${cost_breakdown.total_cost:.2f}")
        logger.info(f"Execution time: {total_time:.1f}s")
        logger.info(f"Avg cost per question: ${cost_breakdown.total_cost/total_questions:.4f}")
        logger.info(f"Avg time per question: {total_time/total_questions:.2f}s")
        logger.info(f"Target (45%): {'✅ PASS' if accuracy >= 0.45 else '❌ BELOW TARGET'}")
        
        # Agent performance breakdown
        if agent_breakdown:
            logger.info(f"\nAgent Performance Breakdown:")
            for agent, stats in agent_breakdown.items():
                avg_cost = stats['total_cost'] / stats['calls'] if stats['calls'] > 0 else 0
                avg_time = stats['total_time'] / stats['calls'] if stats['calls'] > 0 else 0
                logger.info(f"  {agent}: {stats['calls']} calls, ${avg_cost:.4f}/call, {avg_time:.2f}s/call")
        
        # Show question details if requested
        if show_questions:
            display_results = question_results
            if failures_only:
                display_results = [r for r in question_results if not r.is_correct]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"QUESTION-BY-QUESTION RESULTS ({'FAILURES ONLY' if failures_only else 'ALL QUESTIONS'})")
            logger.info(f"{'='*60}")
            
            for i, result in enumerate(display_results, 1):
                status = "✅ CORRECT" if result.is_correct else "❌ INCORRECT"
                logger.info(f"\n{i}. {result.record_id}, Turn {result.turn_index}/{result.total_turns}")
                logger.info(f"Q: {result.question}")
                logger.info(f"Expected: {result.ground_truth}")
                logger.info(f"Predicted: {result.predicted_answer}")
                logger.info(f"Status: {status}")
                logger.info(f"Cost: ${result.estimated_cost:.4f} | Time: {result.execution_time:.1f}s")
                if result.fallback_used:
                    logger.info(f"⚠️  Fallback used: {result.error_message}")
                if result.agent_flow:
                    logger.info(f"Agent flow: {' → '.join(result.agent_flow)}")
        
        # Save results if requested
        if save_results:
            config_hash = predictor.config_hash if hasattr(predictor, 'config_hash') else "unknown"
            results_dir = save_crewai_timestamped_results(
                question_results=question_results,
                config_hash=config_hash,
                agent_models=agent_models,
                num_conversations=num_conversations,
                random_sample=random_sample,
                seed=seed,
                cost_breakdown=cost_breakdown,
                timestamp=timestamp
            )
            logger.info(f"💾 Results saved to: {results_dir}/")
        
        return {
            "accuracy": accuracy,
            "total_questions": total_questions,
            "correct_questions": correct_questions,
            "cost_breakdown": cost_breakdown,
            "agent_breakdown": agent_breakdown,
            "execution_time": total_time,
            "question_results": question_results
        }
        
    except Exception as e:
        logger.error(f"❌ Multi-agent benchmark failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def main():
    """Main entry point for multi-agent benchmark."""
    parser = argparse.ArgumentParser(description="Run Multi-Agent Predictor benchmark evaluation")
    parser.add_argument('--conversations', type=int, default=5,
                       help='Number of conversations to evaluate (default: 5)')
    parser.add_argument('--random-sample', action='store_true',
                       help='Use random sampling instead of first N conversations')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible sampling (requires --random-sample)')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to timestamped results/multi_agent/ folder')
    parser.add_argument('--show-questions', action='store_true',
                       help='Show question-by-question breakdown')
    parser.add_argument('--failures-only', action='store_true',
                       help='Show only incorrect predictions (requires --show-questions)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.seed is not None and not args.random_sample:
        logger.warning("--seed specified without --random-sample, seed will be ignored")
    
    if args.failures_only and not args.show_questions:
        logger.warning("--failures-only requires --show-questions, enabling it automatically")
        args.show_questions = True
    
    # Run benchmark
    logger.info("🚀 Starting Multi-Agent Predictor benchmark")
    logger.info(f"Configuration: {args.conversations} conversations, "
               f"{'random' if args.random_sample else 'deterministic'} sampling")
    
    results = run_multi_agent_benchmark(
        num_conversations=args.conversations,
        random_sample=args.random_sample,
        seed=args.seed,
        save_results=args.save_results,
        show_questions=args.show_questions,
        failures_only=args.failures_only
    )
    
    if "error" in results:
        logger.error(f"❌ Benchmark failed: {results['error']}")
        sys.exit(1)
    else:
        accuracy = results["accuracy"] * 100
        if accuracy >= 45.0:
            logger.info(f"\n🎉 Multi-agent benchmark PASSED: {accuracy:.1f}% accuracy")
            sys.exit(0)
        else:
            logger.warning(f"\n⚠️  Multi-agent benchmark BELOW TARGET: {accuracy:.1f}% (target: 45%)")
            sys.exit(1)


if __name__ == "__main__":
    main() 