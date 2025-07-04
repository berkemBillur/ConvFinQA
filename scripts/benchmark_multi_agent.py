#!/usr/bin/env python3
"""Enhanced benchmark script for Multi-Agent Predictor with comprehensive tracking.

This script extends the original benchmark with complete configuration capture,
performance analysis, and enhanced tracking capabilities.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import signal
import atexit
import random
import traceback

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
from src.utils.config import Config, APIKeyManager
from src.logger import get_logger
from src.predictors.multi_agent.predictor import ConvFinQAMultiAgentPredictorV2 as ConvFinQAMultiAgentPredictor

# Enhanced tracking imports
from src.utils.enhanced_tracker import (
    get_enhanced_tracker,
    PerformanceResults,
    CompleteExperimentSnapshot
)

logger = get_logger(__name__)

# Global variables for logging cleanup
_log_file_path = None
_log_file_handler = None
_original_stdout = None
_original_stderr = None
_current_experiment_id = None


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
    _log_file_path = results_dir / f"run_time_logs.txt"
    
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
    """Handle Ctrl+C gracefully to save logs and tracking data."""
    print(f"\n🛑 Interrupted by signal {signum}. Saving logs and tracking data...")
    cleanup_logging()
    sys.exit(1)


# Register signal handlers and exit handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_logging)


def check_environment() -> bool:
    """Check if the environment is properly configured for multi-agent predictor."""
    api_key = APIKeyManager.load_openai_key()
    if not api_key:
        logger.error("❌ OpenAI API key not configured properly")
        logger.error("   Please set up your API key using one of these methods:")
        logger.error("   1. Create config/api_keys.json with your key")
        logger.error("   2. Set OPENAI_API_KEY environment variable")
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


def analyze_question_types(records: List[Any]) -> Dict[str, int]:
    """Analyze and categorize question types in the dataset."""
    question_types = {
        'calculation': 0,
        'lookup': 0,
        'comparison': 0,
        'temporal': 0,
        'percentage': 0,
        'other': 0
    }
    
    for record in records:
        for question in record.dialogue.conv_questions:
            q_lower = question.lower()
            
            # Calculation questions
            if any(word in q_lower for word in ['calculate', 'compute', 'difference', 'change', 'total', 'sum']):
                question_types['calculation'] += 1
            # Lookup questions
            elif any(word in q_lower for word in ['what was', 'what is', 'how much', 'what were']):
                question_types['lookup'] += 1
            # Comparison questions
            elif any(word in q_lower for word in ['compare', 'ratio', 'versus', 'higher', 'lower']):
                question_types['comparison'] += 1
            # Temporal questions
            elif any(word in q_lower for word in ['year', 'quarter', 'period', 'previous', 'next']):
                question_types['temporal'] += 1
            # Percentage questions
            elif any(word in q_lower for word in ['percent', '%', 'percentage', 'rate']):
                question_types['percentage'] += 1
            else:
                question_types['other'] += 1
    
    return question_types


def collect_enhanced_detailed_results(
    evaluator: ConvFinQAEvaluator,
    predictor: ConvFinQAMultiAgentPredictor,
    num_conversations: int,
    random_sample: bool = False,
    seed: Optional[int] = None,
    experiment_snapshot: Optional[CompleteExperimentSnapshot] = None,
    use_six_agents: bool = False
) -> Tuple[List[CrewAIQuestionResult], PerformanceResults]:
    """Collect detailed multi-agent results with enhanced tracking."""
    dataset = evaluator.dataset
    
    # Get records based on sampling strategy
    if random_sample:
        if seed is not None:
            random.seed(seed)
        all_records = dataset.get_split('dev')
        records = random.sample(all_records, min(num_conversations, len(all_records)))
        logger.info(f"🎲 Random sampling with seed {seed}: {len(records)} conversations")
    else:
        records = dataset.get_split('dev')[:num_conversations]
        logger.info(f"📋 Sequential sampling: {len(records)} conversations")
    
    logger.info(f"Processing {len(records)} conversations...")
    
    question_results = []
    total_questions = 0
    correct_answers = 0
    total_execution_time = 0.0
    total_estimated_cost = 0.0
    failures = []
    error_patterns = {}
    
    # Ensure we have an agent roster
    agent_roster = ['manager','extractor','reasoner','extraction_critic','calculation_critic','synthesiser'] if use_six_agents else ['supervisor','extractor','calculator','validator']

    # Agent-specific tracking dict
    agent_performance = {
        name: {'executions': 0, 'successes': 0, 'total_time': 0.0, 'cost': 0.0}
        for name in agent_roster
    }
    
    # Question type tracking
    question_type_performance = {}
    
    for record_idx, record in enumerate(records):
        logger.info(f"Processing conversation {record_idx + 1}/{len(records)}: {record.id}")
        
        conversation_history = []
        questions = record.dialogue.conv_questions
        expected_answers = record.dialogue.executed_answers
        
        for turn_idx, (question, expected) in enumerate(zip(questions, expected_answers)):
            total_questions += 1
            logger.debug(f"  Turn {turn_idx + 1}/{len(questions)}: {question[:50]}...")
            
            # Determine question type
            q_type = categorize_question(question)
            if q_type not in question_type_performance:
                question_type_performance[q_type] = {
                    'total': 0, 'correct': 0, 'total_time': 0.0, 'total_cost': 0.0
                }
            
            try:
                # Track execution time
                start_time = time.time()
                
                # Get prediction from multi-agent system
                predicted = predictor.predict_turn(record, turn_idx, conversation_history)
                
                execution_time = time.time() - start_time
                total_execution_time += execution_time
                
                # Estimate cost (enhanced estimation)
                estimated_cost = estimate_turn_cost(question, execution_time)
                total_estimated_cost += estimated_cost
                
                # Check correctness with financial-appropriate tolerance
                from src.evaluation.metrics import is_answer_correct
                # Use more reasonable tolerance for financial calculations (3 significant figures)
                financial_tolerance = 0.001  # 0.1% relative tolerance
                is_correct = is_answer_correct(predicted, expected, tolerance=financial_tolerance)
                
                if is_correct:
                    correct_answers += 1
                    question_type_performance[q_type]['correct'] += 1
                
                question_type_performance[q_type]['total'] += 1
                question_type_performance[q_type]['total_time'] += execution_time
                question_type_performance[q_type]['total_cost'] += estimated_cost
                
                # Distribute execution time & cost evenly across agents for approximate accounting
                n_agents = len(agent_performance)
                for agent_name in agent_performance:
                    agent_performance[agent_name]['executions'] += 1
                    agent_performance[agent_name]['total_time'] += execution_time / n_agents
                    agent_performance[agent_name]['cost'] += estimated_cost / n_agents
                    if is_correct:
                        agent_performance[agent_name]['successes'] += 1
                
                # Create detailed result
                result = convert_to_crewai_result(
                    record_id=record.id,
                    turn_index=turn_idx,
                    total_turns=len(questions),
                    question=question,
                    ground_truth=str(expected),
                    predicted_answer=str(predicted),
                    is_correct=is_correct,
                    dsl_program="",  # Would be enhanced to capture actual DSL
                    operation_type=q_type,
                    confidence=1.0,  # Would be enhanced to capture actual confidence
                    reasoning="",  # Would be enhanced to capture reasoning
                    config_hash=getattr(predictor, 'config_hash', 'unknown'),
                    agent_flow=agent_roster,
                    execution_time=execution_time,
                    estimated_cost=estimated_cost,
                    fallback_used=False
                )
                
                question_results.append(result)
                
                # Update conversation history
                conversation_history.append({
                    'question': question,
                    'answer': str(predicted)
                })
                
                logger.debug(f"    ✅ Predicted: {predicted}, Expected: {expected}, Correct: {is_correct}")
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                
                logger.error(f"    ❌ Failed: {error_msg}")
                failures.append(f"Q{total_questions}: {error_msg}")
                
                # Track error patterns
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
                
                # Create failed result
                result = convert_to_crewai_result(
                    record_id=record.id,
                    turn_index=turn_idx,
                    total_turns=len(questions),
                    question=question,
                    ground_truth=str(expected),
                    predicted_answer="ERROR",
                    is_correct=False,
                    dsl_program="",
                    operation_type=q_type,
                    confidence=0.0,
                    reasoning=error_msg,
                    config_hash=getattr(predictor, 'config_hash', 'unknown'),
                    agent_flow=agent_roster,
                    execution_time=0.0,
                    estimated_cost=0.0,
                    fallback_used=False,
                    error_message=error_msg
                )
                
                question_results.append(result)
                
                question_type_performance[q_type]['total'] += 1
    
    # Calculate performance metrics
    accuracy_rate = correct_answers / total_questions if total_questions > 0 else 0.0
    avg_execution_time = total_execution_time / total_questions if total_questions > 0 else 0.0
    avg_cost = total_estimated_cost / total_questions if total_questions > 0 else 0.0
    failure_rate = len(failures) / total_questions if total_questions > 0 else 0.0
    
    # Process agent performance
    processed_agent_performance = {}
    for agent_name, perf in agent_performance.items():
        processed_agent_performance[agent_name] = {
            'total_executions': perf['executions'],
            'successful_executions': perf['successes'],
            'success_rate': perf['successes'] / perf['executions'] if perf['executions'] > 0 else 0.0,
            'avg_execution_time': perf['total_time'] / perf['executions'] if perf['executions'] > 0 else 0.0,
            'estimated_cost': perf['cost']
        }
    
    # Process question type performance
    processed_qt_performance = {}
    for q_type, perf in question_type_performance.items():
        processed_qt_performance[q_type] = {
            'total_questions': perf['total'],
            'correct_answers': perf['correct'],
            'accuracy_rate': perf['correct'] / perf['total'] if perf['total'] > 0 else 0.0,
            'avg_execution_time': perf['total_time'] / perf['total'] if perf['total'] > 0 else 0.0,
            'avg_cost': perf['total_cost'] / perf['total'] if perf['total'] > 0 else 0.0
        }
    
    # Create performance results
    performance_results = PerformanceResults(
        total_questions=total_questions,
        correct_answers=correct_answers,
        accuracy_rate=accuracy_rate,
        total_execution_time=total_execution_time,
        avg_execution_time_per_question=avg_execution_time,
        total_estimated_cost=total_estimated_cost,
        avg_cost_per_question=avg_cost,
        agent_performance=processed_agent_performance,
        question_type_performance=processed_qt_performance,
        failure_count=len(failures),
        failure_rate=failure_rate,
        error_patterns=error_patterns,
        common_failures=failures[:5],  # Top 5 failures
        cost_by_agent={name: perf['estimated_cost'] for name, perf in processed_agent_performance.items()},
        cost_by_question_type={q_type: perf['avg_cost'] * perf['total_questions'] 
                              for q_type, perf in processed_qt_performance.items()}
    )
    
    return question_results, performance_results


def categorize_question(question: str) -> str:
    """Categorize a question by type."""
    q_lower = question.lower()
    
    if any(word in q_lower for word in ['calculate', 'compute', 'difference', 'change']):
        return 'calculation'
    elif any(word in q_lower for word in ['what was', 'what is', 'how much']):
        return 'lookup'
    elif any(word in q_lower for word in ['compare', 'ratio', 'versus']):
        return 'comparison'
    elif any(word in q_lower for word in ['percent', '%', 'percentage']):
        return 'percentage'
    else:
        return 'other'


def estimate_turn_cost(question: str, execution_time: float) -> float:
    """Enhanced cost estimation based on question complexity and execution time."""
    # Base cost estimation
    base_cost = execution_time * 0.001  # $0.001 per second
    
    # Adjust based on question complexity
    complexity_multiplier = 1.0
    
    q_lower = question.lower()
    
    # Complex calculation questions cost more
    if any(word in q_lower for word in ['calculate', 'compute', 'ratio']):
        complexity_multiplier = 1.5
    
    # Long questions cost more
    if len(question) > 100:
        complexity_multiplier *= 1.2
    
    # Multi-step questions cost more
    if any(word in q_lower for word in ['then', 'after', 'next', 'subsequently']):
        complexity_multiplier *= 1.3
    
    return base_cost * complexity_multiplier


def run_enhanced_multi_agent_benchmark(
    num_conversations: int = 10,
    random_sample: bool = False,
    seed: Optional[int] = None,
    save_results: bool = False,
    show_questions: bool = False,
    failures_only: bool = False,
    notes: str = "",
    use_six_agents: bool = False
) -> Dict[str, Any]:
    """Run enhanced multi-agent benchmark with comprehensive tracking."""
    global _current_experiment_id
    
    logger.info("🚀 Starting Enhanced Multi-Agent Benchmark")
    logger.info(f"📊 Configuration: {num_conversations} conversations, random={random_sample}, seed={seed}")
    
    # Initialize enhanced tracker
    tracker = get_enhanced_tracker()
    
    # Load configuration and dataset
    config = Config()
    if use_six_agents:
        # Temporarily override the config for six-agent mode
        config.update("models.use_six_agents", True)
        logger.info("🚀 Six-agent mode enabled")
    else:
        logger.info("🔧 Three-agent mode (default)")
    
    evaluator = ConvFinQAEvaluator()
    
    # Check environment
    if not check_environment():
        return {'error': 'Environment not properly configured'}
    
    # Create predictor
    predictor = create_multi_agent_predictor(config)
    if not predictor:
        return {'error': 'Failed to create multi-agent predictor'}
    
    # Get records for dataset configuration
    dataset = evaluator.dataset
    if random_sample:
        if seed is not None:
            random.seed(seed)
        all_records = dataset.get_split('dev')
        records = random.sample(all_records, min(num_conversations, len(all_records)))
    else:
        records = dataset.get_split('dev')[:num_conversations]
    
    # Capture configuration snapshot
    try:
        # Get agent configurations from predictor
        from src.predictors.multi_agent.agents import build_agents
        agents = build_agents(config)
        
        # Prepare dataset info
        dataset_info = {
            'total_conversations': len(dataset.get_split('dev')),
            'sample_size': len(records),
            'sampling_strategy': 'random' if random_sample else 'sequential',
            'random_seed': seed,
            'conversation_ids': [record.id for record in records],
            'records': records
        }
        
        # Create experiment snapshot
        experiment_snapshot = tracker.create_experiment_snapshot(
            agents=agents,
            crew_config=config.get('crewai', {}),
            dataset_info=dataset_info,
            notes=notes
        )
        
        _current_experiment_id = experiment_snapshot.experiment_id
        
        logger.info(f"📸 Created experiment snapshot: {experiment_snapshot.experiment_id}")
        logger.info(f"🔑 Configuration hash: {experiment_snapshot.config_hash}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create experiment snapshot: {e}")
        logger.error(traceback.format_exc())
        return {'error': f'Failed to create experiment snapshot: {e}'}
    
    # Setup results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = tracker.results_dir / timestamp
    
    if save_results:
        setup_logging_to_file(results_dir)
    
    try:
        # Run benchmark with enhanced tracking
        start_time = time.time()
        
        question_results, performance_results = collect_enhanced_detailed_results(
            evaluator=evaluator,
            predictor=predictor,
            num_conversations=num_conversations,
            random_sample=random_sample,
            seed=seed,
            experiment_snapshot=experiment_snapshot,
            use_six_agents=use_six_agents
        )
        
        total_benchmark_time = time.time() - start_time
        
        # Update experiment with results
        tracker.update_experiment_results(
            experiment_id=experiment_snapshot.experiment_id,
            performance_results=performance_results
        )
        
        logger.info(f"📊 Updated experiment {experiment_snapshot.experiment_id} with results")
        
        # Calculate summary metrics
        accuracy = performance_results.accuracy_rate
        avg_execution_time = performance_results.avg_execution_time_per_question
        total_cost = performance_results.total_estimated_cost
        
        # Print enhanced summary
        print("\n" + "="*80)
        print("🎯 ENHANCED MULTI-AGENT BENCHMARK RESULTS")
        print("="*80)
        print(f"📋 Experiment ID: {experiment_snapshot.experiment_id}")
        print(f"🔑 Config Hash: {experiment_snapshot.config_hash}")
        print(f"📊 Total Questions: {performance_results.total_questions}")
        print(f"✅ Correct Answers: {performance_results.correct_answers}")
        print(f"🎯 Accuracy: {accuracy:.2%}")
        print(f"⏱️  Avg Time/Question: {avg_execution_time:.2f}s")
        print(f"💰 Total Cost: ${total_cost:.4f}")
        print(f"💸 Avg Cost/Question: ${performance_results.avg_cost_per_question:.4f}")
        print(f"❌ Failure Rate: {performance_results.failure_rate:.2%}")
        print(f"⏰ Total Benchmark Time: {total_benchmark_time:.1f}s")
        
        # Agent performance breakdown
        print("\n🤖 AGENT PERFORMANCE BREAKDOWN:")
        print("-" * 60)
        for agent_name, perf in performance_results.agent_performance.items():
            print(f"{agent_name:12}: Success={perf['success_rate']:.2%}, "
                  f"Time={perf['avg_execution_time']:.2f}s, Cost=${perf['estimated_cost']:.4f}")
        
        # Question type performance
        print("\n❓ QUESTION TYPE PERFORMANCE:")
        print("-" * 60)
        for q_type, perf in performance_results.question_type_performance.items():
            print(f"{q_type:12}: {perf['correct_answers']}/{perf['total_questions']} "
                  f"({perf['accuracy_rate']:.2%}), Time={perf['avg_execution_time']:.2f}s")
        
        # Error analysis
        if performance_results.error_patterns:
            print("\n🚨 ERROR PATTERNS:")
            print("-" * 40)
            for error_type, count in performance_results.error_patterns.items():
                print(f"{error_type}: {count} occurrences")
        
        # Save results if requested
        if save_results:
            # Calculate cost breakdown
            cost_breakdown = calculate_cost_breakdown(question_results)
            
            save_crewai_timestamped_results(
                question_results=question_results,
                config_hash=experiment_snapshot.config_hash,
                agent_models={'supervisor': 'gpt-4o', 'extractor': 'gpt-4o-mini', 'calculator': 'gpt-4o', 'validator': 'gpt-4o-mini'},
                num_conversations=num_conversations,
                random_sample=random_sample,
                seed=seed,
                cost_breakdown=cost_breakdown,
                timestamp=timestamp,
                results_base_dir=str(tracker.results_dir)
            )
            
            logger.info(f"💾 Results saved to: {results_dir}")
        
        print(f"\n📈 View results in dashboard: streamlit run scripts/dashboard.py")
        print("="*80)
        
        return {
            'experiment_id': experiment_snapshot.experiment_id,
            'config_hash': experiment_snapshot.config_hash,
            'accuracy': accuracy,
            'total_questions': performance_results.total_questions,
            'correct_answers': performance_results.correct_answers,
            'avg_execution_time': avg_execution_time,
            'total_cost': total_cost,
            'failure_rate': performance_results.failure_rate,
            'agent_performance': performance_results.agent_performance,
            'question_type_performance': performance_results.question_type_performance,
            'results_dir': str(results_dir) if save_results else None
        }
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        logger.error(traceback.format_exc())
        return {'error': f'Benchmark failed: {e}'}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Agent ConvFinQA Benchmark with Comprehensive Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with tracking
  python scripts/benchmark_multi_agent.py -n 5
  
  # Test six-agent architecture
  python scripts/benchmark_multi_agent.py -n 5 --six-agents
  
  # Random sampling with seed and notes
  python scripts/benchmark_multi_agent.py -n 10 --random --seed 42 --notes "Testing new configuration"
  
  # Full run with saved results
  python scripts/benchmark_multi_agent.py -n 20 --save --show-questions
  
  # Compare three vs six agents
  python scripts/benchmark_multi_agent.py -n 10 --save --notes "Three-agent baseline"
  python scripts/benchmark_multi_agent.py -n 10 --save --six-agents --notes "Six-agent comparison"
  
  # View dashboard after running
  streamlit run scripts/dashboard.py
        """
    )
    
    parser.add_argument(
        "-n", "--num-conversations",
        type=int,
        default=10,
        help="Number of conversations to process (default: 10)"
    )
    
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random sampling instead of sequential"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling"
    )
    
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save detailed results to timestamped directory"
    )
    
    parser.add_argument(
        "--show-questions",
        action="store_true",
        help="Include full questions in saved results"
    )
    
    parser.add_argument(
        "--failures-only",
        action="store_true", 
        help="Save only failed predictions"
    )
    
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Notes to include with experiment snapshot"
    )
    
    parser.add_argument(
        "--six-agents",
        action="store_true",
        help="Use six-agent architecture instead of three-agent (experimental)"
    )
    
    args = parser.parse_args()
    
    # Run enhanced benchmark
    result = run_enhanced_multi_agent_benchmark(
        num_conversations=args.num_conversations,
        random_sample=args.random,
        seed=args.seed,
        save_results=args.save,
        show_questions=args.show_questions,
        failures_only=args.failures_only,
        notes=args.notes,
        use_six_agents=args.six_agents
    )
    
    if 'error' in result:
        print(f"❌ Benchmark failed: {result['error']}")
        sys.exit(1)
    else:
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📊 Experiment ID: {result.get('experiment_id', 'N/A')}")
        print(f"🎯 Final Accuracy: {result.get('accuracy', 0):.2%}")


if __name__ == "__main__":
    main() 