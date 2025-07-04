#!/usr/bin/env python3
"""Benchmark script for Hybrid Keyword Predictor approach.

This script evaluates the hybrid keyword-heuristic predictor performance
on the ConvFinQA dataset with detailed analysis and result tracking.
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import time
from datetime import datetime
from dataclasses import dataclass
import random

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.dataset import ConvFinQADataset
from predictors.hybrid_keyword_predictor import HybridKeywordPredictor
from evaluation.evaluator import ConvFinQAEvaluator, SimpleBaselinePredictor, GroundTruthPredictor
from evaluation.metrics import is_answer_correct

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HybridQuestionResult:
    """Detailed result for a single question in hybrid predictor evaluation."""
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
    execution_time: float
    matched_keywords: List[str]
    value_candidates_count: int


def collect_hybrid_detailed_results(
    evaluator: ConvFinQAEvaluator,
    predictor: HybridKeywordPredictor,
    num_conversations: int,
    random_sample: bool = False,
    seed: Optional[int] = None
) -> List[HybridQuestionResult]:
    """Collect detailed results for hybrid predictor evaluation."""
    dataset = evaluator.dataset
    
    # Get records based on sampling strategy
    if random_sample:
        if seed is not None:
            random.seed(seed)
        all_records = dataset.get_split('dev')
        records = random.sample(all_records, min(num_conversations, len(all_records)))
    else:
        records = dataset.get_split('dev')[:num_conversations]
    
    logger.info(f"Processing {len(records)} conversations with hybrid predictor...")
    
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
                
                # Get prediction with detailed reasoning
                prediction_result = predictor._predict_with_reasoning(
                    record, question, turn_idx, conversation_history
                )
                
                execution_time = time.time() - start_time
                
                # Get predicted answer
                predicted_answer = prediction_result.answer
                
                # Check correctness
                is_correct = is_answer_correct(predicted_answer, expected)
                
                # Get additional metadata
                operation_match = predictor._classify_question(question, conversation_history)
                matched_keywords = operation_match.matched_keywords if operation_match else []
                
                value_candidates = predictor._extract_values(record, question, conversation_history)
                value_candidates_count = len(value_candidates)
                
                # Create detailed result
                result = HybridQuestionResult(
                    record_id=record.id,
                    turn_index=turn_idx,
                    total_turns=len(questions),
                    question=question,
                    ground_truth=str(expected),
                    predicted_answer=str(predicted_answer),
                    is_correct=is_correct,
                    dsl_program=prediction_result.dsl_program,
                    operation_type=prediction_result.operation_type,
                    confidence=prediction_result.confidence,
                    reasoning=prediction_result.reasoning,
                    execution_time=execution_time,
                    matched_keywords=matched_keywords,
                    value_candidates_count=value_candidates_count
                )
                
                question_results.append(result)
                
                # Update conversation history
                conversation_history.append({
                    'question': question,
                    'answer': str(predicted_answer)
                })
                
            except Exception as e:
                logger.error(f"Error processing question {turn_idx}: {e}")
                # Create error result
                error_result = HybridQuestionResult(
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
                    execution_time=0.0,
                    matched_keywords=[],
                    value_candidates_count=0
                )
                question_results.append(error_result)
    
    logger.info(f"Processed {total_questions} questions from {len(records)} conversations")
    return question_results


def run_hybrid_benchmark(
    num_conversations: int = 50,
    random_sample: bool = False,
    seed: Optional[int] = None,
    save_results: bool = False,
    show_questions: bool = False,
    failures_only: bool = False,
    compare_baselines: bool = True
) -> Dict[str, Any]:
    """Run hybrid predictor benchmark evaluation."""
    logger.info("Starting Hybrid Keyword Predictor benchmark evaluation")
    logger.info(f"Target: >45% execution accuracy on {num_conversations} conversations")
    
    # Initialize components
    dataset = ConvFinQADataset()
    if not dataset.is_loaded:
        dataset.load()
    
    evaluator = ConvFinQAEvaluator(dataset)
    predictor = HybridKeywordPredictor()
    
    logger.info("✅ Components initialized")
    
    # Collect detailed results
    start_time = time.time()
    question_results = collect_hybrid_detailed_results(
        evaluator, predictor, num_conversations, random_sample, seed
    )
    total_time = time.time() - start_time
    
    # Calculate metrics
    total_questions = len(question_results)
    correct_questions = sum(1 for r in question_results if r.is_correct)
    accuracy = correct_questions / total_questions if total_questions > 0 else 0
    
    # Analyze operation types
    operation_breakdown = {}
    for result in question_results:
        op_type = result.operation_type
        if op_type not in operation_breakdown:
            operation_breakdown[op_type] = {'total': 0, 'correct': 0}
        operation_breakdown[op_type]['total'] += 1
        if result.is_correct:
            operation_breakdown[op_type]['correct'] += 1
    
    # Display results
    logger.info(f"\n{'='*60}")
    logger.info("HYBRID KEYWORD PREDICTOR BENCHMARK RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Conversations: {num_conversations}")
    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Correct: {correct_questions}")
    logger.info(f"Accuracy: {accuracy*100:.1f}%")
    logger.info(f"Execution time: {total_time:.1f}s")
    logger.info(f"Avg time per question: {total_time/total_questions:.3f}s")
    logger.info(f"Target (45%): {'✅ PASS' if accuracy >= 0.45 else '❌ BELOW TARGET'}")
    
    # Operation type breakdown
    logger.info(f"\nOperation Type Breakdown:")
    for op_type, stats in operation_breakdown.items():
        op_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        logger.info(f"  {op_type}: {stats['correct']}/{stats['total']} ({op_accuracy*100:.1f}%)")
    
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
            logger.info(f"Operation: {result.operation_type} | Confidence: {result.confidence:.3f}")
            logger.info(f"DSL: {result.dsl_program}")
            logger.info(f"Keywords: {result.matched_keywords}")
            logger.info(f"Time: {result.execution_time:.3f}s | Candidates: {result.value_candidates_count}")
            if not result.is_correct:
                logger.info(f"Reasoning: {result.reasoning}")
    
    # Compare with baselines if requested
    baseline_results = {}
    if compare_baselines:
        logger.info(f"\n{'='*60}")
        logger.info("BASELINE COMPARISONS")
        logger.info(f"{'='*60}")
        
        baselines = [
            ("Ground Truth", GroundTruthPredictor()),
            ("Simple Baseline", SimpleBaselinePredictor()),
        ]
        
        for name, baseline_predictor in baselines:
            logger.info(f"Evaluating {name}...")
            baseline_result = evaluator.quick_evaluation(
                predictor=baseline_predictor,
                num_conversations=num_conversations,
                split='dev',
                random_sample=random_sample,
                seed=seed
            )
            baseline_accuracy = baseline_result.execution_accuracy / 100.0
            baseline_results[name.lower().replace(' ', '_')] = baseline_accuracy
            logger.info(f"  {name} Accuracy: {baseline_accuracy*100:.1f}%")
    
    # Save results if requested
    if save_results:
        save_hybrid_results(
            question_results=question_results,
            accuracy=accuracy,
            operation_breakdown=operation_breakdown,
            baseline_results=baseline_results,
            num_conversations=num_conversations,
            random_sample=random_sample,
            seed=seed,
            total_time=total_time
        )
    
    return {
        "accuracy": accuracy,
        "total_questions": total_questions,
        "correct_questions": correct_questions,
        "operation_breakdown": operation_breakdown,
        "baseline_results": baseline_results,
        "execution_time": total_time,
        "question_results": question_results
    }


def save_hybrid_results(
    question_results: List[HybridQuestionResult],
    accuracy: float,
    operation_breakdown: Dict[str, Dict[str, int]],
    baseline_results: Dict[str, float],
    num_conversations: int,
    random_sample: bool,
    seed: Optional[int],
    total_time: float
) -> None:
    """Save hybrid benchmark results to timestamped folder."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path("results") / "hybrid" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save all results
        all_results_file = results_dir / "all_results.txt"
        with open(all_results_file, 'w', encoding='utf-8') as f:
            f.write("Hybrid Keyword Predictor - All Results\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(question_results, 1):
                status = "CORRECT" if result.is_correct else "INCORRECT"
                f.write(f"{i}. Record: {result.record_id}, Turn {result.turn_index}/{result.total_turns}\n")
                f.write(f"Question: \"{result.question}\"\n")
                f.write(f"Ground Truth: {result.ground_truth}\n")
                f.write(f"Predicted: {result.predicted_answer}\n")
                f.write(f"Result: {status}\n")
                f.write(f"DSL Program: {result.dsl_program}\n")
                f.write(f"Operation: {result.operation_type}\n")
                f.write(f"Confidence: {result.confidence:.3f}\n")
                f.write(f"Keywords: {result.matched_keywords}\n")
                f.write(f"Execution Time: {result.execution_time:.3f}s\n")
                f.write(f"Value Candidates: {result.value_candidates_count}\n")
                if not result.is_correct:
                    f.write(f"Reasoning: {result.reasoning}\n")
                f.write("-" * 80 + "\n\n")
        
        # Save failed results
        failed_results = [r for r in question_results if not r.is_correct]
        failed_results_file = results_dir / "failed_results.txt"
        with open(failed_results_file, 'w', encoding='utf-8') as f:
            f.write("Hybrid Keyword Predictor - Failed Results\n")
            f.write("=" * 80 + "\n\n")
            
            if not failed_results:
                f.write("No failed results - all questions answered correctly!\n")
            else:
                for i, result in enumerate(failed_results, 1):
                    f.write(f"{i}. Record: {result.record_id}, Turn {result.turn_index}/{result.total_turns}\n")
                    f.write(f"Question: \"{result.question}\"\n")
                    f.write(f"Ground Truth: {result.ground_truth}\n")
                    f.write(f"Predicted: {result.predicted_answer}\n")
                    f.write(f"DSL Program: {result.dsl_program}\n")
                    f.write(f"Operation: {result.operation_type}\n")
                    f.write(f"Confidence: {result.confidence:.3f}\n")
                    f.write(f"Keywords: {result.matched_keywords}\n")
                    f.write(f"Reasoning: {result.reasoning}\n")
                    f.write("-" * 80 + "\n\n")
        
        # Save passed results
        passed_results = [r for r in question_results if r.is_correct]
        passed_results_file = results_dir / "passed_results.txt"
        with open(passed_results_file, 'w', encoding='utf-8') as f:
            f.write("Hybrid Keyword Predictor - Passed Results\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(passed_results, 1):
                f.write(f"{i}. Record: {result.record_id}, Turn {result.turn_index}/{result.total_turns}\n")
                f.write(f"Question: \"{result.question}\"\n")
                f.write(f"Ground Truth: {result.ground_truth}\n")
                f.write(f"Predicted: {result.predicted_answer}\n")
                f.write(f"DSL Program: {result.dsl_program}\n")
                f.write(f"Operation: {result.operation_type}\n")
                f.write(f"Confidence: {result.confidence:.3f}\n")
                f.write("-" * 80 + "\n\n")
        
        # Save run metadata
        metadata_file = results_dir / "run_metadata.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("Hybrid Keyword Predictor - Run Metadata\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("RUN PARAMETERS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Conversations evaluated: {num_conversations}\n")
            f.write(f"Sampling strategy: {'Random' if random_sample else 'Deterministic'}\n")
            if random_sample and seed is not None:
                f.write(f"Random seed: {seed}\n")
            f.write(f"Total questions: {len(question_results)}\n")
            f.write(f"Passed questions: {len(passed_results)}\n")
            f.write(f"Failed questions: {len(failed_results)}\n")
            f.write(f"Total execution time: {total_time:.1f}s\n")
            f.write(f"Average time per question: {total_time/len(question_results):.3f}s\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Hybrid Accuracy: {accuracy*100:.1f}%\n")
            for name, baseline_acc in baseline_results.items():
                f.write(f"{name.replace('_', ' ').title()}: {baseline_acc*100:.1f}%\n")
            f.write("\n")
            
            f.write("OPERATION TYPE BREAKDOWN:\n")
            f.write("-" * 40 + "\n")
            for op_type, stats in operation_breakdown.items():
                op_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                f.write(f"{op_type}: {stats['correct']}/{stats['total']} ({op_accuracy*100:.1f}%)\n")
            f.write("\n")
            
            f.write("TARGET ACHIEVEMENT:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Target: 45.0%\n")
            f.write(f"Achieved: {accuracy*100:.1f}%\n")
            f.write(f"Status: {'PASS' if accuracy >= 0.45 else 'BELOW TARGET'}\n")
        
        logger.info(f"💾 Results saved to: {results_dir}/")
        logger.info(f"  - all_results.txt ({len(question_results)} questions)")
        logger.info(f"  - failed_results.txt ({len(failed_results)} questions)")
        logger.info(f"  - passed_results.txt ({len(passed_results)} questions)")
        logger.info(f"  - run_metadata.txt")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    """Main entry point for hybrid benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Hybrid Keyword Predictor benchmark evaluation")
    parser.add_argument('--conversations', type=int, default=50,
                       help='Number of conversations to evaluate (default: 50)')
    parser.add_argument('--random-sample', action='store_true',
                       help='Use random sampling instead of first N conversations')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible sampling (requires --random-sample)')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to timestamped results/hybrid/ folder')
    parser.add_argument('--show-questions', action='store_true',
                       help='Show question-by-question breakdown')
    parser.add_argument('--failures-only', action='store_true',
                       help='Show only incorrect predictions (requires --show-questions)')
    parser.add_argument('--no-baselines', action='store_true',
                       help='Skip baseline comparisons')
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
    logger.info("🚀 Starting Hybrid Keyword Predictor benchmark")
    logger.info(f"Configuration: {args.conversations} conversations, "
               f"{'random' if args.random_sample else 'deterministic'} sampling")
    
    try:
        results = run_hybrid_benchmark(
            num_conversations=args.conversations,
            random_sample=args.random_sample,
            seed=args.seed,
            save_results=args.save_results,
            show_questions=args.show_questions,
            failures_only=args.failures_only,
            compare_baselines=not args.no_baselines
        )
        
        accuracy = results["accuracy"] * 100
        if accuracy >= 45.0:
            logger.info(f"\n🎉 Hybrid benchmark PASSED: {accuracy:.1f}% accuracy")
            sys.exit(0)
        else:
            logger.warning(f"\n⚠️  Hybrid benchmark BELOW TARGET: {accuracy:.1f}% (target: 45%)")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 