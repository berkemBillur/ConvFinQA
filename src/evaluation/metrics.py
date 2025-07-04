"""Evaluation metrics for ConvFinQA solutions."""

from typing import List, Dict, Union, Tuple, Optional
import math
from dataclasses import dataclass
from collections import defaultdict

try:
    from ..data.models import ConvFinQARecord
except ImportError:
    from data.models import ConvFinQARecord


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    total_questions: int
    correct_answers: int
    execution_accuracy: float
    breakdown_by_turn: Dict[int, float]
    breakdown_by_type: Dict[str, float]
    breakdown_by_length: Dict[int, float]
    error_analysis: Dict[str, int]
    

def calculate_execution_accuracy(
    predicted_answers: List[Union[float, str]],
    ground_truth_answers: List[Union[float, str]],
    tolerance: float = 1e-6
) -> float:
    """Calculate execution accuracy between predicted and ground truth answers.
    
    Args:
        predicted_answers: List of predicted answers.
        ground_truth_answers: List of ground truth answers.
        tolerance: Numerical tolerance for floating point comparison.
        
    Returns:
        Execution accuracy as a percentage (0-100).
    """
    if len(predicted_answers) != len(ground_truth_answers):
        raise ValueError("Predicted and ground truth lists must have same length")
    
    if not predicted_answers:
        return 0.0
    
    correct = 0
    
    for pred, truth in zip(predicted_answers, ground_truth_answers):
        if is_answer_correct(pred, truth, tolerance):
            correct += 1
    
    return (correct / len(predicted_answers)) * 100.0


def is_answer_correct(
    predicted: Union[float, str],
    ground_truth: Union[float, str],
    tolerance: float = 1e-6
) -> bool:
    """Check if a predicted answer matches the ground truth.
    
    Args:
        predicted: Predicted answer.
        ground_truth: Ground truth answer.
        tolerance: Numerical tolerance for floating point comparison.
        
    Returns:
        True if answers match within tolerance.
    """
    # Handle exact string matches
    if isinstance(predicted, str) and isinstance(ground_truth, str):
        return predicted.strip().lower() == ground_truth.strip().lower()
    
    # Try to convert both to numbers for numerical comparison
    try:
        pred_num = float(predicted)
        truth_num = float(ground_truth)
        
        # Handle special cases
        if math.isnan(pred_num) and math.isnan(truth_num):
            return True
        if math.isinf(pred_num) and math.isinf(truth_num):
            return pred_num == truth_num  # Both positive or both negative infinity
        
        # Adaptive tolerance based on financial calculation standards
        adaptive_tolerance = calculate_adaptive_tolerance(pred_num, truth_num, tolerance)
        
        # Numerical comparison with adaptive tolerance
        return abs(pred_num - truth_num) <= adaptive_tolerance
        
    except (ValueError, TypeError):
        # Fallback to string comparison
        return str(predicted).strip() == str(ground_truth).strip()


def evaluate_conversation(
    record: ConvFinQARecord,
    predicted_answers: List[Union[float, str]],
    tolerance: float = 1e-6
) -> Dict[str, Union[float, List[bool], str, int]]:
    """Evaluate predictions for a single conversation.
    
    Args:
        record: ConvFinQA record containing ground truth.
        predicted_answers: List of predicted answers for each turn.
        tolerance: Numerical tolerance for comparison.
        
    Returns:
        Dictionary with evaluation results for this conversation.
    """
    ground_truth = record.dialogue.executed_answers
    
    if len(predicted_answers) != len(ground_truth):
        return {
            'error': f"Length mismatch: predicted {len(predicted_answers)}, expected {len(ground_truth)}",
            'accuracy': 0.0,
            'correct_per_turn': [False] * len(ground_truth)
        }
    
    correct_per_turn = [
        is_answer_correct(pred, truth, tolerance)
        for pred, truth in zip(predicted_answers, ground_truth)
    ]
    
    accuracy = sum(correct_per_turn) / len(correct_per_turn) * 100.0 if correct_per_turn else 0.0
    
    return {
        'accuracy': accuracy,
        'correct_per_turn': correct_per_turn,
        'num_turns': len(correct_per_turn),
        'conversation_type': 'hybrid' if record.features.has_type2_question else 'simple'
    }


def aggregate_evaluation_results(
    conversation_results: List[Dict[str, Union[float, List[bool], str, int]]]
) -> EvaluationResult:
    """Aggregate results from multiple conversations.
    
    Args:
        conversation_results: List of per-conversation evaluation results.
        
    Returns:
        Aggregated evaluation result.
    """
    # Filter out error cases
    valid_results = [r for r in conversation_results if 'error' not in r]
    
    if not valid_results:
        return EvaluationResult(
            total_questions=0,
            correct_answers=0,
            execution_accuracy=0.0,
            breakdown_by_turn={},
            breakdown_by_type={},
            breakdown_by_length={},
            error_analysis={'total_errors': len(conversation_results)}
        )
    
    # Calculate overall metrics
    total_questions = sum(len(r['correct_per_turn']) for r in valid_results if isinstance(r['correct_per_turn'], list))
    total_correct = sum(sum(r['correct_per_turn']) for r in valid_results if isinstance(r['correct_per_turn'], list))
    overall_accuracy = (total_correct / total_questions * 100.0) if total_questions > 0 else 0.0
    
    # Breakdown by turn position
    turn_stats = defaultdict(list)
    for result in valid_results:
        correct_per_turn = result['correct_per_turn']
        if isinstance(correct_per_turn, list):
            for turn_idx, is_correct in enumerate(correct_per_turn):
                turn_stats[turn_idx].append(is_correct)
    
    breakdown_by_turn = {
        turn: sum(correct_list) / len(correct_list) * 100.0
        for turn, correct_list in turn_stats.items()
    }
    
    # Breakdown by conversation type
    type_stats = defaultdict(list)
    for result in valid_results:
        conv_type = result.get('conversation_type', 'unknown')
        correct_per_turn = result['correct_per_turn']
        if isinstance(correct_per_turn, list):
            type_stats[conv_type].extend(correct_per_turn)
    
    breakdown_by_type = {
        conv_type: sum(correct_list) / len(correct_list) * 100.0
        for conv_type, correct_list in type_stats.items()
    }
    
    # Breakdown by conversation length
    length_stats = defaultdict(list)
    for result in valid_results:
        length = result.get('num_turns')
        correct_per_turn = result['correct_per_turn']
        if isinstance(length, int) and isinstance(correct_per_turn, list):
            length_stats[length].extend(correct_per_turn)
    
    breakdown_by_length = {
        length: sum(correct_list) / len(correct_list) * 100.0
        for length, correct_list in length_stats.items()
    }
    
    # Error analysis
    error_count = len(conversation_results) - len(valid_results)
    error_analysis = {
        'total_errors': error_count,
        'valid_conversations': len(valid_results),
        'error_rate': error_count / len(conversation_results) * 100.0 if conversation_results else 0.0
    }
    
    return EvaluationResult(
        total_questions=total_questions,
        correct_answers=total_correct,
        execution_accuracy=overall_accuracy,
        breakdown_by_turn=breakdown_by_turn,
        breakdown_by_type=breakdown_by_type,
        breakdown_by_length=breakdown_by_length,
        error_analysis=error_analysis
    )


def calculate_adaptive_tolerance(
    predicted: float,
    ground_truth: float,
    base_tolerance: float = 1e-6
) -> float:
    """Calculate adaptive tolerance based on magnitude and financial calculation standards.
    
    Args:
        predicted: Predicted numerical value.
        ground_truth: Ground truth numerical value.
        base_tolerance: Base tolerance for very small numbers.
        
    Returns:
        Adaptive tolerance value.
    """
    # Get the magnitude of the ground truth value
    magnitude = abs(ground_truth)
    
    # Financial calculation tolerance levels:
    if magnitude == 0:
        # For zero values, use strict tolerance
        return base_tolerance
    elif magnitude < 0.01:
        # Small percentages/ratios: 0.1% relative tolerance (e.g., 0.00001 for 0.01)
        return max(magnitude * 0.001, base_tolerance)
    elif magnitude < 1:
        # Percentages/ratios: 0.5% relative tolerance (e.g., 0.005 for percentage 0.5)
        return max(magnitude * 0.005, base_tolerance)
    elif magnitude < 100:
        # Small to medium values: 0.1% relative tolerance (e.g., 0.06 for value 60)
        return max(magnitude * 0.001, 0.01)
    elif magnitude < 10000:
        # Large values: 0.05% relative tolerance
        return max(magnitude * 0.0005, 0.1)
    else:
        # Very large values: 0.01% relative tolerance
        return max(magnitude * 0.0001, 1.0)
        
    # Additional tolerance for floating point precision issues
    # If values are close but differ by tiny floating point errors
    relative_diff = abs(predicted - ground_truth) / magnitude if magnitude > 0 else 0
    if relative_diff < 0.001:  # Less than 0.1% difference
        return max(magnitude * 0.001, base_tolerance)


def print_evaluation_summary(result: EvaluationResult) -> None:
    """Print a formatted summary of evaluation results.
    
    Args:
        result: Evaluation result to summarise.
    """
    print(f"Execution Accuracy: {result.execution_accuracy:.2f}%")
    print(f"Total Questions: {result.total_questions}")
    print(f"Correct Answers: {result.correct_answers}")
    
    if result.breakdown_by_type:
        print("\nBreakdown by Conversation Type:")
        for conv_type, accuracy in result.breakdown_by_type.items():
            print(f"  {conv_type.title()}: {accuracy:.2f}%")
    
    if result.breakdown_by_length:
        print("\nBreakdown by Conversation Length:")
        for length in sorted(result.breakdown_by_length.keys()):
            accuracy = result.breakdown_by_length[length]
            print(f"  {length} turns: {accuracy:.2f}%")
    
    if result.breakdown_by_turn:
        print("\nBreakdown by Turn Position:")
        for turn in sorted(result.breakdown_by_turn.keys()):
            accuracy = result.breakdown_by_turn[turn]
            print(f"  Turn {turn + 1}: {accuracy:.2f}%")
    
    if result.error_analysis['total_errors'] > 0:
        print(f"\nErrors: {result.error_analysis['total_errors']} ({result.error_analysis['error_rate']:.1f}%)") 