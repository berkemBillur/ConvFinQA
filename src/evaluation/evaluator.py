"""End-to-end evaluation pipeline for ConvFinQA solutions."""

from typing import List, Dict, Union, Callable, Optional, Protocol
import time
import random
from pathlib import Path

try:
    from ..data.models import ConvFinQARecord
    from ..data.dataset import ConvFinQADataset
except ImportError:
    from data.models import ConvFinQARecord
    from data.dataset import ConvFinQADataset
from .metrics import (
    EvaluationResult,
    evaluate_conversation,
    aggregate_evaluation_results,
    print_evaluation_summary
)
from .executor import execute_dsl_program


class Predictor(Protocol):
    """Protocol for ConvFinQA prediction models."""
    
    def predict_turn(
        self,
        record: ConvFinQARecord,
        turn_index: int,
        conversation_history: List[Dict[str, str]]
    ) -> Union[float, str]:
        """Predict answer for a single conversation turn.
        
        Args:
            record: ConvFinQA record containing document and conversation.
            turn_index: Index of current turn (0-based).
            conversation_history: Previous turns with questions and answers.
            
        Returns:
            Predicted answer for the turn.
        """
        ...


class ConvFinQAEvaluator:
    """Main evaluation pipeline for ConvFinQA solutions."""
    
    def __init__(self, dataset: Optional[ConvFinQADataset] = None):
        """Initialise the evaluator.
        
        Args:
            dataset: ConvFinQA dataset instance. If None, creates a new one.
        """
        self.dataset = dataset or ConvFinQADataset()
        if not self.dataset.is_loaded:
            self.dataset.load()
    
    def quick_evaluation(
        self,
        predictor: Predictor,
        num_conversations: int = 10,
        split: str = 'dev',
        random_sample: bool = False,
        seed: Optional[int] = None
    ) -> EvaluationResult:
        """Quick evaluation on a small subset for rapid iteration.
        
        Args:
            predictor: Model to evaluate.
            num_conversations: Number of conversations to evaluate.
            split: Dataset split to use ('train' or 'dev').
            random_sample: Whether to use random sampling instead of first N.
            seed: Random seed for reproducible random sampling.
            
        Returns:
            Evaluation results.
        """
        all_records = self.dataset.get_split(split)
        
        # Select records based on sampling strategy
        if random_sample:
            if seed is not None:
                random.seed(seed)
            records = random.sample(all_records, min(num_conversations, len(all_records)))
            sampling_info = f"random sampling (seed: {seed})" if seed else "random sampling"
        else:
            records = all_records[:num_conversations]
            sampling_info = "deterministic (first N)"
        
        print(f"Running quick evaluation on {len(records)} conversations from {split} split")
        print(f"Sampling strategy: {sampling_info}")
        
        # Log which conversations were selected for transparency
        if len(records) <= 10:  # Only log for small sets to avoid clutter
            record_ids = [r.id for r in records]
            print(f"Selected conversations: {record_ids}")
        
        return self._evaluate_records(predictor, records)
    
    def comprehensive_evaluation(
        self,
        predictor: Predictor,
        split: str = 'dev'
    ) -> EvaluationResult:
        """Comprehensive evaluation on full dataset split.
        
        Args:
            predictor: Model to evaluate.
            split: Dataset split to use ('train' or 'dev').
            
        Returns:
            Evaluation results.
        """
        records = self.dataset.get_split(split)
        print(f"Running comprehensive evaluation on {len(records)} conversations from {split} split")
        
        return self._evaluate_records(predictor, records, show_progress=True)
    
    def _evaluate_records(
        self,
        predictor: Predictor,
        records: List[ConvFinQARecord],
        show_progress: bool = False
    ) -> EvaluationResult:
        """Evaluate predictor on a list of records.
        
        Args:
            predictor: Model to evaluate.
            records: List of ConvFinQA records.
            show_progress: Whether to show progress information.
            
        Returns:
            Evaluation results.
        """
        conversation_results = []
        start_time = time.time()
        
        for i, record in enumerate(records):
            if show_progress and i % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {i}/{len(records)} ({i/len(records)*100:.1f}%) - {elapsed:.1f}s elapsed")
            
            try:
                predicted_answers = self._predict_conversation(predictor, record)
                result = evaluate_conversation(record, predicted_answers)
                conversation_results.append(result)
                
            except Exception as e:
                conversation_results.append({
                    'error': f"Prediction failed: {str(e)}",
                    'accuracy': 0.0,
                    'correct_per_turn': [False] * len(record.dialogue.conv_questions)
                })
        
        if show_progress:
            elapsed = time.time() - start_time
            print(f"Evaluation completed in {elapsed:.1f}s")
        
        return aggregate_evaluation_results(conversation_results)
    
    def _predict_conversation(
        self,
        predictor: Predictor,
        record: ConvFinQARecord
    ) -> List[Union[float, str]]:
        """Get predictions for all turns in a conversation.
        
        Args:
            predictor: Model to use for predictions.
            record: ConvFinQA record.
            
        Returns:
            List of predicted answers for each turn.
        """
        predicted_answers = []
        conversation_history = []
        
        for turn_index in range(len(record.dialogue.conv_questions)):
            # Get prediction for this turn
            predicted_answer = predictor.predict_turn(record, turn_index, conversation_history)
            predicted_answers.append(predicted_answer)
            
            # Update conversation history
            question = record.dialogue.conv_questions[turn_index]
            conversation_history.append({
                'question': question,
                'answer': str(predicted_answer)
            })
        
        return predicted_answers


class GroundTruthPredictor:
    """Ground truth predictor that executes ground truth DSL programs."""
    
    def predict_turn(
        self,
        record: ConvFinQARecord,
        turn_index: int,
        conversation_history: List[Dict[str, str]]
    ) -> Union[float, str]:
        """Predict by executing the ground truth DSL program.
        
        Args:
            record: ConvFinQA record.
            turn_index: Current turn index.
            conversation_history: Previous conversation turns.
            
        Returns:
            Result of executing the ground truth DSL program.
        """
        if turn_index >= len(record.dialogue.turn_program):
            return "Turn index out of range"
        
        program = record.dialogue.turn_program[turn_index]
        return execute_dsl_program(program)


class SimpleBaselinePredictor:
    """Simple baseline that always predicts the first table value."""
    
    def predict_turn(
        self,
        record: ConvFinQARecord,
        turn_index: int,
        conversation_history: List[Dict[str, str]]
    ) -> Union[float, str]:
        """Predict by returning the first numeric value from the table.
        
        Args:
            record: ConvFinQA record.
            turn_index: Current turn index.
            conversation_history: Previous conversation turns.
            
        Returns:
            First numeric value found in the table, or 0.0 if none found.
        """
        # Simple baseline: return first numeric value from table
        table = record.doc.table
        
        for col_data in table.values():
            for value in col_data.values():
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    try:
                        return float(value.replace(',', '').replace('$', ''))
                    except ValueError:
                        continue
        
        return 0.0


def run_quick_test() -> None:
    """Run a quick test of the evaluation pipeline."""
    print("Running quick evaluation test...")
    
    evaluator = ConvFinQAEvaluator()
    
    # Test with ground truth predictor (should get ~100% accuracy)
    print("\nTesting Ground Truth Predictor:")
    ground_truth_result = evaluator.quick_evaluation(GroundTruthPredictor(), num_conversations=5)
    print_evaluation_summary(ground_truth_result)
    
    # Test with simple baseline (should get low accuracy)
    print("\nTesting Simple Baseline:")
    baseline_result = evaluator.quick_evaluation(SimpleBaselinePredictor(), num_conversations=5)
    print_evaluation_summary(baseline_result)


if __name__ == "__main__":
    run_quick_test() 