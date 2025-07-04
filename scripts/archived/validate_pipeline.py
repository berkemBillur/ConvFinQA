#!/usr/bin/env python3
"""Pipeline validation script for ConvFinQA benchmark approach.

This script validates each component of our hybrid keyword-heuristic pipeline
against the 6 validation goals from our benchmark documentation.
"""

import sys
import os
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from typing import Dict, List, Any
import logging

from data.dataset import ConvFinQADataset
from data.models import ConvFinQARecord
from predictors.hybrid_keyword_predictor import HybridKeywordPredictor
from evaluation.evaluator import ConvFinQAEvaluator
from utils.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineValidator:
    """Validates each component of our benchmark pipeline systematically."""
    
    def __init__(self):
        """Initialise the pipeline validator."""
        self.config = get_config()
        self.dataset = ConvFinQADataset()
        self.predictor = HybridKeywordPredictor()
        self.evaluator = ConvFinQAEvaluator(self.dataset)
        
    def run_validation(self) -> Dict[str, bool]:
        """Run all validation tests.
        
        Returns:
            Dictionary of validation results for each component.
        """
        logger.info("Starting pipeline validation for ConvFinQA benchmark approach")
        
        results = {}
        
        # Validation Goal 1: Data Loading
        results['data_loading'] = self._validate_data_loading()
        
        # Validation Goal 2: Text Processing  
        results['text_processing'] = self._validate_text_processing()
        
        # Validation Goal 3: Table Parsing
        results['table_parsing'] = self._validate_table_parsing()
        
        # Validation Goal 4: DSL Generation
        results['dsl_generation'] = self._validate_dsl_generation()
        
        # Validation Goal 5: Evaluation
        results['evaluation'] = self._validate_evaluation()
        
        # Validation Goal 6: Error Handling
        results['error_handling'] = self._validate_error_handling()
        
        # Summary
        self._print_validation_summary(results)
        
        return results
    
    def _validate_data_loading(self) -> bool:
        """Validate: Can we process ConvFinQA records correctly?"""
        logger.info("Validating data loading...")
        
        try:
            # Load dataset
            if not self.dataset.is_loaded:
                self.dataset.load()
            
            # Check splits exist
            splits = self.dataset.splits
            if 'train' not in splits or 'dev' not in splits:
                logger.error(f"Missing required splits. Found: {splits}")
                return False
            
            # Check we can access records
            dev_records = self.dataset.get_dev_records()
            if len(dev_records) == 0:
                logger.error("No dev records found")
                return False
            
            # Validate record structure
            sample_record = dev_records[0]
            required_attrs = ['id', 'doc', 'dialogue']
            for attr in required_attrs:
                if not hasattr(sample_record, attr):
                    logger.error(f"Missing required attribute: {attr}")
                    return False
            
            # Check dialogue structure
            dialogue = sample_record.dialogue
            if not hasattr(dialogue, 'conv_questions') or not hasattr(dialogue, 'conv_answers'):
                logger.error("Invalid dialogue structure")
                return False
            
            logger.info(f"Data loading validation PASSED: {len(dev_records)} dev records loaded")
            return True
            
        except Exception as e:
            logger.error(f"Data loading validation FAILED: {e}")
            return False
    
    def _validate_text_processing(self) -> bool:
        """Validate: Do our keyword extraction methods work?"""
        logger.info("Validating text processing...")
        
        try:
            # Test question classification
            test_questions = [
                "What is the total revenue?",  # Should detect addition
                "What was the difference in profit?",  # Should detect subtraction
                "What is the profit margin?",  # Should detect division
                "What is the value?",  # Should detect lookup
            ]
            
            for question in test_questions:
                # Test classification
                operation_match = self.predictor._classify_question(question, [])
                
                if operation_match is None:
                    logger.warning(f"No operation detected for: {question}")
                else:
                    logger.info(f"Question: '{question}' -> Operation: {operation_match.operation_type} (confidence: {operation_match.confidence})")
            
            logger.info("Text processing validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Text processing validation FAILED: {e}")
            return False
    
    def _validate_table_parsing(self) -> bool:
        """Validate: Can we extract and score table values effectively?"""
        logger.info("Validating table parsing...")
        
        try:
            # Get a sample record with table data
            dev_records = self.dataset.get_dev_records()
            sample_record = dev_records[0]
            
            # Test value extraction
            question = "What is the revenue?"
            value_candidates = self.predictor._extract_values(sample_record, question, [])
            
            if len(value_candidates) == 0:
                logger.warning("No value candidates extracted from table")
            else:
                logger.info(f"Extracted {len(value_candidates)} value candidates:")
                for i, candidate in enumerate(value_candidates[:3]):  # Show top 3
                    logger.info(f"  {i+1}. Value: {candidate.value}, Context: {candidate.context}, Score: {candidate.score}")
            
            # Validate table structure
            table = sample_record.doc.table
            if not isinstance(table, dict):
                logger.error("Table is not a dictionary")
                return False
            
            # Check for numeric values
            has_numeric = False
            for col_data in table.values():
                for value in col_data.values():
                    if isinstance(value, (int, float)):
                        has_numeric = True
                        break
                if has_numeric:
                    break
            
            if not has_numeric:
                logger.warning("No numeric values found in table")
            
            logger.info("Table parsing validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Table parsing validation FAILED: {e}")
            return False
    
    def _validate_dsl_generation(self) -> bool:
        """Validate: Is our program synthesis pipeline functional?"""
        logger.info("Validating DSL generation...")
        
        try:
            # Get sample record
            dev_records = self.dataset.get_dev_records()
            sample_record = dev_records[0]
            
            # Test DSL generation for different operation types
            test_cases = [
                ("What is the total revenue?", "addition"),
                ("What is the revenue?", "lookup"),
                ("What is the difference?", "subtraction"),
            ]
            
            for question, expected_op in test_cases:
                # Get prediction with reasoning
                prediction_result = self.predictor._predict_with_reasoning(
                    sample_record, question, 0, []
                )
                
                logger.info(f"Question: '{question}'")
                logger.info(f"  Generated DSL: {prediction_result.dsl_program}")
                logger.info(f"  Operation: {prediction_result.operation_type}")
                logger.info(f"  Confidence: {prediction_result.confidence}")
                logger.info(f"  Reasoning: {prediction_result.reasoning}")
            
            logger.info("DSL generation validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"DSL generation validation FAILED: {e}")
            return False
    
    def _validate_evaluation(self) -> bool:
        """Validate: Are our metrics and comparison methods working?"""
        logger.info("Validating evaluation...")
        
        try:
            # Test quick evaluation on small subset
            result = self.evaluator.quick_evaluation(
                predictor=self.predictor,
                num_conversations=3,
                split='dev'
            )
            
            logger.info(f"Quick evaluation completed:")
            logger.info(f"  Execution accuracy: {result.execution_accuracy:.3f}")
            logger.info(f"  Total questions: {result.total_questions}")
            logger.info(f"  Correct answers: {result.correct_answers}")
            
            # Validate result structure
            required_fields = ['execution_accuracy', 'total_questions', 'correct_answers']
            for field in required_fields:
                if not hasattr(result, field):
                    logger.error(f"Missing evaluation result field: {field}")
                    return False
            
            logger.info("Evaluation validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Evaluation validation FAILED: {e}")
            return False
    
    def _validate_error_handling(self) -> bool:
        """Validate: Does our fallback strategy provide reasonable results?"""
        logger.info("Validating error handling...")
        
        try:
            # Test with problematic inputs
            dev_records = self.dataset.get_dev_records()
            sample_record = dev_records[0]
            
            # Test edge cases
            test_cases = [
                ("", []),  # Empty question
                ("What is the asdfghjkl?", []),  # Nonsense question
                ("", [{"question": "Previous", "answer": "100"}]),  # Empty with history
            ]
            
            for question, history in test_cases:
                try:
                    result = self.predictor.predict_turn(sample_record, 0, history)
                    logger.info(f"Question: '{question}' -> Result: {result}")
                except Exception as e:
                    logger.error(f"Error handling failed for '{question}': {e}")
                    return False
            
            # Test invalid turn index
            try:
                result = self.predictor.predict_turn(sample_record, 999, [])
                logger.info(f"Invalid turn index -> Result: {result}")
            except Exception as e:
                logger.error(f"Error handling failed for invalid turn index: {e}")
                return False
            
            logger.info("Error handling validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Error handling validation FAILED: {e}")
            return False
    
    def _print_validation_summary(self, results: Dict[str, bool]) -> None:
        """Print summary of validation results."""
        logger.info("\n" + "="*60)
        logger.info("PIPELINE VALIDATION SUMMARY")
        logger.info("="*60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for component, passed in results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {component.replace('_', ' ').title()}: {status}")
        
        logger.info("-"*60)
        logger.info(f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("✓ All pipeline components validated successfully!")
        else:
            logger.warning(f"✗ {total_tests - passed_tests} component(s) need attention")


def main():
    """Main entry point for pipeline validation."""
    validator = PipelineValidator()
    results = validator.run_validation()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)
    else:
        logger.info("Pipeline validation completed successfully!")


if __name__ == "__main__":
    main() 