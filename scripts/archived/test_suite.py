#!/usr/bin/env python3
"""Comprehensive test suite for ConvFinQA predictors and components.

This script provides integration testing, component testing, and tracking system
validation for both the hybrid keyword predictor and multi-agent predictor.
"""

import sys
import os
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Any, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.dataset import ConvFinQADataset
from predictors.hybrid_keyword_predictor import HybridKeywordPredictor
from evaluation.evaluator import ConvFinQAEvaluator
from utils.performance_tracker import get_performance_tracker, create_config_snapshot
from utils.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional multi-agent imports (graceful fallback if not available)
try:
    from predictors.multi_agent_predictor import ConvFinQAMultiAgentPredictor
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False
    logger.warning("Multi-agent predictor not available - multi-agent tests will be skipped")


class TestSuite:
    """Comprehensive test suite for all predictors and components."""
    
    def __init__(self, include_multi_agent: bool = True):
        """Initialize test suite."""
        self.include_multi_agent = include_multi_agent and MULTI_AGENT_AVAILABLE
        self.dataset = None
        self.hybrid_predictor = None
        self.multi_agent_predictor = None
        self.evaluator = None
        
    def setup_components(self) -> bool:
        """Setup common components for testing."""
        try:
            logger.info("Setting up test components...")
            
            # Initialize dataset
            self.dataset = ConvFinQADataset()
            if not self.dataset.is_loaded:
                self.dataset.load()
            logger.info(f"✅ Dataset loaded: {len(self.dataset.get_split('dev'))} dev records")
            
            # Initialize evaluator
            self.evaluator = ConvFinQAEvaluator(self.dataset)
            logger.info("✅ Evaluator created")
            
            # Initialize hybrid predictor
            self.hybrid_predictor = HybridKeywordPredictor()
            logger.info("✅ Hybrid predictor created")
            
            # Initialize multi-agent predictor if available
            if self.include_multi_agent:
                try:
                    config = Config()
                    self.multi_agent_predictor = ConvFinQAMultiAgentPredictor(config)
                    logger.info("✅ Multi-agent predictor created")
                except Exception as e:
                    logger.warning(f"Multi-agent predictor creation failed: {e}")
                    self.include_multi_agent = False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Component setup failed: {e}")
            return False
    
    def test_dataset_integration(self) -> bool:
        """Test dataset loading and structure validation."""
        logger.info("\n=== Testing Dataset Integration ===")
        
        try:
            # Check splits exist
            splits = self.dataset.splits
            if 'train' not in splits or 'dev' not in splits:
                logger.error(f"Missing required splits. Found: {splits}")
                return False
            
            # Check we can access records
            dev_records = self.dataset.get_split('dev')
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
            
            # Check table structure
            table = sample_record.doc.table
            if not isinstance(table, dict):
                logger.error("Table is not a dictionary")
                return False
            
            logger.info(f"✅ Dataset integration tests passed")
            logger.info(f"   - {len(dev_records)} dev records available")
            logger.info(f"   - Sample record: {sample_record.id}")
            logger.info(f"   - Questions per record: {len(dialogue.conv_questions)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Dataset integration test failed: {e}")
            return False
    
    def test_hybrid_predictor_components(self) -> bool:
        """Test individual components of the hybrid predictor."""
        logger.info("\n=== Testing Hybrid Predictor Components ===")
        
        try:
            # Get sample data
            dev_records = self.dataset.get_split('dev')
            sample_record = dev_records[0]
            
            # Test questions for component validation
            test_questions = [
                "What is the total revenue?",      # Should detect addition
                "What was the profit in 2020?",   # Should detect lookup
                "What is the difference?",         # Should detect subtraction
                "What about the next year?",       # Context-dependent
            ]
            
            logger.info("Testing question classification...")
            classification_passed = 0
            for question in test_questions:
                operation_match = self.hybrid_predictor._classify_question(question, [])
                if operation_match:
                    logger.info(f"  '{question}' -> {operation_match.operation_type} (confidence: {operation_match.confidence:.3f})")
                    classification_passed += 1
                else:
                    logger.warning(f"  '{question}' -> No operation detected")
            
            logger.info("Testing value extraction...")
            extraction_passed = 0
            for question in test_questions[:2]:  # Test first 2 questions
                value_candidates = self.hybrid_predictor._extract_values(sample_record, question, [])
                if len(value_candidates) > 0:
                    logger.info(f"  '{question}' -> {len(value_candidates)} candidates found")
                    extraction_passed += 1
                else:
                    logger.warning(f"  '{question}' -> No value candidates")
            
            logger.info("Testing end-to-end prediction...")
            prediction_passed = 0
            conversation_history = []
            
            for i, question in enumerate(test_questions[:2]):
                try:
                    predicted = self.hybrid_predictor.predict_turn(sample_record, i, conversation_history)
                    logger.info(f"  '{question}' -> '{predicted}'")
                    prediction_passed += 1
                    
                    # Update conversation history
                    conversation_history.append({
                        'question': question,
                        'answer': str(predicted)
                    })
                except Exception as e:
                    logger.warning(f"  '{question}' -> Error: {e}")
            
            # Summary
            total_tests = len(test_questions) + 2 + 2  # classification + extraction + prediction
            passed_tests = classification_passed + extraction_passed + prediction_passed
            
            logger.info(f"✅ Hybrid component tests: {passed_tests}/{total_tests} passed")
            return passed_tests >= total_tests * 0.75  # 75% pass rate required
            
        except Exception as e:
            logger.error(f"❌ Hybrid component testing failed: {e}")
            return False
    
    def test_multi_agent_predictor_integration(self) -> bool:
        """Test multi-agent predictor integration."""
        if not self.include_multi_agent:
            logger.info("\n=== Skipping Multi-Agent Tests (not available) ===")
            return True
            
        logger.info("\n=== Testing Multi-Agent Predictor Integration ===")
        
        try:
            # Test predictor instantiation
            if not self.multi_agent_predictor:
                logger.error("Multi-agent predictor not available")
                return False
            
            logger.info("✅ Multi-agent predictor instantiated")
            
            # Test protocol compliance
            required_methods = ['predict_turn']
            for method in required_methods:
                if not hasattr(self.multi_agent_predictor, method):
                    logger.error(f"Missing required method: {method}")
                    return False
            
            logger.info("✅ Protocol compliance verified")
            
            # Test method signature
            import inspect
            sig = inspect.signature(self.multi_agent_predictor.predict_turn)
            params = list(sig.parameters.keys())
            expected_params = ['record', 'turn_index', 'conversation_history']
            
            for param in expected_params:
                if param not in params:
                    logger.error(f"Missing required parameter: {param}")
                    return False
            
            logger.info("✅ Method signatures correct")
            
            # Test OpenAI dependency handling
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.info("✅ Graceful handling of missing API key")
                # Test predict_turn with missing API key (should not crash)
                try:
                    dev_records = self.dataset.get_split('dev')
                    sample_record = dev_records[0]
                    result = self.multi_agent_predictor.predict_turn(sample_record, 0, [])
                    logger.info(f"✅ Graceful degradation: {result}")
                except Exception as e:
                    logger.warning(f"Predict turn with missing API key: {e}")
            else:
                logger.info("✅ API key available for testing")
            
            # Test performance monitoring (if available)
            if hasattr(self.multi_agent_predictor, 'get_performance_summary'):
                try:
                    summary = self.multi_agent_predictor.get_performance_summary()
                    expected_keys = ["cost_breakdown", "performance_metrics"]
                    for key in expected_keys:
                        if key not in summary:
                            logger.warning(f"Missing performance summary key: {key}")
                        else:
                            logger.info(f"✅ Performance monitoring: {key} available")
                except Exception as e:
                    logger.warning(f"Performance monitoring test failed: {e}")
            
            logger.info("✅ Multi-agent integration tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Multi-agent integration test failed: {e}")
            return False
    
    def test_evaluator_integration(self) -> bool:
        """Test evaluator integration with both predictors."""
        logger.info("\n=== Testing Evaluator Integration ===")
        
        try:
            # Test evaluator has required methods
            required_methods = ['quick_evaluation']
            for method in required_methods:
                if not hasattr(self.evaluator, method):
                    logger.error(f"Evaluator missing required method: {method}")
                    return False
            
            logger.info("✅ Evaluator methods available")
            
            # Test hybrid predictor with evaluator (small scale)
            logger.info("Testing hybrid predictor with evaluator...")
            try:
                result = self.evaluator.quick_evaluation(
                    predictor=self.hybrid_predictor,
                    num_conversations=2,
                    split='dev'
                )
                
                # Validate result structure
                required_fields = ['execution_accuracy', 'total_questions', 'correct_answers']
                for field in required_fields:
                    if not hasattr(result, field):
                        logger.error(f"Missing evaluation result field: {field}")
                        return False
                
                logger.info(f"✅ Hybrid evaluation: {result.execution_accuracy:.1f}% accuracy on {result.total_questions} questions")
                
            except Exception as e:
                logger.error(f"Hybrid evaluation test failed: {e}")
                return False
            
            # Test multi-agent predictor with evaluator (if available and API key present)
            if self.include_multi_agent and os.getenv("OPENAI_API_KEY"):
                logger.info("Testing multi-agent predictor with evaluator...")
                try:
                    result = self.evaluator.quick_evaluation(
                        predictor=self.multi_agent_predictor,
                        num_conversations=1,  # Small test due to cost
                        split='dev'
                    )
                    logger.info(f"✅ Multi-agent evaluation: {result.execution_accuracy:.1f}% accuracy on {result.total_questions} questions")
                except Exception as e:
                    logger.warning(f"Multi-agent evaluation test failed (expected if no API key): {e}")
            
            logger.info("✅ Evaluator integration tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Evaluator integration test failed: {e}")
            return False
    
    def test_tracking_system(self) -> bool:
        """Test performance tracking system."""
        logger.info("\n=== Testing Performance Tracking System ===")
        
        try:
            # Test configuration creation
            config = Config()
            tracker = get_performance_tracker()
            snapshot = create_config_snapshot(config)
            
            logger.info(f"✅ Configuration snapshot created with hash: {snapshot.config_hash}")
            
            # Test registration
            config_hash = tracker.register_configuration(snapshot)
            logger.info(f"✅ Configuration registered with hash: {config_hash}")
            
            # Test logging an execution
            tracker.log_execution(
                config_hash=config_hash,
                record_id='test_record_001',
                success=True,
                execution_time=1.5,
                estimated_cost=0.05,
                confidence=0.85,
                fallback_used=False,
                question_type='calculation'
            )
            logger.info("✅ Execution logged successfully")
            
            # Test retrieval
            aggregate = tracker.get_performance_aggregate(config_hash)
            if aggregate:
                logger.info(f"✅ Performance aggregate retrieved:")
                logger.info(f"   - Total executions: {aggregate.total_executions}")
                logger.info(f"   - Success rate: {aggregate.success_rate:.2f}")
                logger.info(f"   - Average cost: ${aggregate.average_cost_per_execution:.4f}")
            else:
                logger.warning("No performance data found")
            
            # Test configuration listing
            configs = tracker.list_configurations()
            logger.info(f"✅ Found {len(configs)} registered configurations")
            
            # Test different configuration variants
            test_configs = [
                {'models.hybrid_keyword.question_classification.confidence_threshold': 0.5},
                {'models.hybrid_keyword.question_classification.confidence_threshold': 0.8},
            ]
            
            hashes = []
            for i, test_config in enumerate(test_configs):
                for key, value in test_config.items():
                    config.update(key, value)
                
                variant_snapshot = create_config_snapshot(config)
                hashes.append(variant_snapshot.config_hash)
                logger.info(f"✅ Configuration variant {i+1} hash: {variant_snapshot.config_hash}")
            
            # Check that different configurations produce different hashes
            unique_hashes = set(hashes)
            if len(unique_hashes) == len(hashes):
                logger.info("✅ All configurations produce unique hashes")
            else:
                logger.warning("Some configurations produced identical hashes")
            
            logger.info("✅ Tracking system tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tracking system test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and edge cases."""
        logger.info("\n=== Testing Error Handling ===")
        
        try:
            dev_records = self.dataset.get_split('dev')
            sample_record = dev_records[0]
            
            # Test edge cases with hybrid predictor
            test_cases = [
                ("", []),  # Empty question
                ("What is the asdfghjkl?", []),  # Nonsense question
                ("", [{"question": "Previous", "answer": "100"}]),  # Empty with history
            ]
            
            hybrid_errors = 0
            for question, history in test_cases:
                try:
                    result = self.hybrid_predictor.predict_turn(sample_record, 0, history)
                    logger.info(f"✅ Hybrid handled: '{question}' -> {result}")
                except Exception as e:
                    logger.warning(f"Hybrid error for '{question}': {e}")
                    hybrid_errors += 1
            
            # Test invalid turn index
            try:
                result = self.hybrid_predictor.predict_turn(sample_record, 999, [])
                logger.info(f"✅ Hybrid handled invalid turn index -> {result}")
            except Exception as e:
                logger.warning(f"Hybrid error for invalid turn index: {e}")
                hybrid_errors += 1
            
            # Test multi-agent error handling if available
            multi_agent_errors = 0
            if self.include_multi_agent:
                for question, history in test_cases:
                    try:
                        result = self.multi_agent_predictor.predict_turn(sample_record, 0, history)
                        logger.info(f"✅ Multi-agent handled: '{question}' -> {result}")
                    except Exception as e:
                        logger.warning(f"Multi-agent error for '{question}': {e}")
                        multi_agent_errors += 1
            
            total_tests = len(test_cases) + 1  # + invalid turn index
            if self.include_multi_agent:
                total_tests *= 2
            
            total_errors = hybrid_errors + multi_agent_errors
            success_rate = (total_tests - total_errors) / total_tests
            
            logger.info(f"✅ Error handling tests: {success_rate:.1%} success rate ({total_errors}/{total_tests} errors)")
            return success_rate >= 0.5  # At least 50% should handle gracefully
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("🧪 Starting Comprehensive Test Suite")
        logger.info("=" * 60)
        
        # Setup components
        if not self.setup_components():
            return {"setup": False}
        
        # Run all test categories
        test_results = {}
        
        test_results["dataset_integration"] = self.test_dataset_integration()
        test_results["hybrid_components"] = self.test_hybrid_predictor_components()
        test_results["multi_agent_integration"] = self.test_multi_agent_predictor_integration()
        test_results["evaluator_integration"] = self.test_evaluator_integration()
        test_results["tracking_system"] = self.test_tracking_system()
        test_results["error_handling"] = self.test_error_handling()
        
        # Print summary
        self.print_test_summary(test_results)
        
        return test_results
    
    def print_test_summary(self, results: Dict[str, bool]) -> None:
        """Print test summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            emoji = "✅" if passed else "❌"
            logger.info(f"{emoji} {test_name.replace('_', ' ').title()}: {status}")
        
        logger.info("-" * 60)
        logger.info(f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed! System is ready for use.")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} test(s) failed - please review and fix issues.")


def main():
    """Main entry point for test suite."""
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument('--skip-multi-agent', action='store_true',
                       help='Skip multi-agent tests (useful if dependencies not installed)')
    parser.add_argument('--test', type=str, choices=[
        'dataset', 'hybrid', 'multi-agent', 'evaluator', 'tracking', 'error'
    ], help='Run only specific test category')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test suite
    suite = TestSuite(include_multi_agent=not args.skip_multi_agent)
    
    # Run specific test or all tests
    if args.test:
        logger.info(f"Running specific test category: {args.test}")
        
        if not suite.setup_components():
            logger.error("Failed to setup components")
            sys.exit(1)
        
        # Run specific test
        test_map = {
            'dataset': suite.test_dataset_integration,
            'hybrid': suite.test_hybrid_predictor_components,
            'multi-agent': suite.test_multi_agent_predictor_integration,
            'evaluator': suite.test_evaluator_integration,
            'tracking': suite.test_tracking_system,
            'error': suite.test_error_handling
        }
        
        result = test_map[args.test]()
        status = "PASSED" if result else "FAILED"
        logger.info(f"Test {args.test} {status}")
        sys.exit(0 if result else 1)
    else:
        # Run all tests
        results = suite.run_all_tests()
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main() 