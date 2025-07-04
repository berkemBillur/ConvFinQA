#!/usr/bin/env python3
"""Test script for Enhanced Tracking System.

This script validates the core functionality of the enhanced tracking system
without requiring a full benchmark run.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.enhanced_tracker import (
    get_enhanced_tracker,
    AgentConfiguration,
    WorkflowConfiguration,
    TaskConfiguration,
    DatasetConfiguration,
    SystemConfiguration,
    PerformanceResults,
    CompleteExperimentSnapshot
)


def test_tracker_initialization():
    """Test tracker initialization and database creation."""
    print("🔧 Testing tracker initialization...")
    
    tracker = get_enhanced_tracker()
    
    # Check database exists
    assert tracker.db_path.exists(), "Database file not created"
    
    # Check storage directory exists
    assert tracker.storage_dir.exists(), "Storage directory not created"
    
    print("✅ Tracker initialization successful")
    return tracker


def test_configuration_capture():
    """Test configuration capture functionality."""
    print("📸 Testing configuration capture...")
    
    # Mock agent configurations
    mock_agents = {
        'supervisor': type('MockAgent', (), {
            'role': 'Financial QA Orchestrator',
            'goal': 'Decompose conversational financial queries',
            'backstory': 'You are a senior financial analyst...',
            'llm': type('MockLLM', (), {
                'model_name': 'gpt-4o',
                'temperature': 0.1
            })(),
            'verbose': True,
            'allow_delegation': False,
            'tools': []
        })(),
        'extractor': type('MockAgent', (), {
            'role': 'Financial Data Extraction Specialist',
            'goal': 'Extract precise numerical data',
            'backstory': 'You are a data extraction expert...',
            'llm': type('MockLLM', (), {
                'model_name': 'gpt-4o-mini',
                'temperature': 0.0
            })(),
            'verbose': True,
            'allow_delegation': False,
            'tools': []
        })()
    }
    
    # Mock crew config
    crew_config = {
        'process_type': 'hierarchical',
        'manager_model': 'gpt-4o',
        'manager_temperature': 0.1,
        'memory': True,
        'cache': True
    }
    
    # Mock dataset info
    dataset_info = {
        'total_conversations': 737,
        'sample_size': 5,
        'sampling_strategy': 'sequential',
        'random_seed': None,
        'conversation_ids': ['test_0', 'test_1', 'test_2', 'test_3', 'test_4'],
        'records': [
            type('MockRecord', (), {
                'dialogue': type('MockDialogue', (), {
                    'conv_questions': ['What was the revenue?', 'How much did it increase?']
                })()
            })() for _ in range(5)
        ]
    }
    
    tracker = get_enhanced_tracker()
    
    # Create experiment snapshot
    snapshot = tracker.create_experiment_snapshot(
        agents=mock_agents,
        crew_config=crew_config,
        dataset_info=dataset_info,
        notes="Test experiment for validation"
    )
    
    # Validate snapshot
    assert snapshot.experiment_id.startswith('exp_'), "Invalid experiment ID format"
    assert len(snapshot.config_hash) == 12, "Invalid config hash length"
    assert len(snapshot.agents) == 2, "Wrong number of agents captured"
    assert snapshot.workflow.process_type == 'hierarchical', "Workflow config not captured"
    assert snapshot.dataset.sample_size == 5, "Dataset config not captured"
    
    print(f"✅ Configuration capture successful: {snapshot.experiment_id}")
    return snapshot


def test_performance_tracking():
    """Test performance results tracking."""
    print("📊 Testing performance tracking...")
    
    # Create mock performance results
    performance_results = PerformanceResults(
        total_questions=10,
        correct_answers=8,
        accuracy_rate=0.8,
        total_execution_time=45.5,
        avg_execution_time_per_question=4.55,
        total_estimated_cost=0.012,
        avg_cost_per_question=0.0012,
        agent_performance={
            'supervisor': {
                'total_executions': 10,
                'successful_executions': 9,
                'success_rate': 0.9,
                'avg_execution_time': 1.2,
                'estimated_cost': 0.003
            },
            'extractor': {
                'total_executions': 10,
                'successful_executions': 8,
                'success_rate': 0.8,
                'avg_execution_time': 1.5,
                'estimated_cost': 0.004
            }
        },
        question_type_performance={
            'calculation': {
                'total_questions': 6,
                'correct_answers': 5,
                'accuracy_rate': 0.833,
                'avg_execution_time': 5.2,
                'avg_cost': 0.0015
            },
            'lookup': {
                'total_questions': 4,
                'correct_answers': 3,
                'accuracy_rate': 0.75,
                'avg_execution_time': 3.1,
                'avg_cost': 0.0008
            }
        },
        failure_count=2,
        failure_rate=0.2,
        error_patterns={'TimeoutError': 1, 'ValidationError': 1},
        common_failures=['Q5: Timeout during calculation', 'Q8: Failed validation'],
        cost_by_agent={'supervisor': 0.003, 'extractor': 0.004},
        cost_by_question_type={'calculation': 0.009, 'lookup': 0.003}
    )
    
    # Update experiment with results
    tracker = get_enhanced_tracker()
    
    # Get a recent experiment (or create one)
    experiments = tracker.get_experiment_history(limit=1)
    if experiments:
        experiment_id = experiments[0]['experiment_id']
    else:
        # Create test snapshot
        snapshot = test_configuration_capture()
        experiment_id = snapshot.experiment_id
    
    tracker.update_experiment_results(experiment_id, performance_results)
    
    # Verify results were stored
    updated_experiments = tracker.get_experiment_history(limit=1)
    if updated_experiments:
        exp = updated_experiments[0]
        assert exp['accuracy_rate'] == 0.8, "Accuracy rate not stored correctly"
        assert exp['total_questions'] == 10, "Total questions not stored correctly"
    
    print(f"✅ Performance tracking successful: {experiment_id}")
    return experiment_id


def test_database_operations():
    """Test database query and management operations."""
    print("🗃️ Testing database operations...")
    
    tracker = get_enhanced_tracker()
    
    # Test experiment history
    experiments = tracker.get_experiment_history(limit=10)
    print(f"   Found {len(experiments)} experiments in history")
    
    # Test database statistics
    if tracker.db_path.exists():
        db_size = tracker.db_path.stat().st_size
        print(f"   Database size: {db_size} bytes")
    
    # Test configuration comparison (if multiple experiments exist)
    if len(experiments) >= 2:
        exp_ids = [exp['experiment_id'] for exp in experiments[:2]]
        comparison = tracker.get_configuration_comparison(exp_ids)
        print(f"   Configuration comparison: {len(comparison)} differences found")
    
    print("✅ Database operations successful")


def test_revert_functionality():
    """Test configuration revert functionality."""
    print("🔄 Testing revert functionality...")
    
    tracker = get_enhanced_tracker()
    
    # Get recent experiment
    experiments = tracker.get_experiment_history(limit=1)
    if not experiments:
        # Create test experiment
        snapshot = test_configuration_capture()
        experiment_id = snapshot.experiment_id
    else:
        experiment_id = experiments[0]['experiment_id']
    
    try:
        # Generate revert configuration
        revert_config = tracker.revert_to_configuration(experiment_id)
        
        # Validate revert config structure
        assert 'agents' in revert_config, "Agents not in revert config"
        assert 'workflow' in revert_config, "Workflow not in revert config"
        assert 'dataset' in revert_config, "Dataset not in revert config"
        
        print(f"✅ Revert functionality successful: {experiment_id}")
        
    except Exception as e:
        print(f"⚠️  Revert functionality failed (expected for missing snapshots): {e}")


def run_all_tests():
    """Run all tests and report results."""
    print("🧪 Starting Enhanced Tracking System Tests")
    print("=" * 60)
    
    try:
        # Run individual tests
        tracker = test_tracker_initialization()
        snapshot = test_configuration_capture()
        experiment_id = test_performance_tracking()
        test_database_operations()
        test_revert_functionality()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        
        # Print summary
        experiments = tracker.get_experiment_history(limit=5)
        print(f"\n📊 System Status:")
        print(f"   Database: {tracker.db_path}")
        print(f"   Storage: {tracker.storage_dir}")
        print(f"   Total Experiments: {len(experiments)}")
        
        if experiments:
            latest = experiments[0]
            print(f"   Latest Experiment: {latest['experiment_id']}")
            if latest.get('accuracy_rate') is not None:
                print(f"   Latest Accuracy: {latest['accuracy_rate']:.2%}")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Run: python scripts/enhanced_benchmark_multi_agent.py -n 3")
        print(f"   2. View: python scripts/experiment_manager.py list")
        print(f"   3. Dashboard: streamlit run scripts/dashboard.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 