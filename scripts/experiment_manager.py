#!/usr/bin/env python3
"""Experiment management utility for ConvFinQA multi-agent experiments."""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.enhanced_tracker import get_enhanced_tracker


def list_experiments(limit: int = 20) -> None:
    """List recent experiments with basic information."""
    tracker = get_enhanced_tracker()
    experiments = tracker.get_experiment_history(limit=limit)
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"\n📊 Recent Experiments (showing {len(experiments)})")
    print("=" * 100)
    print(f"{'ID':<20} {'Timestamp':<20} {'Accuracy':<10} {'Questions':<10} {'Cost':<10}")
    print("-" * 100)
    
    for exp in experiments:
        exp_id = exp.get('experiment_id', 'N/A')[:18]
        timestamp = exp.get('timestamp', 'N/A')[:19] if exp.get('timestamp') else 'N/A'
        accuracy = f"{exp.get('accuracy_rate', 0):.1%}" if exp.get('accuracy_rate') is not None else 'N/A'
        questions = str(exp.get('total_questions', 'N/A'))
        cost = f"${exp.get('total_estimated_cost', 0):.3f}" if exp.get('total_estimated_cost') is not None else 'N/A'
        
        print(f"{exp_id:<20} {timestamp:<20} {accuracy:<10} {questions:<10} {cost:<10}")
    
    print("-" * 100)


def database_stats() -> None:
    """Show database statistics."""
    tracker = get_enhanced_tracker()
    
    print("\n🗃️  Database Statistics")
    print("=" * 50)
    
    if tracker.db_path.exists():
        db_size = tracker.db_path.stat().st_size / 1024 / 1024  # MB
        print(f"Database Size: {db_size:.2f} MB")
        print(f"Database Path: {tracker.db_path}")
    else:
        print("Database: Not found")
        return


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="ConvFinQA Experiment Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List recent experiments')
    list_parser.add_argument('-n', '--limit', type=int, default=20, 
                           help='Number of experiments to show (default: 20)')
    
    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_experiments(args.limit)
        elif args.command == 'stats':
            database_stats()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
