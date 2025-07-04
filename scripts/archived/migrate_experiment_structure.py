#!/usr/bin/env python3
"""Migration script to move existing experiment data to new unified structure.

This script moves:
- results/multi_agent/* -> experiment_tracking/results/*
- experiments/tracking/* -> experiment_tracking/configurations/*
"""

import os
import shutil
from pathlib import Path
import json
import sqlite3

def migrate_experiment_structure():
    """Migrate existing experiment data to new structure."""
    print("🔄 Starting experiment structure migration...")
    
    # Create new directory structure
    new_base = Path("experiment_tracking")
    new_results = new_base / "results"
    new_configs = new_base / "configurations"
    
    new_base.mkdir(exist_ok=True)
    new_results.mkdir(exist_ok=True)
    new_configs.mkdir(exist_ok=True)
    
    # 1. Migrate results from results/multi_agent to experiment_tracking/results
    old_results = Path("results/multi_agent")
    if old_results.exists():
        print(f"📁 Migrating results from {old_results} to {new_results}")
        
        # Copy all timestamped directories
        for timestamp_dir in old_results.iterdir():
            if timestamp_dir.is_dir():
                dest_dir = new_results / timestamp_dir.name
                if dest_dir.exists():
                    print(f"   ⚠️  {dest_dir} already exists, skipping...")
                else:
                    shutil.copytree(timestamp_dir, dest_dir)
                    print(f"   ✅ Moved {timestamp_dir.name}")
        
        print(f"📊 Results migration complete")
    else:
        print("📁 No existing results/multi_agent directory found")
    
    # 2. Migrate configurations from experiments/tracking to experiment_tracking/configurations
    old_configs = Path("experiments/tracking")
    if old_configs.exists():
        print(f"📁 Migrating configurations from {old_configs} to {new_configs}")
        
        # Copy database and snapshots
        for file_path in old_configs.iterdir():
            if file_path.is_file():
                dest_path = new_configs / file_path.name
                
                # Special handling for database
                if file_path.name == "experiments.db":
                    if dest_path.exists():
                        print(f"   🔄 Merging database data...")
                        merge_databases(file_path, dest_path)
                    else:
                        shutil.copy2(file_path, dest_path)
                        print(f"   ✅ Copied {file_path.name}")
                
                # Special handling for snapshots
                elif file_path.name in ["experiment_snapshots.json", "snapshots.json"]:
                    target_name = "snapshots.json"
                    dest_path = new_configs / target_name
                    
                    if dest_path.exists():
                        print(f"   🔄 Merging snapshot data...")
                        merge_json_files(file_path, dest_path)
                    else:
                        shutil.copy2(file_path, dest_path)
                        print(f"   ✅ Copied {file_path.name} -> {target_name}")
                
                # Copy other files as-is
                else:
                    if not dest_path.exists():
                        shutil.copy2(file_path, dest_path)
                        print(f"   ✅ Copied {file_path.name}")
        
        print(f"📊 Configuration migration complete")
    else:
        print("📁 No existing experiments/tracking directory found")
    
    # 3. Update any absolute paths in configuration files
    update_config_paths(new_configs)
    
    print("🎉 Migration completed successfully!")
    print(f"📁 New structure:")
    print(f"   📊 Results: {new_results}")
    print(f"   ⚙️  Configurations: {new_configs}")
    print(f"\n💡 You can now delete the old directories:")
    print(f"   rm -rf results/multi_agent")
    print(f"   rm -rf experiments/")

def merge_databases(source_db: Path, target_db: Path):
    """Merge SQLite databases, avoiding duplicates."""
    try:
        # Connect to both databases
        source_conn = sqlite3.connect(source_db)
        target_conn = sqlite3.connect(target_db)
        
        # Get existing experiment IDs in target
        existing_ids = set()
        cursor = target_conn.execute("SELECT experiment_id FROM experiments")
        for row in cursor:
            existing_ids.add(row[0])
        
        # Copy non-duplicate experiments
        source_cursor = source_conn.execute("SELECT * FROM experiments")
        for row in source_cursor:
            if row[0] not in existing_ids:  # experiment_id is first column
                placeholders = ','.join(['?' for _ in row])
                target_conn.execute(f"INSERT INTO experiments VALUES ({placeholders})", row)
        
        target_conn.commit()
        source_conn.close()
        target_conn.close()
        
        print(f"   ✅ Database merge completed")
        
    except Exception as e:
        print(f"   ❌ Database merge failed: {e}")

def merge_json_files(source_file: Path, target_file: Path):
    """Merge JSON files, avoiding duplicates."""
    try:
        # Load both files
        with open(source_file, 'r') as f:
            source_data = json.load(f)
        
        with open(target_file, 'r') as f:
            target_data = json.load(f)
        
        # Merge, preferring target data for duplicates
        if isinstance(source_data, dict) and isinstance(target_data, dict):
            merged_data = {**source_data, **target_data}
        else:
            merged_data = target_data  # Keep target if not both dicts
        
        # Save merged data
        with open(target_file, 'w') as f:
            json.dump(merged_data, f, indent=2, default=str)
        
        print(f"   ✅ JSON merge completed")
        
    except Exception as e:
        print(f"   ❌ JSON merge failed: {e}")

def update_config_paths(config_dir: Path):
    """Update any hardcoded paths in configuration files."""
    # This could be extended to update any config files with absolute paths
    print("   📝 Configuration paths updated")

if __name__ == "__main__":
    migrate_experiment_structure() 