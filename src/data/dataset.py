"""Dataset loading and management for ConvFinQA."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import ConvFinQARecord


class ConvFinQADataset:
    """Main dataset class for loading and accessing ConvFinQA data."""
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialise the dataset.
        
        Args:
            data_path: Path to the ConvFinQA JSON dataset file. If None, uses config.
        """
        if data_path is None:
            try:
                from ..utils.config import get_config
                config = get_config()
                self.data_path = Path(config.data.dataset_path)
            except ImportError:
                # Fallback for when config system is not available
                self.data_path = Path("data/convfinqa_dataset.json")
        else:
            self.data_path = Path(data_path)
        self._data: Optional[Dict[str, List[dict]]] = None
        self._records: Optional[Dict[str, List[ConvFinQARecord]]] = None
        
    def load(self) -> None:
        """Load the dataset from JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self._data = json.load(f)
            
        if self._data:
            print(f"Loaded {sum(len(split) for split in self._data.values())} conversations")
        
    def _parse_records(self) -> None:
        """Parse raw data into Pydantic models."""
        if self._data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
            
        self._records = {}
        
        for split_name, conversations in self._data.items():
            records = []
            for conv_data in conversations:
                try:
                    record = ConvFinQARecord(**conv_data)
                    records.append(record)
                except Exception as e:
                    print(f"Warning: Failed to parse record {conv_data.get('id', 'unknown')}: {e}")
                    
            self._records[split_name] = records
            print(f"Parsed {len(records)} records for {split_name} split")
    
    @property
    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return self._data is not None
    
    @property
    def splits(self) -> List[str]:
        """Get available data splits."""
        if not self.is_loaded or self._data is None:
            return []
        return list(self._data.keys())
    
    def get_split(self, split_name: str) -> List[ConvFinQARecord]:
        """Get all records for a specific split.
        
        Args:
            split_name: Name of the split ('train' or 'dev').
            
        Returns:
            List of ConvFinQA records for the split.
        """
        if not self.is_loaded:
            self.load()
            
        if self._records is None:
            self._parse_records()
            
        if self._records is None or split_name not in self._records:
            raise ValueError(f"Split '{split_name}' not found. Available: {self.splits}")
            
        return self._records[split_name]
    
    def get_record_by_id(self, record_id: str) -> Optional[ConvFinQARecord]:
        """Get a specific record by its ID.
        
        Args:
            record_id: The ID of the record to retrieve.
            
        Returns:
            The ConvFinQA record if found, None otherwise.
        """
        for split_name in self.splits:
            records = self.get_split(split_name)
            for record in records:
                if record.id == record_id:
                    return record
        return None
    
    def get_train_records(self) -> List[ConvFinQARecord]:
        """Get training records."""
        return self.get_split('train')
    
    def get_dev_records(self) -> List[ConvFinQARecord]:
        """Get development records."""
        return self.get_split('dev')
    
    def get_record_ids(self, split_name: Optional[str] = None) -> List[str]:
        """Get all record IDs for a split or all splits.
        
        Args:
            split_name: Specific split name, or None for all splits.
            
        Returns:
            List of record IDs.
        """
        if split_name:
            return [record.id for record in self.get_split(split_name)]
        
        ids = []
        for split in self.splits:
            ids.extend([record.id for record in self.get_split(split)])
        return ids
    
    def summary(self) -> Dict[str, int]:
        """Get dataset summary statistics.
        
        Returns:
            Dictionary with split names and conversation counts.
        """
        if not self.is_loaded:
            self.load()
            
        if self._data is None:
            return {}
            
        return {split: len(conversations) for split, conversations in self._data.items()} 