"""Script to list actual record IDs from the dataset."""

from src.data.dataset import ConvFinQADataset

def main():
    # Load dataset
    dataset = ConvFinQADataset()
    dataset.load()
    
    # Get all record IDs
    all_ids = dataset.get_record_ids()
    
    # Print some example IDs
    print("\nFirst 10 record IDs from the dataset:")
    for idx, record_id in enumerate(all_ids[:10], 1):
        print(f"{idx}. {record_id}")
    
    # Print specific company records if they exist
    companies = ['VRTX', 'JKHY', 'MSFT', 'AAPL']
    print("\nSearching for specific company records:")
    for company in companies:
        matching = [rid for rid in all_ids if company in rid]
        if matching:
            print(f"\n{company} records (first 3):")
            for rid in matching[:3]:
                print(f"  {rid}")

if __name__ == "__main__":
    main() 