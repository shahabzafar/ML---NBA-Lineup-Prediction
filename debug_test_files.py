import pandas as pd
import os

def inspect_test_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_path = os.path.join(current_dir, 'data', 'NBA_test.csv')
    labels_path = os.path.join(current_dir, 'data', 'NBA_test_labels.csv')
    
    # Check if files exist
    print(f"Test file exists: {os.path.exists(test_path)}")
    print(f"Labels file exists: {os.path.exists(labels_path)}")
    
    # Load and inspect files
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print("\nTest file info:")
        print(f"Shape: {test_df.shape}")
        print(f"Columns: {test_df.columns.tolist()}")
        print("\nFirst few rows:")
        print(test_df.head(2))
    
    if os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        print("\nLabels file info:")
        print(f"Shape: {labels_df.shape}")
        print(f"Columns: {labels_df.columns.tolist()}")
        print("\nFirst few rows:")
        print(labels_df.head(2))
    
    # Count matches per year in test data
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        if 'season' in test_df.columns:
            matches_per_year = test_df['season'].value_counts().sort_index()
            print("\nMatches per year in test data:")
            for year, count in matches_per_year.items():
                print(f"Year {year}: {count} matches")
            print(f"Average matches per year: {matches_per_year.mean():.2f}")

if __name__ == "__main__":
    inspect_test_files() 