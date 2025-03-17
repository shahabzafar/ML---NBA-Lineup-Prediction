import pandas as pd
import os

# Path to the detailed test results - fixed path
current_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(current_dir, 'detailed_test_results.csv')  # Removed extra 'evaluation' folder reference

# Check if the file exists
if not os.path.exists(results_path):
    print(f"Error: Could not find the file at {results_path}")
else:
    # Load the results
    results_df = pd.read_csv(results_path)
    
    # Count true and false predictions
    true_count = results_df['is_correct'].sum()
    false_count = len(results_df) - true_count
    
    # Calculate accuracy
    accuracy = (true_count / len(results_df)) * 100 if len(results_df) > 0 else 0
    
    # Print summary statistics
    print("\n===== TEST RESULTS SUMMARY =====")
    print(f"Total test cases: {len(results_df)}")
    print(f"Correct predictions: {true_count} ({accuracy:.2f}%)")
    print(f"Incorrect predictions: {false_count} ({100-accuracy:.2f}%)")
    
    # Results by season
    print("\n===== ACCURACY BY SEASON =====")
    season_results = results_df.groupby('season').agg({
        'is_correct': ['count', 'sum', 'mean']
    })
    season_results.columns = ['Total', 'Correct', 'Accuracy']
    season_results['Accuracy'] = season_results['Accuracy'] * 100
    season_results['Incorrect'] = season_results['Total'] - season_results['Correct']
    
    # Print season stats in a table format
    print(f"{'Season':<10}{'Total':<10}{'Correct':<10}{'Incorrect':<10}{'Accuracy':<10}")
    print("-" * 50)
    for season, row in season_results.iterrows():
        print(f"{season:<10}{int(row['Total']):<10}{int(row['Correct']):<10}{int(row['Incorrect']):<10}{row['Accuracy']:.2f}%")