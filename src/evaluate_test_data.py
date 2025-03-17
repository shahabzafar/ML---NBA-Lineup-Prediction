import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data_preprocessing import DataPreprocessor
from src.model import LineupPredictor as BaseModel
from src.position_features import PositionFeatureGenerator
from src.chemistry_analysis import ChemistryAnalyzer
from src.time_analysis import TimeAnalyzer
from src.prediction_interface import LineupPredictor as PredictorInterface

def load_and_prepare_model():
    """Load and prepare the model with existing data"""
    print("Loading and preparing model...")
    
    # Initialize components
    data_preprocessor = DataPreprocessor()
    position_generator = PositionFeatureGenerator()
    chemistry_analyzer = ChemistryAnalyzer()
    time_analyzer = TimeAnalyzer()
    base_model = BaseModel()
    
    # Load training data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Load each CSV file with full path
    dfs = []
    for year in range(2007, 2016):  # 2007 to 2015 inclusive
        file_path = os.path.join(data_dir, f'matchups-{year}.csv')
        if os.path.exists(file_path):
            print(f"Loading {year} data...")
            df = pd.read_csv(file_path)
            df['season'] = year  # Add season column
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No matchup files found in {data_dir}")
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    
    # Preprocess and train
    print("\nPreprocessing and training model...")
    df = data_preprocessor.encode_categorical(df)
    base_model.set_encoders(data_preprocessor.team_encoder, data_preprocessor.player_encoder)
    df = position_generator.create_position_features(df)
    df = chemistry_analyzer.calculate_pair_chemistry(df)
    time_analyzer.analyze_time_patterns(df)
    X, y = data_preprocessor.prepare_data(df)
    base_model.train(X, y)
    
    predictor = PredictorInterface(base_model, position_generator, chemistry_analyzer, time_analyzer, data=df)
    
    return predictor, df

def load_test_data():
    """Load test data and labels"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    test_path = os.path.join(project_dir, 'data', 'NBA_test.csv')
    labels_path = os.path.join(project_dir, 'data', 'NBA_test_labels.csv')
    
    test_df = pd.read_csv(test_path)
    labels_df = pd.read_csv(labels_path)
    
    print(f"Loaded {len(test_df)} test samples")
    print(f"Loaded {len(labels_df)} label samples")
    
    # Print column names for debugging
    print("Test data columns:", test_df.columns.tolist())
    print("Labels data columns:", labels_df.columns.tolist())
    
    return test_df, labels_df

def evaluate_model(predictor, test_df, labels_df):
    """Evaluate model on test data"""
    print("Evaluating model on test data...")
    results = []
    
    # The actual label column is "removed_value" not "removed_player"
    label_column = "removed_value"
    print(f"Using '{label_column}' as the player label column")
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Extract data from test row
        season = str(row['season'])
        home_team = row['home_team']
        away_team = row['away_team']
        
        # We need to determine which player was removed from the home team
        # The test file actually has home_0 through home_4, so all positions are filled
        # We need to compare with the label to see which one was artificially added
        
        true_label = labels_df.loc[i, label_column] if i < len(labels_df) else None
        
        # Get game time
        game_time = row.get('starting_min', 15.0)
        
        # In the test data, all 5 positions are filled, but one is artificially added
        # We need to select 4 players excluding the one that was removed (and now in labels)
        home_players = []
        for j in range(5):  # Check all 5 home players
            col_name = f'home_{j}'
            player = row[col_name]
            # Skip the player that was removed (now in labels)
            if player != true_label:
                home_players.append(player)
                if len(home_players) >= 4:  # Only take 4 players
                    break
        
        # Get away players
        away_players = []
        for j in range(5):
            col_name = f'away_{j}'
            if col_name in row and pd.notna(row[col_name]):
                away_players.append(row[col_name])
        
        # Skip if we don't have enough players
        if len(home_players) < 4 or len(away_players) < 5:
            print(f"Warning: Not enough players in row {i}, home:{len(home_players)}, away:{len(away_players)}")
            continue
        
        # Make prediction
        predictions = predictor.predict_fifth_player(
            season=season,
            home_team=home_team,
            away_team=away_team,
            current_players=home_players,
            away_players=away_players,
            game_time=game_time
        )
        
        # Record result
        if predictions and true_label:
            predicted_player = list(predictions.keys())[0]
            result = {
                'id': i,
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'true_player': true_label,
                'predicted_player': predicted_player,
                'confidence': list(predictions.values())[0],
                'is_correct': predicted_player == true_label
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)
    return results_df

def analyze_results(results_df):
    """Analyze evaluation results"""
    if len(results_df) == 0:
        print("No valid results to analyze.")
        return
    
    # Calculate overall accuracy
    accuracy = results_df['is_correct'].mean() * 100
    print(f"Overall accuracy: {accuracy:.2f}%")
    
    # Analyze results by season
    season_accuracy = results_df.groupby('season')['is_correct'].mean() * 100
    print("\nAccuracy by season:")
    for season, acc in season_accuracy.items():
        print(f"  {season}: {acc:.2f}%")
    
    # Count matches per year
    matches_per_year = results_df['season'].value_counts().sort_index()
    avg_matches = matches_per_year.mean()
    
    print("\nMatches per year in test dataset:")
    for year, count in matches_per_year.items():
        print(f"  {year}: {count}")
    
    print(f"\nAverage number of matches across the dataset: {avg_matches:.2f}")
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    plt.bar(matches_per_year.index.astype(str), matches_per_year.values)
    plt.title('Number of Matches per Year in Test Dataset')
    plt.xlabel('Year')
    plt.ylabel('Number of Matches')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, 'evaluation')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, 'matches_per_year.png'))
    print(f"\nVisualization saved to {output_dir}/matches_per_year.png")
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    print(f"Results saved to {output_dir}/evaluation_results.csv")
    
    # Generate required metrics for the slide
    metrics = {
        'matches_per_year': matches_per_year.to_dict(),
        'average_matches': avg_matches,
        'overall_accuracy': accuracy
    }
    
    return metrics

def main():
    """Main evaluation function"""
    # Load and prepare model
    predictor, training_df = load_and_prepare_model()
    
    # Load test data
    test_df, labels_df = load_test_data()
    
    # Evaluate model
    results_df = evaluate_model(predictor, test_df, labels_df)
    
    # Analyze results
    metrics = analyze_results(results_df)
    
    return metrics

if __name__ == "__main__":
    main() 