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
    
    # Initialize components required for the prediction system
    data_preprocessor = DataPreprocessor()
    position_generator = PositionFeatureGenerator()
    chemistry_analyzer = ChemistryAnalyzer()
    time_analyzer = TimeAnalyzer()
    base_model = BaseModel()
    
    # Set up paths to load training data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # Load individual season data files and combine them
    dfs = []
    for year in range(2007, 2016):  # Load data from 2007-2015 seasons
        file_path = os.path.join(data_dir, f'matchups-{year}.csv')
        if os.path.exists(file_path):
            print(f"Loading {year} data...")
            df = pd.read_csv(file_path)
            df['season'] = year  # Add season identifier
            dfs.append(df)
    
    # Validate that data was found
    if not dfs:
        raise FileNotFoundError(f"No matchup files found in {data_dir}")
        
    # Combine all seasons into a single dataset
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    
    # Apply full preprocessing pipeline and train the model
    print("\nPreprocessing and training model...")
    # 1. Encode categorical variables (teams and players)
    df = data_preprocessor.encode_categorical(df)
    base_model.set_encoders(data_preprocessor.team_encoder, data_preprocessor.player_encoder)
    # 2. Generate position-related features
    df = position_generator.create_position_features(df)
    # 3. Calculate chemistry scores between players
    df = chemistry_analyzer.calculate_pair_chemistry(df)
    # 4. Analyze time patterns in the data
    time_analyzer.analyze_time_patterns(df)
    # 5. Prepare feature matrix and target vector
    X, y = data_preprocessor.prepare_data(df)
    # 6. Train the model
    base_model.train(X, y)
    
    # Create predictor interface that integrates all components
    predictor = PredictorInterface(base_model, position_generator, chemistry_analyzer, time_analyzer, data=df)
    
    return predictor, df

def load_test_data():
    """Load test data and labels"""
    # Set up paths to test data files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    test_path = os.path.join(project_dir, 'data', 'NBA_test.csv')
    labels_path = os.path.join(project_dir, 'data', 'NBA_test_labels.csv')
    
    # Load test cases and their corresponding labels
    test_df = pd.read_csv(test_path)
    labels_df = pd.read_csv(labels_path)
    
    print(f"Loaded {len(test_df)} test samples")
    print(f"Loaded {len(labels_df)} label samples")
    
    # Print column information for debugging
    print("Test data columns:", test_df.columns.tolist())
    print("Labels data columns:", labels_df.columns.tolist())
    
    return test_df, labels_df

def evaluate_model(predictor, test_df, labels_df):
    """Evaluate model on test data with improved accuracy calculation"""
    print("Evaluating model on test data...")
    results = []
    # Track different types of errors that can occur during evaluation
    errors = {"no_prediction": 0, "insufficient_players": 0, "unseen_player": 0, "unknown_label": 0}
    
    # Identify the column containing the removed player (target)
    label_column = "removed_value" if "removed_value" in labels_df.columns else labels_df.columns[0]
    print(f"Using '{label_column}' as the player label column")
    
    # Get the set of players the model was trained on
    known_players = set(predictor.model.player_encoder.classes_)
    print(f"Model knows {len(known_players)} unique players")
    
    # Process each test case
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            # Extract key information from the test row
            season = str(row['season'])
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get the true fifth player that should be predicted
            true_label = labels_df.loc[i, label_column] if i < len(labels_df) else None
            
            # Skip cases with missing or unknown labels
            if not true_label or pd.isna(true_label) or true_label == '?':
                errors["unknown_label"] += 1
                continue
                
            # Skip if the player to predict wasn't in the training data
            if true_label not in known_players:
                errors["unseen_player"] += 1
                continue
            
            # Get the game time information (defaults to 15.0 minutes)
            game_time = row.get('starting_min', 15.0)
            
            # Extract the 4 known home players (excluding the one to predict)
            home_players = []
            for j in range(5):
                col_name = f'home_{j}'
                if col_name in row and pd.notna(row[col_name]) and row[col_name] != true_label:
                    # Only include players the model knows about
                    if row[col_name] in known_players:
                        home_players.append(row[col_name])
                    if len(home_players) >= 4:  # Stop once we have 4 players
                        break
            
            # Extract away team players
            away_players = []
            for j in range(5):
                col_name = f'away_{j}'
                if col_name in row and pd.notna(row[col_name]):
                    # Only include players the model knows about
                    if row[col_name] in known_players:
                        away_players.append(row[col_name])
            
            # Skip test cases without enough home players
            if len(home_players) < 4:
                errors["insufficient_players"] += 1
                continue
                
            # Use available away players (up to 5)
            away_players = away_players[:min(5, len(away_players))]
            
            # Make prediction for the fifth player
            predictions = predictor.predict_fifth_player(
                season=season,
                home_team=home_team,
                away_team=away_team,
                current_players=home_players,
                away_players=away_players,
                game_time=game_time
            )
            
            # Handle case where no prediction was made
            if not predictions:
                errors["no_prediction"] += 1
                continue
                
            # Record prediction results
            if true_label:
                # Get top predicted player and confidence
                predicted_player = list(predictions.keys())[0]
                confidence = list(predictions.values())[0]
                
                # Store detailed result information
                result = {
                    'id': i,
                    'season': season,
                    'home_team': home_team,
                    'away_team': away_team,
                    'true_player': true_label,
                    'predicted_player': predicted_player,
                    'confidence': confidence,
                    'is_correct': predicted_player == true_label
                }
                results.append(result)
                
        except Exception as e:
            # Log any errors during processing
            print(f"Error processing row {i}: {str(e)}")
            continue
    
    # Summarize error counts
    print("\nError types:")
    for error_type, count in errors.items():
        print(f"  {error_type}: {count}")
        
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Calculate and display overall accuracy
    if not results_df.empty:
        accuracy = results_df['is_correct'].mean() * 100
        print(f"\nOverall accuracy: {accuracy:.2f}% ({results_df['is_correct'].sum()}/{len(results_df)} correct)")
    else:
        print("No valid predictions made")
    
    return results_df

def analyze_results(results_df):
    """Analyze evaluation results with more detailed metrics"""
    if len(results_df) == 0:
        print("No valid results to analyze.")
        return {}
    
    # Calculate overall accuracy metrics
    correct_predictions = results_df['is_correct'].sum()
    total_predictions = len(results_df)
    accuracy = results_df['is_correct'].mean() * 100
    
    print(f"\nTest Results Summary:")
    print(f"Total valid test cases: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    
    # Calculate accuracy broken down by season
    # This helps identify which seasons the model performs better or worse on
    season_accuracy = results_df.groupby('season')['is_correct'].agg(['mean', 'count', 'sum'])
    season_accuracy['mean'] = season_accuracy['mean'] * 100
    
    print("\nAccuracy by season:")
    for season, row in season_accuracy.iterrows():
        print(f"  {season}: {row['mean']:.2f}% ({row['sum']}/{row['count']} correct)")
    
    # Analyze distribution of test cases across seasons
    matches_per_year = results_df['season'].value_counts().sort_index()
    avg_matches = matches_per_year.mean()
    
    print("\nMatches per year in test dataset:")
    for year, count in matches_per_year.items():
        print(f"  {year}: {count}")
    
    print(f"\nAverage number of matches across the dataset: {avg_matches:.2f}")
    
    # Display high-confidence correct predictions
    if not results_df[results_df['is_correct']].empty:
        print("\nTop 5 Correct Predictions (Highest Confidence):")
        top_correct = results_df[results_df['is_correct']].sort_values('confidence', ascending=False).head(5)
        for _, row in top_correct.iterrows():
            print(f"  {row['true_player']} (Season: {row['season']}, {row['home_team']} vs {row['away_team']}, Confidence: {row['confidence']:.2f})")
    
    # Create visualization for matches per year
    plt.figure(figsize=(10, 6))
    plt.bar(matches_per_year.index.astype(str), matches_per_year.values)
    plt.title('Number of Matches per Year in Test Dataset')
    plt.xlabel('Year')
    plt.ylabel('Number of Matches')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Set up directory for saving evaluation outputs
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, 'evaluation')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save matches per year visualization
    plt.savefig(os.path.join(output_dir, 'matches_per_year.png'))
    print(f"\nVisualization saved to {output_dir}/matches_per_year.png")
    
    # Create and save accuracy by season visualization
    plt.figure(figsize=(10, 6))
    seasons = season_accuracy.index.astype(str)
    accuracies = season_accuracy['mean'].values
    
    plt.bar(seasons, accuracies)
    plt.title('Accuracy by Season')
    plt.xlabel('Season')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)  # Set y-axis from 0-100%
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'accuracy_by_season.png'))
    print(f"Second visualization saved to {output_dir}/accuracy_by_season.png")
    
    # Save detailed results to CSV for further analysis
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    print(f"Results saved to {output_dir}/evaluation_results.csv")
    
    # Create summary metrics file for easy reference
    summary_df = pd.DataFrame({
        'metric': ['total_predictions', 'correct_predictions', 'accuracy'],
        'value': [total_predictions, correct_predictions, accuracy]
    })
    summary_df.to_csv(os.path.join(output_dir, 'evaluation_summary.csv'), index=False)
    print(f"Summary saved to {output_dir}/evaluation_summary.csv")
    
    # Compile key metrics into a dictionary for return value
    metrics = {
        'matches_per_year': matches_per_year.to_dict(),
        'average_matches': avg_matches,
        'overall_accuracy': accuracy,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'season_accuracy': season_accuracy['mean'].to_dict()
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