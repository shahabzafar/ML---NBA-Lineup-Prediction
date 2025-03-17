import pandas as pd
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.prediction_interface import LineupPredictor
from src.data_preprocessing import DataPreprocessor
from src.model import LineupPredictor as BaseModel
from src.position_features import PositionFeatureGenerator
from src.chemistry_analysis import ChemistryAnalyzer
from src.time_analysis import TimeAnalyzer

def load_model_and_test_data():
    """Load the model and test data"""
    print("Loading model and test data...")
    
    # Initialize components
    data_preprocessor = DataPreprocessor()
    position_generator = PositionFeatureGenerator()
    chemistry_analyzer = ChemistryAnalyzer()
    time_analyzer = TimeAnalyzer()
    base_model = BaseModel()
    
    # Load training data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
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
    print(f"Total training rows loaded: {len(df)}")
    
    # Preprocess and train
    print("\nPreprocessing and training model...")
    df = data_preprocessor.encode_categorical(df)
    base_model.set_encoders(data_preprocessor.team_encoder, data_preprocessor.player_encoder)
    df = position_generator.create_position_features(df)
    df = chemistry_analyzer.calculate_pair_chemistry(df)
    time_analyzer.analyze_time_patterns(df)
    X, y = data_preprocessor.prepare_data(df)
    base_model.train(X, y)
    
    # Create predictor interface
    predictor = LineupPredictor(base_model, position_generator, chemistry_analyzer, time_analyzer, data=df)
    
    # Load test data
    test_path = os.path.join(data_dir, 'NBA_test.csv')
    labels_path = os.path.join(data_dir, 'NBA_test_labels.csv')
    
    if not os.path.exists(test_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Test files not found. Make sure NBA_test.csv and NBA_test_labels.csv are in {data_dir}")
    
    test_df = pd.read_csv(test_path)
    labels_df = pd.read_csv(labels_path)
    
    print(f"Loaded {len(test_df)} test samples")
    print(f"Loaded {len(labels_df)} label samples")
    
    # Check for unknown player labels
    label_column = labels_df.columns[0]
    unknown_labels = labels_df[labels_df[label_column] == '?']
    if not unknown_labels.empty:
        print(f"WARNING: Found {len(unknown_labels)} unknown player labels ('?') in the test data")
    
    return predictor, test_df, labels_df, data_preprocessor

def analyze_test_results():
    """Analyze test results in detail, handling unknown players and errors"""
    predictor, test_df, labels_df, data_preprocessor = load_model_and_test_data()
    
    # Get the label column name (first column in the labels file)
    label_column = labels_df.columns[0]
    print(f"Using '{label_column}' as the player label column")
    
    # Get known player names from the encoder
    known_players = set(data_preprocessor.player_encoder.classes_)
    print(f"Model knows {len(known_players)} unique players")
    
    results = []
    errors = []
    total = len(test_df)
    
    print(f"\nAnalyzing {total} test cases...")
    for i, row in tqdm(test_df.iterrows(), total=total):
        if i >= len(labels_df):
            break
            
        # Extract data from test row
        season = str(row['season'])
        home_team = row['home_team']
        away_team = row['away_team']
        true_label = labels_df.iloc[i][label_column]
        
        # Skip cases with unknown labels
        if true_label == '?' or not isinstance(true_label, str):
            errors.append({
                'id': i,
                'error_type': 'unknown_label',
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'label': str(true_label)
            })
            continue
        
        # Skip if true player isn't in our training data
        if true_label not in known_players:
            errors.append({
                'id': i,
                'error_type': 'unseen_player',
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'label': true_label
            })
            continue
        
        # Get players for prediction
        home_players = []
        for j in range(5):
            col = f'home_{j}'
            if col in row and pd.notna(row[col]) and row[col] != true_label:
                # Skip unknown or invalid players
                if row[col] == '?' or not isinstance(row[col], str):
                    continue
                if row[col] in known_players:
                    home_players.append(row[col])
                if len(home_players) >= 4:  # Only need 4 players
                    break
        
        away_players = []
        for j in range(5):
            col = f'away_{j}'
            if col in row and pd.notna(row[col]):
                # Skip unknown or invalid players
                if row[col] == '?' or not isinstance(row[col], str):
                    continue
                if row[col] in known_players:
                    away_players.append(row[col])
        
        # Skip if we don't have enough players
        if len(home_players) < 4 or len(away_players) < 1:  # Only need 1 away player minimum
            errors.append({
                'id': i,
                'error_type': 'insufficient_players',
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'home_players_found': len(home_players),
                'away_players_found': len(away_players)
            })
            continue
        
        # Make prediction with error handling
        try:
            game_time = row.get('starting_min', 15.0)
            predictions = predictor.predict_fifth_player(
                season=season,
                home_team=home_team,
                away_team=away_team,
                current_players=home_players[:4],  # Use first 4 players
                away_players=away_players[:min(5, len(away_players))],  # Use up to 5 players
                game_time=game_time
            )
            
            # No prediction made
            if not predictions:
                errors.append({
                    'id': i,
                    'error_type': 'no_prediction',
                    'season': season,
                    'home_team': home_team,
                    'away_team': away_team
                })
                continue
            
            # Get top prediction
            top_prediction = list(predictions.keys())[0]
            confidence = list(predictions.values())[0]
            
            # Record result
            result = {
                'id': i,
                'season': int(season),
                'home_team': home_team,
                'away_team': away_team,
                'true_player': true_label,
                'predicted_player': top_prediction,
                'confidence': confidence,
                'is_correct': top_prediction == true_label
            }
            results.append(result)
            
        except Exception as e:
            errors.append({
                'id': i,
                'error_type': 'prediction_error',
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'error': str(e)
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    errors_df = pd.DataFrame(errors) if errors else pd.DataFrame()
    
    # Save detailed results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not results_df.empty:
        results_df.to_csv(os.path.join(output_dir, 'detailed_test_results.csv'), index=False)
    
    if not errors_df.empty:
        errors_df.to_csv(os.path.join(output_dir, 'test_errors.csv'), index=False)
        print(f"\nErrors encountered in {len(errors_df)} test cases")
        print(f"Error types:")
        for error_type, count in errors_df['error_type'].value_counts().items():
            print(f"  {error_type}: {count}")
    
    # Print summary statistics
    if not results_df.empty:
        total_predictions = len(results_df)
        correct_predictions = results_df['is_correct'].sum()
        accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        
        print(f"\nTest Results Summary:")
        print(f"Total valid test cases: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Print accuracy by season
        if 'season' in results_df.columns:
            print("\nAccuracy by Season:")
            season_accuracy = results_df.groupby('season')['is_correct'].agg(['mean', 'count'])
            season_accuracy['mean'] = season_accuracy['mean'] * 100
            for season, row in season_accuracy.iterrows():
                print(f"  {season}: {row['mean']:.2f}% ({row['count']} predictions)")
        
        # Print top 5 correct predictions with highest confidence
        if not results_df[results_df['is_correct']].empty:
            print("\nTop 5 Correct Predictions (Highest Confidence):")
            top_correct = results_df[results_df['is_correct']].sort_values('confidence', ascending=False).head(5)
            for _, row in top_correct.iterrows():
                print(f"  {row['true_player']} (Season: {row['season']}, {row['home_team']} vs {row['away_team']}, Confidence: {row['confidence']:.2f})")
        
        # Print incorrect predictions with highest confidence (model was confidently wrong)
        if not results_df[~results_df['is_correct']].empty:
            print("\nTop 5 Incorrect Predictions (Highest Confidence):")
            top_incorrect = results_df[~results_df['is_correct']].sort_values('confidence', ascending=False).head(5)
            for _, row in top_incorrect.iterrows():
                print(f"  Predicted: {row['predicted_player']}, Actual: {row['true_player']} (Season: {row['season']}, Confidence: {row['confidence']:.2f})")
        
        # Analyze team performance
        if 'home_team' in results_df.columns:
            team_results = results_df.groupby('home_team').agg({
                'is_correct': ['count', 'sum', 'mean']
            })
            team_results.columns = ['total', 'correct', 'accuracy']
            team_results['accuracy'] = team_results['accuracy'] * 100
            
            # Only show teams with at least 3 predictions
            if not team_results[team_results['total'] >= 3].empty:
                print("\nTeam Performance (min 3 predictions):")
                for team, row in team_results[team_results['total'] >= 3].sort_values('accuracy', ascending=False).iterrows():
                    print(f"  {team}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
        
        # Save a more readable version of results
        readable_results = results_df[['id', 'season', 'home_team', 'away_team', 'true_player', 
                                      'predicted_player', 'confidence', 'is_correct']]
        readable_results.to_csv(os.path.join(output_dir, 'readable_test_results.csv'), index=False)
        
        print(f"\nDetailed results saved to: {os.path.join(output_dir, 'detailed_test_results.csv')}")
        print(f"Readable results saved to: {os.path.join(output_dir, 'readable_test_results.csv')}")
    else:
        print("\nNo valid predictions could be made on the test data.")
    
    if not errors_df.empty:
        print(f"Error logs saved to: {os.path.join(output_dir, 'test_errors.csv')}")
    
    return results_df, errors_df

if __name__ == "__main__":
    results_df, errors_df = analyze_test_results() 