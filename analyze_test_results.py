import pandas as pd
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

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
    start_time = time.time()
    
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
    
    # Step 1: Encode categorical data
    print("Encoding categorical data...")
    df = data_preprocessor.encode_categorical(df)
    base_model.set_encoders(data_preprocessor.team_encoder, data_preprocessor.player_encoder)
    
    # Step 2: Generate position features
    print("Generating position features...")
    df = position_generator.create_position_features(df)
    
    # Step 3: Calculate chemistry 
    print("Calculating chemistry scores...")
    df = chemistry_analyzer.calculate_pair_chemistry(df)
    
    # Step 4: Analyze time patterns
    print("Analyzing time patterns...")
    time_analyzer.analyze_time_patterns(df)
    
    # Step 5: Prepare data and train model
    print("Preparing training data...")
    X, y = data_preprocessor.prepare_data(df)
    
    print("Training model...")
    base_model.train(X, y)
    
    # Create predictor interface
    predictor = LineupPredictor(base_model, position_generator, chemistry_analyzer, time_analyzer, data=df)
    
    # Load test data
    test_path = os.path.join(data_dir, 'NBA_test.csv')
    labels_path = os.path.join(data_dir, 'NBA_test_labels.csv')
    
    if not os.path.exists(test_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"Test files not found. Make sure NBA_test.csv and NBA_test_labels.csv are in {data_dir}")
    
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    labels_df = pd.read_csv(labels_path)
    
    print(f"Loaded {len(test_df)} test samples")
    print(f"Loaded {len(labels_df)} label samples")
    
    # Check for unknown player labels
    label_column = labels_df.columns[0]
    unknown_labels = labels_df[labels_df[label_column] == '?']
    if not unknown_labels.empty:
        print(f"WARNING: Found {len(unknown_labels)} unknown player labels ('?') in the test data")
    
    elapsed_time = time.time() - start_time
    print(f"Model preparation completed in {elapsed_time:.1f} seconds")
    
    return predictor, test_df, labels_df, data_preprocessor

def analyze_test_results():
    """Analyze test results in detail, handling unknown players and errors"""
    try:
        predictor, test_df, labels_df, data_preprocessor = load_model_and_test_data()
        
        # Get the label column name (first column in the labels file)
        label_column = labels_df.columns[0]
        print(f"\nUsing '{label_column}' as the player label column")
        
        # Get known player names from the encoder
        known_players = set(data_preprocessor.player_encoder.classes_)
        print(f"Model knows {len(known_players)} unique players")
        
        results = []
        errors = {"no_prediction": 0, "insufficient_players": 0, "unseen_player": 0, "unknown_label": 0}
        total = len(test_df)
        
        print(f"\nAnalyzing {total} test cases...")
        prediction_start = time.time()
        
        for i, row in tqdm(test_df.iterrows(), total=total):
            if i >= len(labels_df):
                break
                
            # Extract data from test row
            season = str(row['season'])
            home_team = row['home_team']
            away_team = row['away_team']
            true_label = labels_df.iloc[i][label_column]
            
            # Skip cases with unknown labels
            if true_label == '?' or not isinstance(true_label, str) or pd.isna(true_label):
                errors["unknown_label"] += 1
                continue
            
            # Skip if true player isn't in our training data
            if true_label not in known_players:
                errors["unseen_player"] += 1
                continue
            
            # Get home players (excluding the removed player)
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
            
            # Get away players
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
            if len(home_players) < 4:
                errors["insufficient_players"] += 1
                continue
                
            # Allow prediction with limited away players if needed
            away_players = away_players[:min(5, len(away_players))]
            
            # Make prediction with error handling
            try:
                game_time = row.get('starting_min', 15.0)
                predictions = predictor.predict_fifth_player(
                    season=season,
                    home_team=home_team,
                    away_team=away_team,
                    current_players=home_players,
                    away_players=away_players,
                    game_time=game_time
                )
                
                # No prediction made
                if not predictions:
                    errors["no_prediction"] += 1
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
                errors["no_prediction"] += 1
                if i < 5:  # Only print the first few errors to avoid log flooding
                    print(f"Error predicting for row {i}: {str(e)}")
                continue
        
        prediction_time = time.time() - prediction_start
        print(f"Completed predictions in {prediction_time:.1f} seconds")
        
        # Print error stats
        print("\nError types:")
        for error_type, count in errors.items():
            print(f"  {error_type}: {count}")
                
        # Convert to DataFrame
        results_df = pd.DataFrame(results) if results else pd.DataFrame()
        
        # Print summary statistics
        if not results_df.empty:
            total_predictions = len(results_df)
            correct_predictions = results_df['is_correct'].sum()
            accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
            
            print(f"\nTest Results Summary:")
            print(f"Total valid test cases: {total_predictions}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Accuracy: {accuracy:.2f}%")
            
            # Add debug information to help identify breakdown
            print(f"\nResults Breakdown:")
            print(f"- Total dataset rows: {total}")
            print(f"- Rows with valid predictions: {total_predictions}")
            print(f"- Correct predictions: {correct_predictions} ({accuracy:.2f}%)")
            
            # Print accuracy by season
            if 'season' in results_df.columns:
                print("\nAccuracy by Season:")
                season_accuracy = results_df.groupby('season')['is_correct'].agg(['mean', 'count', 'sum'])
                season_accuracy['mean'] = season_accuracy['mean'] * 100
                for season, row in season_accuracy.iterrows():
                    print(f"  {season}: {row['mean']:.2f}% ({row['sum']}/{row['count']} correct)")
            
            # Create 'evaluation' directory if it doesn't exist
            eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            
            # Save results for additional analysis
            results_df.to_csv(os.path.join(eval_dir, 'detailed_test_results.csv'), index=False)
            
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
        else:
            print("\nNo valid predictions could be made on the test data.")
        
        return results_df, errors
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}

if __name__ == "__main__":
    print("Starting NBA Lineup Predictor Evaluation...")
    results_df, errors = analyze_test_results()
    print("\nEvaluation completed!") 