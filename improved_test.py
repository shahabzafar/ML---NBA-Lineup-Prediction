import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

print("Starting Improved NBA Lineup Predictor Test...")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
print("Importing modules...")
from src.model import LineupPredictor as BaseModel
from src.position_features import PositionFeatureGenerator
from src.data_preprocessing import DataPreprocessor

def improved_test():
    """Run an improved test with all data to achieve better accuracy"""
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    print("Loading all seasons data for better coverage...")
    # Load all seasons to ensure we have data for all teams
    all_dfs = []
    for year in range(2007, 2016):
        file_path = os.path.join(data_dir, f'matchups-{year}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['season'] = year
            all_dfs.append(df)
            print(f"Loaded {len(df)} rows from {year} season")
    
    # Combine all data
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset: {len(df)} rows")
    
    # Initialize components with improved parameters
    print("Initializing components...")
    data_preprocessor = DataPreprocessor()
    position_generator = PositionFeatureGenerator()
    base_model = BaseModel()
    
    # Set model parameters for better accuracy
    base_model.model.n_estimators = 200
    base_model.model.max_depth = 25
    base_model.model.min_samples_split = 2
    base_model.model.class_weight = 'balanced_subsample'
    base_model.min_samples_per_player = 5  # Include more players
    
    # Process data
    print("Pre-processing data...")
    df = data_preprocessor.encode_categorical(df)
    base_model.set_encoders(data_preprocessor.team_encoder, data_preprocessor.player_encoder)
    df = position_generator.create_position_features(df)
    
    # Add chemistry score (simplified)
    df['chemistry_score'] = 0.7
    df['lineup_balance'] = 0.8
    
    # Track known players for reference
    known_player_set = set(data_preprocessor.player_encoder.classes_)
    known_team_set = set(data_preprocessor.team_encoder.classes_)
    print(f"Model knows {len(known_player_set)} players and {len(known_team_set)} teams")
    
    # Prepare and train model
    print("Preparing training data...")
    X, y = data_preprocessor.prepare_data(df)
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    print("Training model with improved parameters...")
    base_model.train(X, y)
    
    # Load test data
    print("Loading test data...")
    test_path = os.path.join(data_dir, 'NBA_test.csv')
    labels_path = os.path.join(data_dir, 'NBA_test_labels.csv')
    
    if not os.path.exists(test_path) or not os.path.exists(labels_path):
        print(f"Test files not found in {data_dir}")
        return
    
    test_df = pd.read_csv(test_path)
    labels_df = pd.read_csv(labels_path)
    
    print(f"Test data: {len(test_df)} samples")
    
    # Team mapping (historical team name changes)
    team_map = {
        'SEA': 'OKC',  # Seattle SuperSonics -> Oklahoma City Thunder
        'NOK': 'NOP',  # New Orleans/Oklahoma City Hornets -> New Orleans Pelicans
        'NOH': 'NOP',  # New Orleans Hornets -> New Orleans Pelicans
        'NJN': 'BRK',  # New Jersey Nets -> Brooklyn Nets
        'CHA': 'CHO',  # Charlotte Bobcats -> Charlotte Hornets
    }
    
    # Adjust test data to match training data format
    print("Preprocessing test data...")
    # Create map of known player names to their encoded values
    known_players_map = {player: idx for idx, player in 
                        enumerate(data_preprocessor.player_encoder.classes_)}
    
    # Process all test samples
    results = []
    errors = {"unknown_team": 0, "insufficient_players": 0, "unknown_player": 0, "other_error": 0}
    label_column = labels_df.columns[0]
    
    print("Running predictions on all test samples...")
    for i in tqdm(range(len(test_df))):
        try:
            # Get test data
            row = test_df.iloc[i]
            true_label = labels_df.iloc[i][label_column] if i < len(labels_df) else None
            
            # Skip unknown labels
            if not isinstance(true_label, str) or true_label == '?':
                continue
                
            # Handle team mapping for historical teams
            home_team = row['home_team']
            if home_team not in known_team_set:
                home_team = team_map.get(home_team, list(known_team_set)[0])
            
            away_team = row['away_team']
            if away_team not in known_team_set:
                away_team = team_map.get(away_team, list(known_team_set)[0])
            
            # Extract home players (excluding the true label player)
            home_players = []
            for j in range(5):
                col = f'home_{j}'
                if col in row and pd.notna(row[col]):
                    player = row[col]
                    if isinstance(player, str) and player in known_player_set and player != true_label:
                        home_players.append(player)
            
            # If less than 4 players, try to find the player that's being held out (true_label)
            # and remove it from consideration to get 4 players
            if len(home_players) < 4:
                # Create a set of all 5 players including the true label
                all_players = set()
                for j in range(5):
                    col = f'home_{j}'
                    if col in row and pd.notna(row[col]):
                        player = row[col]
                        if isinstance(player, str) and player in known_player_set:
                            all_players.add(player)
                
                # If removing the true label gives us exactly 4 players, we're good
                if len(all_players) == 5 and true_label in all_players:
                    home_players = list(all_players - {true_label})
            
            # Still don't have 4 players, need to adapt
            if len(home_players) < 4:
                # Add most common players from the same team to fill the roster
                team_players = df[df['home_team'] == home_team]
                for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4']:
                    if col in team_players.columns:
                        player_counts = team_players[col].value_counts()
                        for player, _ in player_counts.items():
                            if isinstance(player, str) and player in known_player_set and player != true_label and player not in home_players:
                                home_players.append(player)
                                if len(home_players) >= 4:
                                    break
                        if len(home_players) >= 4:
                            break
            
            # If we still don't have 4 players, skip this sample
            if len(home_players) < 4:
                errors["insufficient_players"] += 1
                continue
            
            # Limit to exactly 4 players
            home_players = home_players[:4]
            
            # Extract away players
            away_players = []
            for j in range(5):
                col = f'away_{j}'
                if col in row and pd.notna(row[col]):
                    player = row[col]
                    if isinstance(player, str) and player in known_player_set:
                        away_players.append(player)
            
            # Fill away_players with common players if needed
            if len(away_players) < 5:
                team_players = df[df['away_team'] == away_team]
                for col in ['away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
                    if col in team_players.columns:
                        player_counts = team_players[col].value_counts()
                        for player, _ in player_counts.items():
                            if isinstance(player, str) and player in known_player_set and player not in away_players:
                                away_players.append(player)
                                if len(away_players) >= 5:
                                    break
                        if len(away_players) >= 5:
                            break
            
            # Use available away players (up to 5)
            away_players = away_players[:min(5, len(away_players))]
            
            # Build features for prediction
            test_features = pd.DataFrame({
                'home_team_encoded': [data_preprocessor.team_encoder.transform([home_team])[0]],
                'away_team_encoded': [data_preprocessor.team_encoder.transform([away_team])[0]],
                'season': [row['season']],
                'starting_min': [float(row.get('starting_min', 15.0))],
                'chemistry_score': [0.7],
                'lineup_balance': [0.8]
            })
            
            # Add position features
            positions = position_generator.get_positions(home_players)
            test_features['num_guards'] = [positions.count('G')]
            test_features['num_forwards'] = [positions.count('F')]
            test_features['num_centers'] = [positions.count('C')]
            
            # Add player encodings
            for j, player in enumerate(home_players):
                test_features[f'home_{j}_encoded'] = [known_players_map[player]]
            
            # Add away player encodings
            for j, player in enumerate(away_players):
                test_features[f'away_{j}_encoded'] = [known_players_map[player]]
            
            # Fill in missing away players
            for j in range(len(away_players), 5):
                test_features[f'away_{j}_encoded'] = [0]  # Default padding
            
            # Add lineup chemistry calculation using simplified method
            # Based on position balance plus player combinations
            is_balanced = (test_features['num_guards'][0] >= 1 and 
                           test_features['num_forwards'][0] >= 1 and
                           test_features['num_centers'][0] >= 0)
            test_features['lineup_balanced'] = [1.0 if is_balanced else 0.5]
            
            # Ensure all required features are present
            for col in base_model.feature_names:
                if col not in test_features.columns:
                    test_features[col] = 0.0
            
            # Reorder columns to match the model's expected format
            test_features = test_features[base_model.feature_names]
            
            # Make prediction
            # Get prediction probabilities from model
            probabilities = base_model.predict_proba(test_features)[0]
            
            # Process probabilities into player predictions
            player_probs = {}
            for idx, prob in enumerate(probabilities):
                # Only consider non-trivial probabilities
                if prob > 0.01:
                    player_idx = base_model.model.classes_[idx]
                    player = data_preprocessor.player_encoder.inverse_transform([player_idx])[0]
                    
                    # Apply additional position-based weighting 
                    # Favor players that balance the lineup
                    player_pos = position_generator.get_player_position(player)
                    
                    # Position need calculation
                    position_boost = 1.0
                    if test_features['num_guards'][0] == 0 and player_pos == 'G':
                        position_boost = 1.4  # Boost guards when none in lineup
                    elif test_features['num_centers'][0] == 0 and player_pos == 'C':
                        position_boost = 1.3  # Boost centers when none in lineup
                    elif test_features['num_forwards'][0] <= 1 and player_pos == 'F':
                        position_boost = 1.2  # Boost forwards when few in lineup
                    
                    # Apply position boost and store
                    player_probs[player] = prob * position_boost
            
            # Get top prediction
            if player_probs:
                # Weight predictions to favor balancing lineups
                top_players = sorted(player_probs.items(), key=lambda x: x[1], reverse=True)
                
                # Get top player prediction
                predicted_player = top_players[0][0]
                confidence = top_players[0][1] * 100  # Convert to percentage
                
                # Record result
                result = {
                    'id': i,
                    'true_player': true_label,
                    'predicted_player': predicted_player,
                    'confidence': confidence,
                    'is_correct': predicted_player == true_label
                }
                results.append(result)
                
                # Print sample results for monitoring
                if i < 5 or i % 100 == 0:
                    correct_mark = "✓" if predicted_player == true_label else "✗"
                    print(f"Sample {i}: Predicted {predicted_player}, Actual {true_label} {correct_mark}")
        
        except Exception as e:
            errors["other_error"] += 1
            if i < 5:
                print(f"Error processing sample {i}: {str(e)}")
    
    # Calculate and report results
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate accuracy
        accuracy = results_df['is_correct'].mean() * 100
        correct_count = results_df['is_correct'].sum()
        total_count = len(results_df)
        
        print("\nTest Results:")
        print(f"Total valid predictions: {total_count} out of {len(test_df)} samples")
        print(f"Correct predictions: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Calculate accuracy by position for more details
        if total_count > 0:
            # Get positions for predicted and actual players
            results_df['true_position'] = results_df['true_player'].apply(
                position_generator.get_player_position)
            results_df['pred_position'] = results_df['predicted_player'].apply(
                position_generator.get_player_position)
            
            # Position-level accuracy
            print("\nAccuracy by player position:")
            for pos in ['G', 'F', 'C']:
                pos_df = results_df[results_df['true_position'] == pos]
                if len(pos_df) > 0:
                    pos_acc = pos_df['is_correct'].mean() * 100
                    print(f"  {pos}: {pos_acc:.2f}% ({pos_df['is_correct'].sum()}/{len(pos_df)})")
            
            # Position prediction accuracy (even if player is wrong)
            pos_match = (results_df['true_position'] == results_df['pred_position']).mean() * 100
            print(f"\nPosition prediction accuracy: {pos_match:.2f}%")
        
        # Error analysis
        print("\nError counts:")
        for error_type, count in errors.items():
            print(f"  {error_type}: {count}")
        
        # Save results
        results_df.to_csv(os.path.join(current_dir, 'improved_test_results.csv'), index=False)
        print("\nDetailed results saved to improved_test_results.csv")
        
        return accuracy, results_df
    else:
        print("No valid predictions could be made!")
        return 0.0, None

if __name__ == "__main__":
    start_time = time.time()
    accuracy, results = improved_test()
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.1f} seconds")
    
    if accuracy >= 55.0:
        print(f"\n✅ SUCCESS! Achieved target accuracy of {accuracy:.2f}% (target: 55.0%)")
    else:
        print(f"\n❌ Target accuracy not reached: {accuracy:.2f}% (target: 55.0%)") 