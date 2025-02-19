import os
from data_preprocessing import DataPreprocessor
from position_features import PositionFeatureGenerator
from chemistry_analysis import ChemistryAnalyzer
from time_analysis import TimeAnalyzer
from prediction_interface import LineupPredictor as PredictorInterface
from model import LineupPredictor as BaseModel
import pandas as pd
import glob
from typing import List

def load_data(data_dir: str) -> pd.DataFrame:
    """Load and combine all matchup data"""
    all_data = []
    csv_files = glob.glob(os.path.join(data_dir, 'matchups-*.csv'))
    
    for file in csv_files:
        season = os.path.basename(file).split('-')[1].split('.')[0]
        df = pd.read_csv(file)
        df['season'] = season
        print(f"Loaded {len(df)} rows from season {season}")
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: Loaded {len(combined_df)} rows from {len(csv_files)} seasons")
    return combined_df

def get_team_players(df: pd.DataFrame, team: str) -> List[str]:
    """Get all players who played for a team"""
    team_data = df[df['home_team'] == team]
    players = set()
    for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4']:
        players.update(team_data[col].unique())
    return sorted(list(players))

def interactive_prediction():
    """Interactive interface for making predictions"""
    # Load and prepare data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, 'data')
    
    print("Loading and preparing data...")
    df = load_data(data_dir)
    
    # Initialize components
    data_preprocessor = DataPreprocessor()
    position_generator = PositionFeatureGenerator()
    chemistry_analyzer = ChemistryAnalyzer()
    time_analyzer = TimeAnalyzer()
    base_model = BaseModel()
    
    # Preprocess and train
    print("\nPreprocessing and training model...")
    df = data_preprocessor.encode_categorical(df)
    base_model.set_encoders(data_preprocessor.team_encoder, data_preprocessor.player_encoder)
    df = position_generator.create_position_features(df)
    df = chemistry_analyzer.calculate_pair_chemistry(df)
    time_patterns = time_analyzer.analyze_time_patterns(df)
    X, y = data_preprocessor.prepare_data(df)
    base_model.train(X, y)
    
    predictor = PredictorInterface(base_model, position_generator, chemistry_analyzer, time_analyzer)
    
    # Interactive loop
    while True:
        print("\n=== NBA Lineup Predictor (2007-2015) ===")
        available_teams = sorted(df['home_team'].unique())
        print("Available teams:", ', '.join(available_teams))
        
        # Get team input
        team = input("\nEnter team code (or 'quit' to exit): ").upper()
        if team.lower() == 'quit':
            break
        
        if team not in available_teams:
            print(f"Invalid team code! Please choose from: {', '.join(available_teams)}")
            continue
        
        # Show available players for the team
        team_players = get_team_players(df, team)
        print(f"\nAvailable players for {team}:")
        for i, player in enumerate(team_players, 1):
            print(f"{i}. {player}")
        
        # Get player inputs
        try:
            print("\nEnter the numbers of the 4 players currently on court (separated by spaces):")
            player_indices = [int(x)-1 for x in input().split()]
            if len(player_indices) != 4:
                print("Please select exactly 4 players!")
                continue
            
            if any(i < 0 or i >= len(team_players) for i in player_indices):
                print("Invalid player number(s)! Please try again.")
                continue
            
            current_players = [team_players[i] for i in player_indices]
            
            # Get game time
            game_time = float(input("\nEnter game time (in minutes, 0-48): "))
            if not 0 <= game_time <= 48:
                print("Game time must be between 0 and 48 minutes!")
                continue
            
            # Make prediction
            print("\nPredicting best fifth player...")
            prediction = predictor.predict_fifth_player(team, current_players, game_time)
            
            # Show top 10 predictions with probabilities
            print("\nTop 10 recommended players:")
            sorted_predictions = sorted(prediction.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (player, prob) in enumerate(sorted_predictions, 1):
                print(f"{i}. {player:<20} {prob:.1%} confidence")
            
        except (ValueError, IndexError) as e:
            print(f"Invalid input! Please try again. Error: {str(e)}")
            continue

if __name__ == "__main__":
    interactive_prediction() 