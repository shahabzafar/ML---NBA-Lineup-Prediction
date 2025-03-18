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
    # Use glob to find all matchup CSV files with the pattern 'matchups-*.csv'
    csv_files = glob.glob(os.path.join(data_dir, 'matchups-*.csv'))
    
    for file in csv_files:
        # Extract season year from the filename
        season = os.path.basename(file).split('-')[1].split('.')[0]
        df = pd.read_csv(file)
        # Add a season column to identify data source
        df['season'] = season
        print(f"Loaded {len(df)} rows from season {season}")
        all_data.append(df)
    
    # Concatenate all seasons into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: Loaded {len(combined_df)} rows from {len(csv_files)} seasons")
    return combined_df

def get_team_players(df: pd.DataFrame, team: str) -> List[str]:
    """Get all players who played for a team"""
    # Filter DataFrame to only include the specified team's home games
    team_data = df[df['home_team'] == team]
    players = set()
    # Collect unique players from all 5 home player positions
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
    
    # Initialize components of the prediction system
    data_preprocessor = DataPreprocessor()
    position_generator = PositionFeatureGenerator()
    chemistry_analyzer = ChemistryAnalyzer()
    time_analyzer = TimeAnalyzer()
    base_model = BaseModel()
    
    # Preprocess data and train the model
    print("\nPreprocessing and training model...")
    # 1. Encode categorical variables
    df = data_preprocessor.encode_categorical(df)
    base_model.set_encoders(data_preprocessor.team_encoder, data_preprocessor.player_encoder)
    # 2. Generate position-based features
    df = position_generator.create_position_features(df)
    # 3. Analyze player chemistry between teammates
    df = chemistry_analyzer.calculate_pair_chemistry(df)
    # 4. Analyze time patterns in lineup usage
    time_patterns = time_analyzer.analyze_time_patterns(df)
    # 5. Prepare data for model training
    X, y = data_preprocessor.prepare_data(df)
    # 6. Train the lineup prediction model
    base_model.train(X, y)
    
    # Create predictor interface that combines all components
    predictor = PredictorInterface(base_model, position_generator, chemistry_analyzer, time_analyzer)
    
    # Interactive loop for user predictions
    while True:
        print("\n=== NBA Lineup Predictor (2007-2015) ===")
        available_teams = sorted(df['home_team'].unique())
        print("Available teams:", ', '.join(available_teams))
        
        # Get team input from user
        team = input("\nEnter team code (or 'quit' to exit): ").upper()
        if team.lower() == 'quit':
            break
        
        # Validate team selection
        if team not in available_teams:
            print(f"Invalid team code! Please choose from: {', '.join(available_teams)}")
            continue
        
        # Show available players for the selected team
        team_players = get_team_players(df, team)
        print(f"\nAvailable players for {team}:")
        for i, player in enumerate(team_players, 1):
            print(f"{i}. {player}")
        
        # Get player inputs from user
        try:
            print("\nEnter the numbers of the 4 players currently on court (separated by spaces):")
            # Convert input string to list of player indices
            player_indices = [int(x)-1 for x in input().split()]
            if len(player_indices) != 4:
                print("Please select exactly 4 players!")
                continue
            
            # Validate player indices
            if any(i < 0 or i >= len(team_players) for i in player_indices):
                print("Invalid player number(s)! Please try again.")
                continue
            
            # Create list of selected players
            current_players = [team_players[i] for i in player_indices]
            
            # Get game time from user
            game_time = float(input("\nEnter game time (in minutes, 0-48): "))
            if not 0 <= game_time <= 48:
                print("Game time must be between 0 and 48 minutes!")
                continue
            
            # Make lineup prediction based on inputs
            print("\nPredicting best fifth player...")
            prediction = predictor.predict_fifth_player(team, current_players, game_time)
            
            # Display top 10 predictions with confidence scores
            print("\nTop 10 recommended players:")
            sorted_predictions = sorted(prediction.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (player, prob) in enumerate(sorted_predictions, 1):
                print(f"{i}. {player:<20} {prob:.1%} confidence")
            
        except (ValueError, IndexError) as e:
            print(f"Invalid input! Please try again. Error: {str(e)}")
            continue

if __name__ == "__main__":
    interactive_prediction() 