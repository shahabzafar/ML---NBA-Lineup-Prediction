from flask import Flask, render_template, request, jsonify
from src.position_features import PositionFeatureGenerator
from src.chemistry_analysis import ChemistryAnalyzer
from src.time_analysis import TimeAnalyzer
from src.prediction_interface import LineupPredictor as PredictorInterface
from src.model import LineupPredictor as BaseModel
from src.data_preprocessing import DataPreprocessor
import pandas as pd
import os

app = Flask(__name__)

def initialize_model():
    position_generator = PositionFeatureGenerator()
    chemistry_analyzer = ChemistryAnalyzer()
    time_analyzer = TimeAnalyzer()
    base_model = BaseModel()
    data_preprocessor = DataPreprocessor()
    
    # Fix the data loading path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    print(f"Looking for data in: {data_dir}")
    
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
    
    # Continue with preprocessing and training
    df = data_preprocessor.encode_categorical(df)
    base_model.set_encoders(data_preprocessor.team_encoder, data_preprocessor.player_encoder)
    df = position_generator.create_position_features(df)
    df = chemistry_analyzer.calculate_pair_chemistry(df)
    time_analyzer.analyze_time_patterns(df)
    X, y = data_preprocessor.prepare_data(df)
    base_model.train(X, y)
    
    predictor = PredictorInterface(base_model, position_generator, chemistry_analyzer, time_analyzer, data=df)
    return predictor, df

predictor, data = initialize_model()

@app.route('/')
def home():
    teams = sorted(data['home_team'].unique())
    return render_template('index.html', teams=teams)

@app.route('/get_teams/<season>')
def get_teams(season):
    try:
        # Filter data for the selected season
        season_data = data[data['season'] == int(season)]
        teams = sorted(list(set(season_data['home_team'].unique()) | set(season_data['away_team'].unique())))
        return jsonify(teams)
    except Exception as e:
        print(f"Error in get_teams: {str(e)}")
        return jsonify([]), 500

@app.route('/get_players/<season>/<team>')
def get_players(season, team):
    # Filter data for the selected season and team
    season_data = data[data['season'] == int(season)]
    team_data = season_data[
        (season_data['home_team'] == team) | 
        (season_data['away_team'] == team)
    ]
    
    players = set()
    # Get players from home team columns
    if len(team_data[team_data['home_team'] == team]) > 0:
        for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4']:
            players.update(team_data[team_data['home_team'] == team][col].unique())
    
    # Get players from away team columns
    if len(team_data[team_data['away_team'] == team]) > 0:
        for col in ['away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
            players.update(team_data[team_data['away_team'] == team][col].unique())
    
    return jsonify(sorted(list(players)))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.json
        season = request_data.get('season')
        home_team = request_data['home_team']
        away_team = request_data['away_team']
        home_players = request_data['home_players']
        away_players = request_data['away_players']
        game_time = float(request_data['gameTime'])
        
        # Get prediction
        predictions = predictor.predict_fifth_player(
            season=season,
            home_team=home_team,
            away_team=away_team,
            current_players=home_players,
            away_players=away_players,
            game_time=game_time
        )
        
        if not predictions:
            return jsonify({
                'predictions': [],
                'reasoning': f'No valid fifth player found for {home_team} lineup.'
            })
        
        # Get the predicted player and their stats
        player, confidence = list(predictions.items())[0]
        
        # Get current lineup positions
        positions = predictor.position_generator.get_positions(home_players)
        num_guards = positions.count('G')
        num_forwards = positions.count('F')
        num_centers = positions.count('C')
        
        # Get predicted player's position
        predicted_pos = predictor.position_generator.get_player_position(player)
        
        # Generate position-based reasoning
        position_need = ""
        if num_guards < 2:
            position_need = "need for backcourt presence"
        elif num_centers == 0:
            position_need = "need for a center"
        elif num_forwards < 2:
            position_need = "need for frontcourt strength"
        
        # Calculate chemistry score
        chemistry_score = predictor.chemistry_analyzer.calculate_chemistry(home_players)
        chemistry_rating = "excellent" if chemistry_score > 0.8 else \
                         "good" if chemistry_score > 0.6 else \
                         "average" if chemistry_score > 0.4 else "below average"
        
        # Generate game situation context
        game_context = ""
        if game_time < 6:
            game_context = "critical late-game situation"
        elif game_time < 24:
            game_context = "key rotation period"
        else:
            game_context = "standard lineup adjustment"
        
        reasoning = (
            f"Selected Player\n"
            f"Player Name: {player} ({predicted_pos}) with {confidence:.1%} confidence\n\n"
            f"Current Formation\n"
            f"Lineup Structure: {num_guards}G-{num_forwards}F-{num_centers}C\n"
            f"Position Need: {position_need if position_need else 'Balanced lineup'}\n\n"
            f"Team Chemistry\n"
            f"Chemistry Rating: {chemistry_rating}\n"
            f"Chemistry Score: {chemistry_score:.2f} with current lineup\n\n"
            f"Game Context\n"
            f"Time Situation: {game_context}\n"
            f"Game Time: {game_time} minutes\n\n"
            f"Matchup Information\n"
            f"Teams: {home_team} vs {away_team}\n"
            f"Season: {season}\n\n"
            f"Historical Data\n"
            f"Analysis: Based on past successful lineups and player combinations"
        )
        
        return jsonify({
            'predictions': [{
                'player': player,
                'confidence': f"{confidence:.1%}"
            }],
            'reasoning': reasoning
        })
        
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'predictions': [],
            'reasoning': f'Error during prediction: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 