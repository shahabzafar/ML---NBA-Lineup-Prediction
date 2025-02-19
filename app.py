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
    # Filter data for the selected season
    season_data = data[data['season'] == int(season)]
    home_teams = sorted(season_data['home_team'].unique())
    away_teams = sorted(season_data['away_team'].unique())
    teams = sorted(list(set(home_teams + away_teams)))
    return jsonify(teams)

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
        
        # Generate meaningful reasoning
        positions = predictor.position_generator.get_positions(home_players)
        current_formation = (
            f"{positions.count('G')} Guards, "
            f"{positions.count('F')} Forwards, "
            f"{positions.count('C')} Centers"
        )
        
        reasoning = (
            f"Selected {player} (confidence: {confidence:.1%}) based on:\n"
            f"• Current formation: {current_formation}\n"
            f"• Team chemistry score: {predictor.chemistry_analyzer.calculate_chemistry(home_players):.2f}\n"
            f"• Matchup against {away_team}\n"
            f"• {season} season performance data"
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