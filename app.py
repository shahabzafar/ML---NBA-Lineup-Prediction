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
                'reasoning': 'No valid fifth player found.'
            })
        
        # Get the predicted player and confidence
        player, confidence = list(predictions.items())[0]
        
        # Ensure confidence is capped at 100%
        confidence = min(100.0, confidence)
        
        # Get current lineup positions and structure
        current_lineup_positions = [predictor.position_generator.get_player_position(p) for p in home_players]
        num_guards = current_lineup_positions.count('G')
        num_forwards = current_lineup_positions.count('F')
        num_centers = current_lineup_positions.count('C')
        predicted_pos = predictor.position_generator.get_player_position(player)
        
        # Calculate lineup structure
        lineup_balance = (
            "Guard-heavy" if num_guards > 2 else
            "Forward-heavy" if num_forwards > 2 else
            "Center-heavy" if num_centers > 1 else
            "Balanced" if num_guards == 2 and num_forwards == 2 else
            "Mixed"
        )
        lineup_structure = f"{num_guards}G-{num_forwards}F-{num_centers}C ({lineup_balance})"
        
        # Dynamic position need based on current lineup
        if num_guards == 0:
            position_need = "Critical need for backcourt presence - no guards in lineup"
        elif num_centers == 0 and num_forwards < 2:
            position_need = "Critical need for frontcourt presence - lacking size and interior presence"
        elif num_guards == 1:
            position_need = "Need for additional backcourt playmaker/shooter"
        elif num_forwards == 0:
            position_need = "Need for wing players - lacking forward presence"
        elif num_centers == 0:
            position_need = "Need for interior presence - no true center"
        else:
            position_need = "Balanced lineup - seeking best available player"

        # Calculate chemistry scores
        new_lineup = home_players + [player]
        chemistry_score = predictor.chemistry_analyzer.calculate_chemistry(new_lineup)
        current_chemistry = predictor.chemistry_analyzer.calculate_chemistry(home_players)
        
        # Dynamic chemistry impact assessment
        chemistry_diff = chemistry_score - current_chemistry
        chemistry_impact = (
            "Significant improvement" if chemistry_diff > 0.2 else
            "Moderate improvement" if chemistry_diff > 0.1 else
            "Slight improvement" if chemistry_diff > 0 else
            "Slight decrease" if chemistry_diff < 0 else
            "No significant change"
        )
        
        # Format chemistry score with proper parentheses
        chemistry_display = f"{chemistry_score:.2f} with predicted lineup (Current: {current_chemistry:.2f})"
        
        # Confidence assessment
        confidence_assessment = (
            "Very High" if confidence > 75 else
            "High" if confidence > 50 else
            "Moderate" if confidence > 25 else
            "Low" if confidence > 10 else
            "Very Low"
        )
        
        # Format analysis text to prevent wrapping
        analysis_text = f"Based on {len(predictor.data[(predictor.data['season'] == int(season)) & ((predictor.data['home_team'] == home_team) | (predictor.data['away_team'] == home_team))])} historical lineup combinations for {home_team} in {season}"
        
        # Determine game context based on game time
        game_context = (
            "Early game rotation" if game_time < 12 else
            "Mid-first half adjustment" if game_time < 24 else
            "Late first half strategy" if game_time < 36 else
            "Third quarter adjustment" if game_time < 48 else
            "Critical end-game situation"
        )
        
        # Generate reasoning text with fixed formatting
        reasoning = (
            f"Player Name: {player} ({predicted_pos}) with {confidence:.1f}% confidence\n"
            f"Confidence Level: {confidence_assessment} - {confidence:.1f}% probability based on historical data\n"
            f"Lineup Structure: {lineup_structure}\n"
            f"Position Need: {position_need}\n"
            f"Team Chemistry: {chemistry_impact}\n"
            f"Chemistry Score: {chemistry_display}\n"
            f"Game Context: {game_context}\n"
            f"Game Time: {game_time} minutes\n"
            f"Matchup: {home_team} vs {away_team}\n"
            f"Season: {season}\n"
            f"Analysis: {analysis_text}"
        )
        
        return jsonify({
            'predictions': [{
                'player': player,
                'confidence': f"{confidence:.1f}%"
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