from flask import Flask, render_template, request, jsonify, redirect, url_for
from src.position_features import PositionFeatureGenerator
from src.chemistry_analysis import ChemistryAnalyzer
from src.time_analysis import TimeAnalyzer
from src.prediction_interface import LineupPredictor as PredictorInterface
from src.model import LineupPredictor as BaseModel
from src.data_preprocessing import DataPreprocessor
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid display issues
import matplotlib.pyplot as plt
import json
import sys
import glob
import subprocess
from datetime import datetime

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
        # Extract data from request
        data = request.json
        season = data.get('season')
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        home_players = data.get('home_players', [])
        away_players = data.get('away_players', [])
        game_time = float(data.get('game_time', 15.0))
        
        # Validate required inputs
        if not all([season, home_team, home_players, len(home_players) > 3]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Validate lineup has exactly 4 players
        if len(home_players) != 4:
            return jsonify({'error': 'Home lineup must contain exactly 4 players'}), 400
            
        print(f"Predicting fifth player for {home_team} - Current players: {home_players}")
        
        # Make prediction
        predictions = predictor.predict_fifth_player(
            season=season,
            home_team=home_team,
            away_team=away_team,
            current_players=home_players,
            away_players=away_players,
            game_time=game_time
        )
        
        if not predictions:
            return jsonify({'error': 'No valid predictions could be made'}), 404
            
        # Get top prediction
        player = list(predictions.keys())[0]
        confidence = list(predictions.values())[0]
        
        # Get position info
        positions = predictor.position_generator.get_positions(home_players)
        num_guards = positions.count('G')
        num_forwards = positions.count('F')
        num_centers = positions.count('C')
        predicted_pos = predictor.position_generator.get_player_position(player)
        
        # Generate lineup structure description
        lineup_structure = f"{num_guards}G-{num_forwards}F-{num_centers}C lineup"
        
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

@app.route('/evaluation')
def evaluation():
    """Display evaluation results"""
    # Set up paths to evaluation output files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(current_dir, 'evaluation')
    summary_path = os.path.join(eval_dir, 'evaluation_summary.csv')
    
    # Default metrics
    metrics = {
        'overall_accuracy': 0,
        'total_matches': 0,
        'correct_predictions': 0,
        'average_matches': 0,
        'matches_per_year': {},
    }
    
    # Check if evaluation results exist
    if os.path.exists(summary_path):
        # Load summary metrics
        summary_df = pd.read_csv(summary_path)
        metrics_dict = dict(zip(summary_df['metric'], summary_df['value']))
        
        metrics['overall_accuracy'] = metrics_dict.get('accuracy', 0)
        metrics['total_matches'] = int(metrics_dict.get('total_predictions', 0))
        metrics['correct_predictions'] = int(metrics_dict.get('correct_predictions', 0))
        
        # Load matches per year data
        matches_per_year_path = os.path.join(eval_dir, 'matches_per_year.json')
        if os.path.exists(matches_per_year_path):
            with open(matches_per_year_path, 'r') as f:
                metrics['matches_per_year'] = json.load(f)
            
            if metrics['matches_per_year']:
                metrics['average_matches'] = sum(metrics['matches_per_year'].values()) / len(metrics['matches_per_year'])
                
                # Format data for charts
                matches_years = list(metrics['matches_per_year'].keys())
                matches_counts = [metrics['matches_per_year'][year] for year in matches_years]
                
                metrics['matches_years'] = matches_years
                metrics['matches_counts'] = matches_counts
        
        # Load accuracy by season data
        accuracy_by_season_path = os.path.join(eval_dir, 'accuracy_by_season.json')
        if os.path.exists(accuracy_by_season_path):
            with open(accuracy_by_season_path, 'r') as f:
                season_accuracy = json.load(f)
            
            if season_accuracy:
                # Format data for charts
                accuracy_years = list(season_accuracy.keys())
                accuracy_values = [season_accuracy[year] for year in accuracy_years]
                
                metrics['accuracy_years'] = accuracy_years
                metrics['accuracy_values'] = accuracy_values
    
    # Load season test results
    season_test_data = load_season_test_results()
    
    # Combine all metrics
    template_data = {**metrics, **season_test_data}
    
    return render_template('evaluation.html', **template_data)

@app.route('/get_team_players')
def get_team_players():
    try:
        team = request.args.get('team', '')
        season = request.args.get('season', '2007')
        
        if not team:
            return jsonify({'error': 'Team name is required'}), 400
            
        # Get available players
        available_players = predictor.get_team_players(team, season)
        
        # Get injured players (for display)
        all_players = []
        team_data = data[data['season'] == int(season)]
        team_rows = team_data[(team_data['home_team'] == team) | (team_data['away_team'] == team)]
        
        for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 
                   'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
            if col in team_rows.columns:
                all_players.extend(team_rows[col].unique())
        
        all_players = sorted(list(set([p for p in all_players if isinstance(p, str)])))
        injured_players = [p for p in all_players if p not in available_players]
        
        return jsonify({
            'players': available_players,
            'injured_players': injured_players
        })
    except Exception as e:
        print(f"Error getting team players: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/run_evaluation')
def run_evaluation_route():
    try:
        subprocess.run(["python", "src/evaluate_test_data.py"], check=True)
        return redirect(url_for('evaluation'))
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/run_season_test')
def run_season_test():
    """Run the model evaluation script and redirect to evaluation page"""
    try:
        # Execute the season test script
        subprocess.Popen(["python", "season_test.py"])
        return "Model evaluation started. Please check the terminal for prompts and inputs. Refresh the evaluation page when complete."
    except Exception as e:
        return f"Error running model evaluation: {str(e)}"

def load_season_test_results():
    """Load results from seasonal testing"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all season result files
    result_files = glob.glob(os.path.join(current_dir, "season_*_results.csv"))
    all_seasons_file = os.path.join(current_dir, "all_seasons_results.csv")
    
    season_results = {}
    all_seasons_data = None
    position_accuracy = {}
    has_position_data = False
    
    for file_path in result_files:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            season = filename.replace("season_", "").replace("_results.csv", "")
            season_df = pd.read_csv(file_path)
            
            # Calculate accuracy
            accuracy = season_df['is_correct'].mean() * 100
            correct_count = season_df['is_correct'].sum()
            total_count = len(season_df)
            
            # Store metrics for this season
            season_results[season] = {
                'accuracy': accuracy,
                'total_predictions': total_count,
                'correct_predictions': correct_count
            }
            
            # Check for position data
            if 'true_position' in season_df.columns and 'pred_position' in season_df.columns:
                has_position_data = True
    
    # Load all seasons combined data if available
    if os.path.exists(all_seasons_file):
        all_seasons_df = pd.read_csv(all_seasons_file)
        accuracy = all_seasons_df['is_correct'].mean() * 100
        correct_count = all_seasons_df['is_correct'].sum()
        total_count = len(all_seasons_df)
        
        season_results['all'] = {
            'accuracy': accuracy,
            'total_predictions': total_count,
            'correct_predictions': correct_count
        }
        
        # Calculate position accuracy if available
        if 'true_position' in all_seasons_df.columns and 'pred_position' in all_seasons_df.columns:
            has_position_data = True
            
            # Position-level accuracy
            for pos in ['G', 'F', 'C']:
                pos_df = all_seasons_df[all_seasons_df['true_position'] == pos]
                if len(pos_df) > 0:
                    position_accuracy[pos] = pos_df['is_correct'].mean() * 100
            
            # Position match accuracy
            position_match_accuracy = (all_seasons_df['true_position'] == all_seasons_df['pred_position']).mean() * 100
    
    season_test_data = {
        'season_metrics': season_results,
        'position_accuracy': position_accuracy,
        'has_position_data': has_position_data
    }
    
    if 'all' in season_results:
        season_test_data['all_seasons_accuracy'] = season_results['all']['accuracy']
        season_test_data['all_seasons_total'] = season_results['all']['total_predictions']
        season_test_data['position_match_accuracy'] = position_match_accuracy if has_position_data else 0
    
    # Prepare data for charts
    if season_results:
        seasons = [s for s in season_results.keys() if s != 'all']
        season_test_data['season_count'] = len(seasons)
        
        # Format data for chart
        season_test_seasons = []
        season_test_accuracies = []
        
        for s in sorted(seasons, key=lambda x: int(x) if x.isdigit() else float('inf')):
            season_test_seasons.append(s)
            season_test_accuracies.append(season_results[s]['accuracy'])
            
        # Add all seasons combined at the end if available
        if 'all' in season_results:
            season_test_seasons.append('All Combined')
            season_test_accuracies.append(season_results['all']['accuracy'])
            
        season_test_data['season_test_seasons'] = season_test_seasons
        season_test_data['season_test_accuracies'] = season_test_accuracies
    
    return season_test_data

if __name__ == '__main__':
    app.run(debug=True) 