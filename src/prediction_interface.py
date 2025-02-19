import pandas as pd
from typing import List, Dict
import numpy as np

class LineupPredictor:
    def __init__(self, model, position_generator, chemistry_analyzer, time_analyzer, data=None):
        self.model = model
        self.position_generator = position_generator
        self.chemistry_analyzer = chemistry_analyzer
        self.time_analyzer = time_analyzer
        self.data = data  # Store the data during initialization
    
    def predict_fifth_player(self, 
                           season: str,
                           home_team: str,
                           away_team: str,
                           current_players: List[str],
                           away_players: List[str],
                           game_time: float) -> Dict[str, float]:
        """
        Predict probabilities for the fifth player
        Returns dict of player_id: probability
        """
        try:
            # Get valid players for this team and season
            valid_players = self.get_team_players(home_team, season)
            
            # Remove current players from valid players
            valid_players = [p for p in valid_players if p not in current_players]
            
            if not valid_players:
                return {}
            
            # Create features for prediction
            data = {
                'season': [int(season)],
                'home_team_encoded': [self.model.team_encoder.transform([home_team])[0]],
                'away_team_encoded': [self.model.team_encoder.transform([away_team])[0]],
                'starting_min': [game_time],
                'home_0_encoded': [self.model.player_encoder.transform([current_players[0]])[0]],
                'home_1_encoded': [self.model.player_encoder.transform([current_players[1]])[0]],
                'home_2_encoded': [self.model.player_encoder.transform([current_players[2]])[0]],
                'home_3_encoded': [self.model.player_encoder.transform([current_players[3]])[0]],
                'away_0_encoded': [self.model.player_encoder.transform([away_players[0]])[0]],
                'away_1_encoded': [self.model.player_encoder.transform([away_players[1]])[0]],
                'away_2_encoded': [self.model.player_encoder.transform([away_players[2]])[0]],
                'away_3_encoded': [self.model.player_encoder.transform([away_players[3]])[0]],
                'away_4_encoded': [self.model.player_encoder.transform([away_players[4]])[0]],
            }
            
            df = pd.DataFrame(data)
            
            # Add required features
            positions = self.position_generator.get_positions(current_players)
            df['num_centers'] = [positions.count('C')]
            df['num_forwards'] = [positions.count('F')]
            df['num_guards'] = [positions.count('G')]
            df['chemistry_score'] = [self.chemistry_analyzer.calculate_chemistry(current_players)]
            
            # Get predictions
            model_probs = self.model.predict_proba(df)[0]
            
            # Get only the best prediction
            player_probs = {}
            for player in valid_players:
                try:
                    player_id = self.model.player_encoder.transform([player])[0]
                    idx = list(self.model.model.classes_).index(player_id)
                    player_probs[player] = model_probs[idx]
                except:
                    continue
            
            # Return only the highest probability player
            if player_probs:
                best_player = max(player_probs.items(), key=lambda x: x[1])
                return {best_player[0]: best_player[1]}
            return {}
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise e
    
    def get_team_players(self, team: str, season: str) -> List[str]:
        """Get all players who played for a given team in a specific season"""
        if self.data is None:
            return []  # Return empty list if no data available
            
        season_data = self.data[self.data['season'] == int(season)]
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
        
        return sorted(list(players)) 