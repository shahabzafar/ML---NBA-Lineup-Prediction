import pandas as pd
from typing import List, Dict
import numpy as np
from src.player_availability import PlayerAvailability

class LineupPredictor:
    def __init__(self, model, position_generator, chemistry_analyzer, time_analyzer, data=None):
        self.model = model
        self.position_generator = position_generator
        self.chemistry_analyzer = chemistry_analyzer
        self.time_analyzer = time_analyzer
        self.data = data  # Store the data during initialization
        self.player_availability = PlayerAvailability()  # Add this line
    
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
            valid_players = [p for p in valid_players if p not in current_players]
            
            if not valid_players:
                return {}
            
            # Calculate position counts and chemistry
            positions = self.position_generator.get_positions(current_players)
            num_guards = positions.count('G')
            num_forwards = positions.count('F')
            num_centers = positions.count('C')
            chemistry_score = self.chemistry_analyzer.calculate_chemistry(current_players)
            
            # Get time-based probabilities
            time_bin = f"{5 * (game_time // 5)}-{5 * (game_time // 5 + 1)}"
            time_probs = self.time_analyzer.get_time_based_probabilities(time_bin)
            
            # Create features for prediction
            data = {
                'season': [int(season)],
                'home_team_encoded': [self.model.team_encoder.transform([home_team])[0]],
                'away_team_encoded': [self.model.team_encoder.transform([away_team])[0]],
                'starting_min': [game_time],
                'num_guards': [num_guards],
                'num_forwards': [num_forwards],
                'num_centers': [num_centers],
                'chemistry_score': [chemistry_score]
            }
            
            # Add encoded player columns
            for i, player in enumerate(current_players):
                data[f'home_{i}_encoded'] = [self.model.player_encoder.transform([player])[0]]
            for i, player in enumerate(away_players):
                data[f'away_{i}_encoded'] = [self.model.player_encoder.transform([player])[0]]
            
            df = pd.DataFrame(data)
            
            try:
                # Get model probabilities
                model_probs = self.model.predict_proba(df)[0]
                
                # Process each valid player
                player_probs = {}
                total_prob = 0
                
                for player in valid_players:
                    try:
                        # Get player index in model classes
                        player_encoded = self.model.player_encoder.transform([player])[0]
                        if player_encoded in self.model.model.classes_:
                            idx = list(self.model.model.classes_).index(player_encoded)
                            base_prob = model_probs[idx]
                            
                            # Adjust probability based on time patterns
                            time_factor = time_probs.get(player, 0.5)
                            
                            # Calculate chemistry impact
                            test_lineup = current_players + [player]
                            new_chemistry = self.chemistry_analyzer.calculate_chemistry(test_lineup)
                            chemistry_impact = new_chemistry - chemistry_score
                            
                            # Combine factors
                            final_prob = (
                                base_prob * 0.4 +
                                time_factor * 0.3 +
                                max(0, min(1, (chemistry_impact + 1) * 0.3))
                            )
                            
                            player_probs[player] = final_prob
                            total_prob += final_prob
                            
                    except Exception as e:
                        print(f"Error processing player {player}: {str(e)}")
                        continue
                
                # Normalize probabilities
                if player_probs and total_prob > 0:
                    normalized_probs = {
                        player: (prob / total_prob) * 100 
                        for player, prob in player_probs.items()
                    }
                    
                    # Return highest probability player
                    best_player = max(normalized_probs.items(), key=lambda x: x[1])
                    return {best_player[0]: min(100, best_player[1])}  # Cap at 100%
                
                return {}
                
            except Exception as e:
                print(f"Error in probability calculation: {str(e)}")
                return {}
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {}
    
    def get_team_players(self, team: str, season: str) -> List[str]:
        """Get all available players who played for a given team in a specific season"""
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
        
        all_players = sorted(list(players))
        
        # Filter out injured players
        available_players = self.player_availability.get_available_players(all_players, season)
        
        return available_players 