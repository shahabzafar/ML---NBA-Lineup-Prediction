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
        self.player_availability = PlayerAvailability()  # Track player injuries/availability
    
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
            # Input validation and normalization
            if len(current_players) != 4:
                raise ValueError(f"Need exactly 4 current players, got {len(current_players)}")
            
            # Convert season to int for consistency
            try:
                season_int = int(season)
            except ValueError:
                # Default to middle of range if invalid
                season_int = 2011
            
            # Get valid players for this team and season
            valid_players = self.get_team_players(home_team, season)
            valid_players = [p for p in valid_players if p not in current_players]
            
            if not valid_players:
                return {}
            
            # Calculate more detailed position-based features
            positions = self.position_generator.get_positions(current_players)
            num_guards = positions.count('G')
            num_forwards = positions.count('F')
            num_centers = positions.count('C')
            
            # Calculate chemistry score of current lineup
            chemistry_score = self.chemistry_analyzer.calculate_chemistry(current_players)
            
            # Get time-based probabilities
            time_bin = f"{5 * (game_time // 5)}-{5 * (game_time // 5 + 1)}"
            time_probs = self.time_analyzer.get_time_based_probabilities(time_bin)
            
            # Create lineup balance score (0-1)
            lineup_balance = 0.0
            if (num_guards in [1, 2]) and (num_forwards in [2, 3]) and (num_centers in [0, 1]):
                lineup_balance = 1.0  # Balanced lineup
            elif num_guards == 0 or num_forwards == 0:
                lineup_balance = 0.2  # Very unbalanced
            elif num_guards + num_forwards + num_centers != 4:
                lineup_balance = 0.5  # Something odd with the positions
            else:
                lineup_balance = 0.8  # Somewhat balanced
            
            # Create features for prediction with more detailed information
            data = {
                'season': [season_int],
                'home_team_encoded': [self.model.team_encoder.transform([home_team])[0]],
                'away_team_encoded': [self.model.team_encoder.transform([away_team])[0]],
                'starting_min': [game_time],
                'num_guards': [num_guards],
                'num_forwards': [num_forwards],
                'num_centers': [num_centers],
                'chemistry_score': [chemistry_score],
                'lineup_balance': [lineup_balance]
            }
            
            # Add encoded player columns
            for i, player in enumerate(current_players):
                data[f'home_{i}_encoded'] = [self.model.player_encoder.transform([player])[0]]
            
            # Add encoded away team players - pad with -1 if not enough
            away_count = min(5, len(away_players))
            for i in range(away_count):
                data[f'away_{i}_encoded'] = [self.model.player_encoder.transform([away_players[i]])[0]]
            
            # Add dummy values for missing away players
            for i in range(away_count, 5):
                data[f'away_{i}_encoded'] = [-1]  # Padding value
            
            # Convert to DataFrame for model prediction
            df = pd.DataFrame(data)
            
            try:
                # Get model probabilities 
                # If some features are missing but are required, just use their median values
                required_features = set(self.model.feature_names)
                missing_features = required_features - set(df.columns)
                if missing_features:
                    # Add median values for missing features from training data
                    for feature in missing_features:
                        # If feature isn't in the model but required, set to median or 0
                        df[feature] = 0
                
                # Ensure all required columns are present and in the correct order
                df = df[self.model.feature_names]
                
                # Get model probabilities
                model_probs = self.model.predict_fifth_player(df)
                
                # If no prediction, return empty dict
                if not model_probs:
                    return {}
                
                # Process each valid player
                player_probs = {}
                total_prob = 0
                
                # Calculate position needs based on current lineup
                def get_position_need():
                    if num_guards == 0:
                        return {'G': 1.0, 'F': 0.4, 'C': 0.2}  # Critical need for guard
                    elif num_centers == 0 and num_forwards < 2:
                        return {'G': 0.2, 'F': 0.6, 'C': 0.9}  # Critical need for frontcourt
                    elif num_guards == 1:
                        return {'G': 0.8, 'F': 0.5, 'C': 0.3}  # Need another guard
                    elif num_forwards == 0:
                        return {'G': 0.3, 'F': 0.9, 'C': 0.4}  # Need forwards
                    elif num_forwards == 1:
                        return {'G': 0.4, 'F': 0.8, 'C': 0.5}  # Need more forwards
                    elif num_centers == 0:
                        return {'G': 0.3, 'F': 0.5, 'C': 0.8}  # Need a center
                    else:
                        return {'G': 0.6, 'F': 0.6, 'C': 0.6}  # Balanced need
                
                position_need = get_position_need()
                
                # Get predicted players from model
                for player, base_prob in model_probs.items():
                    if player not in valid_players:
                        continue
                        
                    # Get player position and chemistry impact
                    player_pos = self.position_generator.get_player_position(player)
                    position_factor = position_need.get(player_pos, 0.5)
                    
                    # Calculate chemistry impact if this player is added
                    test_lineup = current_players + [player]
                    new_chemistry = self.chemistry_analyzer.calculate_chemistry(test_lineup)
                    chemistry_impact = new_chemistry - chemistry_score
                    
                    # Get time-based probability adjustment
                    time_factor = time_probs.get(player, 0.5)
                    
                    # Combine all factors with appropriate weights
                    # Higher weights for model probability and chemistry impact
                    # Also consider position needs and time patterns
                    # The weights sum to 1.0 to maintain proper scaling
                    final_prob = (
                        (base_prob / 100) * 0.60 +       # Model prediction (primary)
                        time_factor * 0.15 +             # Time-based patterns
                        position_factor * 0.15 +         # Position needs  
                        max(0, min(1, (chemistry_impact + 0.5) * 0.5)) * 0.10  # Chemistry impact
                    )
                    
                    player_probs[player] = final_prob
                    total_prob += final_prob
                
                # Normalize probabilities and convert to percentages
                if player_probs and total_prob > 0:
                    normalized_probs = {
                        player: (prob / total_prob) * 100 
                        for player, prob in player_probs.items()
                    }
                    
                    # Sort by probability and return top result
                    sorted_predictions = dict(sorted(normalized_probs.items(), 
                                                  key=lambda x: x[1], 
                                                  reverse=True))
                    
                    # Return top 1 result
                    top_player = list(sorted_predictions.keys())[0]
                    top_prob = sorted_predictions[top_player]
                    
                    return {top_player: min(100, top_prob)}
                
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
        
        # Convert season to integer for filtering
        try:
            season_int = int(season)
        except ValueError:
            # Default to middle of range if invalid
            season_int = 2011
        
        # Get season data with tolerance for neighboring seasons if needed
        season_data = self.data[self.data['season'] == season_int]
        
        # If no data for exact season, try neighboring seasons
        if len(season_data) == 0:
            # Try to find closest available season
            available_seasons = sorted(self.data['season'].unique())
            if not available_seasons:
                return []
            
            # Find closest season
            closest_season = min(available_seasons, key=lambda x: abs(x - season_int))
            season_data = self.data[self.data['season'] == closest_season]
            print(f"Warning: No data for season {season_int}, using {closest_season} instead")
        
        # Filter for the specified team
        team_data = season_data[
            (season_data['home_team'] == team) | 
            (season_data['away_team'] == team)
        ]
        
        if len(team_data) == 0:
            # If no data for this team in this season, try to find data for this team in any season
            team_in_any_season = self.data[
                (self.data['home_team'] == team) | 
                (self.data['away_team'] == team)
            ]
            
            if len(team_in_any_season) > 0:
                # Use the most recent season with data for this team
                most_recent = team_in_any_season['season'].max()
                team_data = self.data[
                    (self.data['season'] == most_recent) & 
                    ((self.data['home_team'] == team) | (self.data['away_team'] == team))
                ]
                print(f"Warning: No data for team {team} in season {season_int}, using season {most_recent}")
        
        # Collect all players for this team
        players = set()
        
        # Get players from home team columns
        home_data = team_data[team_data['home_team'] == team]
        for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4']:
            if col in home_data.columns:
                players.update(home_data[col].dropna().unique())
        
        # Get players from away team columns
        away_data = team_data[team_data['away_team'] == team]
        for col in ['away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
            if col in away_data.columns:
                players.update(away_data[col].dropna().unique())
        
        # Convert to list and filter out non-string values
        all_players = [p for p in players if isinstance(p, str)]
        
        # Filter out injured players
        available_players = self.player_availability.get_available_players(all_players, str(season_int))
        
        return sorted(available_players) 