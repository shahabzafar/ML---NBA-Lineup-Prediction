import pandas as pd
import os
import random
from typing import List, Dict

class PlayerAvailability:
    def __init__(self):
        pass  # No initialization needed for simplified version
    
    def get_available_players(self, players, season):
        """
        Simplified method that returns all players as available
        for faster testing
        """
        # Return all players as available
        return players

    def create_mock_injury_data(self):
        """Create mock injury data for demonstration"""
        # In a real system, this would load actual injury data
        # For now, randomly mark ~5% of players as injured per season
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, 'data')
        
        # Load player names from first season as sample
        try:
            sample_data = pd.read_csv(os.path.join(data_dir, 'matchups-2007.csv'))
            all_players = set()
            
            # Collect all unique player names from both home and away teams
            for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                       'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
                all_players.update(sample_data[col].unique())
            
            all_players = list(all_players)
            
            # Generate mock injury data for each season in our dataset
            for season in range(2007, 2016):
                # Randomly select ~5% of players as injured in each season
                num_injured = max(5, int(len(all_players) * 0.05))
                injured_players = random.sample(all_players, num_injured)
                
                # Store injured players by season
                self.injured_players[str(season)] = injured_players
                
                print(f"Season {season}: {len(injured_players)} players marked as injured")
                
        except Exception as e:
            print(f"Error creating mock injury data: {e}")
            # Initialize empty injury lists if there's an error
            for season in range(2007, 2016):
                self.injured_players[str(season)] = []
    
    def is_player_available(self, player: str, season: str) -> bool:
        """Check if a player is available (not injured)"""
        # If season not in our records, assume all players available
        if season not in self.injured_players:
            return True
        
        # Return True if player is not in the injured list for this season
        return player not in self.injured_players[season]
    
    def get_injured_players(self, season: str) -> List[str]:
        """Get all injured players for a season"""
        # Return the list of injured players for the specified season
        return self.injured_players.get(season, []) 