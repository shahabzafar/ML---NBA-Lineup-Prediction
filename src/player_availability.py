import pandas as pd
import os
import random
from typing import List, Dict

class PlayerAvailability:
    def __init__(self):
        """Initialize player availability tracker"""
        self.injured_players = {}
        self.create_mock_injury_data()
    
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
            
            for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                       'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
                all_players.update(sample_data[col].unique())
            
            all_players = list(all_players)
            
            # Mock injury data for each season
            for season in range(2007, 2016):
                # Randomly select ~5% of players as injured
                num_injured = max(5, int(len(all_players) * 0.05))
                injured_players = random.sample(all_players, num_injured)
                
                self.injured_players[str(season)] = injured_players
                
                print(f"Season {season}: {len(injured_players)} players marked as injured")
                
        except Exception as e:
            print(f"Error creating mock injury data: {e}")
            # Empty data if there's an error
            for season in range(2007, 2016):
                self.injured_players[str(season)] = []
    
    def is_player_available(self, player: str, season: str) -> bool:
        """Check if a player is available (not injured)"""
        if season not in self.injured_players:
            return True
        
        return player not in self.injured_players[season]
    
    def get_available_players(self, players: List[str], season: str) -> List[str]:
        """Filter a list of players to only those available"""
        return [p for p in players if self.is_player_available(p, season)]
    
    def get_injured_players(self, season: str) -> List[str]:
        """Get all injured players for a season"""
        return self.injured_players.get(season, []) 