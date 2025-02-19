from typing import Dict, List
import pandas as pd
import numpy as np

class PositionFeatureGenerator:
    def __init__(self):
        # Initialize with player position data
        self.player_positions = self.load_player_positions()
    
    def load_player_positions(self) -> Dict[str, str]:
        """
        Load player positions from your dataset
        Returns a dictionary of {player_name: position}
        """
        # This should load from your actual data source
        # For now, creating a sample dictionary with known players
        positions = {
            # Sample data - add more based on your dataset
            'Markel Brown': 'G',
            'Kevin Garnett': 'F',
            'Mirza Teletovic': 'F',
            'Thaddeus Young': 'F',
            'Udonis Haslem': 'F',
            'Justin Hamilton': 'C',
            'Josh McRoberts': 'F',
            'Dwyane Wade': 'G',
            'Chris Andersen': 'C',
            # Add more players and their positions
        }
        return positions
    
    def create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create position-based features for the dataset
        """
        # Initialize position count columns if they don't exist
        if 'num_guards' not in df.columns:
            df['num_guards'] = 0
        if 'num_forwards' not in df.columns:
            df['num_forwards'] = 0
        if 'num_centers' not in df.columns:
            df['num_centers'] = 0
            
        return df
    
    def get_positions(self, players: List[str]) -> List[str]:
        """Get positions for a list of players"""
        positions = []
        for player in players:
            pos = self.get_player_position(player)
            positions.append(pos)
        return positions
    
    def get_player_position(self, player: str) -> str:
        """Get position for a single player"""
        return self.player_positions.get(player, 'F')  # Default to Forward if unknown
    
    def calculate_position_counts(self, players: List[str]) -> Dict[str, int]:
        """Calculate counts of each position in a lineup"""
        positions = self.get_positions(players)
        return {
            'num_guards': positions.count('G'),
            'num_forwards': positions.count('F'),
            'num_centers': positions.count('C')
        } 