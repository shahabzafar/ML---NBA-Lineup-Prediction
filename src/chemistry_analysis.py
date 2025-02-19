from typing import List, Dict
import pandas as pd
import numpy as np

class ChemistryAnalyzer:
    def __init__(self):
        # Initialize with historical player chemistry data
        self.player_chemistry = self.load_player_chemistry()
    
    def load_player_chemistry(self) -> Dict[tuple, float]:
        """
        Load historical player chemistry data
        Returns a dictionary of (player1, player2): chemistry_score
        """
        # This should load from your actual data source
        # For now, creating a sample dictionary
        chemistry_data = {
            # Sample data for the players in your example
            ('Rajon Rondo', 'J.J. Barea'): 0.75,
            ('Rajon Rondo', 'Ricky Ledo'): 0.65,
            ('Rajon Rondo', 'Greg Smith'): 0.70,
            ('J.J. Barea', 'Ricky Ledo'): 0.80,
            ('J.J. Barea', 'Greg Smith'): 0.72,
            ('Ricky Ledo', 'Greg Smith'): 0.68,
            # Add more player pairs as needed
        }
        return chemistry_data
    
    def calculate_pair_chemistry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate chemistry scores for all player pairs in the DataFrame
        """
        # Add chemistry_score column if it doesn't exist
        if 'chemistry_score' not in df.columns:
            df['chemistry_score'] = df.apply(lambda row: self.calculate_lineup_chemistry(row), axis=1)
        return df
    
    def calculate_lineup_chemistry(self, row) -> float:
        """
        Calculate chemistry score for a lineup based on row data
        """
        # Get all players from the row
        players = []
        for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4']:
            if col in row and pd.notna(row[col]):
                players.append(row[col])
        
        return self.calculate_chemistry(players)
    
    def calculate_chemistry(self, players: List[str]) -> float:
        """
        Calculate overall chemistry score for a group of players
        """
        if len(players) < 2:
            return 0.0
            
        # Calculate average chemistry between all pairs
        chemistry_scores = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                player1, player2 = sorted([players[i], players[j]])  # Sort to ensure consistent lookup
                pair_chemistry = self.get_pair_chemistry(player1, player2)
                chemistry_scores.append(pair_chemistry)
        
        # Return average chemistry score
        return np.mean(chemistry_scores) if chemistry_scores else 0.5
    
    def get_pair_chemistry(self, player1: str, player2: str) -> float:
        """
        Get chemistry score for a pair of players
        Returns a value between 0 and 1
        """
        # Try both orderings of the pair
        pair = (player1, player2)
        reverse_pair = (player2, player1)
        
        # Return stored chemistry or default to 0.5 if not found
        return self.player_chemistry.get(pair, 
               self.player_chemistry.get(reverse_pair, 0.5))
    
    def update_chemistry(self, player1: str, player2: str, score: float):
        """
        Update chemistry score for a pair of players
        """
        pair = tuple(sorted([player1, player2]))  # Sort to ensure consistent storage
        self.player_chemistry[pair] = score 