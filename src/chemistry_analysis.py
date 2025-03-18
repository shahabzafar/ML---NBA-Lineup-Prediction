from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

class ChemistryAnalyzer:
    def __init__(self):
        # Initialize with historical player chemistry data
        self.player_chemistry = self.load_player_chemistry()
        # Dictionary mapping position abbreviations to full position names
        self.position_map = {
            'G': 'Guard',
            'F': 'Forward',
            'C': 'Center'
        }
    
    def load_player_chemistry(self) -> Dict[tuple, float]:
        """
        Load historical player chemistry data
        Returns a dictionary of (player1, player2): chemistry_score
        """
        # Initialize with some realistic chemistry data
        chemistry_data = {}
        
        # Define some base chemistry values based on common lineup patterns
        # These values represent average chemistry between different position combinations
        base_values = {
            ('G', 'G'): 0.75,  # Guard-Guard chemistry
            ('F', 'F'): 0.70,  # Forward-Forward chemistry
            ('C', 'C'): 0.60,  # Center-Center (less common)
            ('G', 'F'): 0.80,  # Guard-Forward (common combination)
            ('G', 'C'): 0.75,  # Guard-Center (pick and roll)
            ('F', 'C'): 0.75   # Forward-Center (frontcourt)
        }
        
        # Sample player combinations with realistic chemistry values
        # Based on well-known NBA pairings that had good on-court chemistry
        sample_data = {
            ('Chris Paul', 'Blake Griffin'): 0.85,
            ('Tony Parker', 'Tim Duncan'): 0.90,
            ('Kobe Bryant', 'Pau Gasol'): 0.88,
            ('Steve Nash', 'Amar\'e Stoudemire'): 0.87,
            ('Dwyane Wade', 'LeBron James'): 0.92,
            ('Russell Westbrook', 'Kevin Durant'): 0.89,
            ('Rajon Rondo', 'Kevin Garnett'): 0.86,
            ('Jason Kidd', 'Dirk Nowitzki'): 0.88
        }
        
        chemistry_data.update(sample_data)
        return chemistry_data
    
    def get_lineup_structure(self, players: List[str]) -> str:
        """Get the lineup structure description"""
        # Extract position information for each player in the lineup
        positions = [self.get_player_position(p) for p in players]
        # Count the number of players at each position
        guards = positions.count('G')
        forwards = positions.count('F')
        centers = positions.count('C')
        
        # Determine lineup balance based on position distribution
        # This classifies lineups as guard-heavy, forward-heavy, etc.
        balance = (
            "Guard-heavy" if guards > 2 else
            "Forward-heavy" if forwards > 2 else
            "Center-heavy" if centers > 1 else
            "Balanced" if guards == 2 and forwards == 2 else
            "Mixed"
        )
        
        return f"{guards}G-{forwards}F-{centers}C ({balance})"
    
    def calculate_chemistry(self, players: List[str]) -> float:
        """Calculate chemistry score for a lineup"""
        if not players:
            return 0.0
        
        # Get current lineup positions to analyze position balance
        positions = [self.get_player_position(p) for p in players]
        guards = positions.count('G')
        forwards = positions.count('F')
        centers = positions.count('C')
        
        # Start with neutral chemistry as baseline
        base_chemistry = 0.5
        
        # Calculate pair chemistry for all player pairs in the lineup
        # This captures the synergy between each pair of players
        pair_scores = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                pair_score = self.get_pair_chemistry(players[i], players[j])
                pair_scores.append(pair_score)
        
        # Average the pair scores if any exist
        if pair_scores:
            base_chemistry = sum(pair_scores) / len(pair_scores)
        
        # Apply position-based adjustments to the chemistry score
        # Balanced lineups receive a boost, unbalanced lineups a penalty
        if 1 <= guards <= 2 and 2 <= forwards <= 3 and 0 <= centers <= 1:
            base_chemistry *= 1.1  # Reduced boost for balanced lineup
        elif guards == 0 or forwards == 0:
            base_chemistry *= 0.9  # Reduced penalty for unbalanced lineup
        
        # Ensure chemistry score doesn't exceed 1.0
        return min(1.0, base_chemistry)
    
    def get_pair_chemistry(self, player1: str, player2: str) -> float:
        """
        Get chemistry score for a pair of players
        Returns a value between 0 and 1
        """
        # Try both orderings of the pair since chemistry is symmetric
        pair = (player1, player2)
        reverse_pair = (player2, player1)
        
        # Check stored chemistry data for existing pairs
        if pair in self.player_chemistry:
            return self.player_chemistry[pair]
        if reverse_pair in self.player_chemistry:
            return self.player_chemistry[reverse_pair]
        
        # If not found, generate a realistic chemistry score based on positions
        # This handles cases where we don't have historical data
        pos1 = self.get_player_position(player1)
        pos2 = self.get_player_position(player2)
        
        # Assign base chemistry value based on position combination patterns
        if pos1 == pos2:
            base_chemistry = 0.70  # Same position
        elif (pos1 == 'G' and pos2 == 'F') or (pos1 == 'F' and pos2 == 'G'):
            base_chemistry = 0.75  # Guard-Forward combo
        elif (pos1 == 'G' and pos2 == 'C') or (pos1 == 'C' and pos2 == 'G'):
            base_chemistry = 0.72  # Guard-Center combo
        else:
            base_chemistry = 0.68  # Forward-Center combo
        
        # Add some random variation to avoid deterministic values
        # This simulates the unpredictability of real player chemistry
        chemistry = base_chemistry + np.random.uniform(-0.05, 0.05)
        
        # Cache the calculated chemistry for future use
        self.player_chemistry[pair] = chemistry
        
        return chemistry
    
    def get_player_position(self, player: str) -> str:
        """Get position for a player (G, F, or C)"""
        # This is a simplified position determination method
        # In a production system, this would use a player position database
        if 'guard' in player.lower() or any(name in player.lower() for name in ['paul', 'parker', 'nash', 'wade', 'bryant', 'rondo']):
            return 'G'
        elif 'center' in player.lower() or any(name in player.lower() for name in ['howard', 'gasol', 'duncan', 'garnett']):
            return 'C'
        else:
            return 'F'  # Default to Forward
    
    def update_chemistry(self, player1: str, player2: str, score: float):
        """Update chemistry score for a pair of players"""
        # Sort player names to ensure consistent storage regardless of order
        pair = tuple(sorted([player1, player2]))
        # Ensure chemistry score is within valid range [0.0, 1.0]
        self.player_chemistry[pair] = min(1.0, max(0.0, score))
    
    def calculate_pair_chemistry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate chemistry scores for all player pairs in the dataset
        Args:
            df: DataFrame containing lineup data
        Returns:
            DataFrame with chemistry scores added
        """
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Add chemistry score column if it doesn't exist
        if 'chemistry_score' not in df.columns:
            df['chemistry_score'] = 0.0
        
        # Calculate chemistry for each row (lineup) in the dataset
        for idx, row in df.iterrows():
            # Extract home team players from the row
            home_players = [
                row[f'home_{i}'] for i in range(5) 
                if pd.notna(row[f'home_{i}'])
            ]
            
            # Calculate overall chemistry score for this lineup
            chemistry_score = self.calculate_chemistry(home_players)
            
            # Store the chemistry score in the dataframe
            df.at[idx, 'chemistry_score'] = chemistry_score
            
            # Calculate and store individual pair chemistry for future reference
            # This builds up our chemistry database as we process more data
            for i in range(len(home_players)):
                for j in range(i + 1, len(home_players)):
                    player1, player2 = sorted([home_players[i], home_players[j]])
                    pair = (player1, player2)
                    if pair not in self.player_chemistry:
                        self.player_chemistry[pair] = self.get_pair_chemistry(player1, player2)
        
        return df 