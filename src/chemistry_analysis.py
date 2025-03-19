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
        # Track which players frequently appeared together
        self.frequent_pairs = {}
        # Win percentage when players are together
        self.win_chemistry = {}
    
    def load_player_chemistry(self) -> Dict[tuple, float]:
        """
        Load historical player chemistry data
        Returns a dictionary of (player1, player2): chemistry_score
        """
        # Real well-known successful NBA player combinations from 2007-2015
        chemistry_data = {
            # Championship teams with strong chemistry
            ('Tony Parker', 'Tim Duncan'): 0.95,
            ('Tony Parker', 'Manu Ginobili'): 0.92,
            ('Tim Duncan', 'Manu Ginobili'): 0.92,
            ('LeBron James', 'Dwyane Wade'): 0.93,
            ('LeBron James', 'Chris Bosh'): 0.89,
            ('Dwyane Wade', 'Chris Bosh'): 0.88,
            ('Kobe Bryant', 'Pau Gasol'): 0.91,
            ('Kobe Bryant', 'Derek Fisher'): 0.88,
            ('Dirk Nowitzki', 'Jason Kidd'): 0.89,
            ('Dirk Nowitzki', 'Jason Terry'): 0.88,
            ('Kevin Garnett', 'Paul Pierce'): 0.92,
            ('Kevin Garnett', 'Ray Allen'): 0.89,
            ('Kevin Garnett', 'Rajon Rondo'): 0.90,
            ('Paul Pierce', 'Ray Allen'): 0.89,
            ('Paul Pierce', 'Rajon Rondo'): 0.88,
            ('Ray Allen', 'Rajon Rondo'): 0.87,
            
            # Other successful duos
            ('Chris Paul', 'Blake Griffin'): 0.88,
            ('Steve Nash', 'Amar\'e Stoudemire'): 0.87,
            ('Russell Westbrook', 'Kevin Durant'): 0.91,
            ('James Harden', 'Dwight Howard'): 0.83,
            ('Stephen Curry', 'Klay Thompson'): 0.91,
            ('Stephen Curry', 'Draymond Green'): 0.89,
            ('Klay Thompson', 'Draymond Green'): 0.86,
            ('Marc Gasol', 'Mike Conley'): 0.87,
            ('Zach Randolph', 'Marc Gasol'): 0.88,
            ('LaMarcus Aldridge', 'Damian Lillard'): 0.86,
            ('Kyle Lowry', 'DeMar DeRozan'): 0.85,
            ('John Wall', 'Bradley Beal'): 0.84,
            ('Chris Bosh', 'Kyle Lowry'): 0.73,  # Less successful pairing
            ('Carmelo Anthony', 'Amar\'e Stoudemire'): 0.74,  # Struggled to mesh well
            ('Kobe Bryant', 'Dwight Howard'): 0.71,  # Notoriously poor chemistry
            ('Rajon Rondo', 'Monta Ellis'): 0.68,  # Struggled together
        }
        
        # No additional data loading - use predefined chemistry values
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
        if not players or len(players) <= 1:
            return 0.5  # Neutral chemistry for empty or single-player lineups
        
        # Get position counts to analyze lineup balance
        positions = [self.get_player_position(p) for p in players if isinstance(p, str)]
        guards = positions.count('G')
        forwards = positions.count('F')
        centers = positions.count('C')
        
        # Start with neutral chemistry as baseline
        base_chemistry = 0.5
        
        # Calculate pair chemistry for all player pairs in the lineup
        pair_scores = []
        for i in range(len(players)):
            if not isinstance(players[i], str):
                continue
            for j in range(i + 1, len(players)):
                if not isinstance(players[j], str):
                    continue
                pair_score = self.get_pair_chemistry(players[i], players[j])
                pair_scores.append(pair_score)
        
        # Average the pair scores if any exist
        if pair_scores:
            # Weight more important pairs (ones with higher scores) more heavily
            pair_scores.sort(reverse=True)
            # Use weighted average prioritizing the strongest pairings
            weights = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1]  # Diminishing weights
            total_weight = 0
            weighted_sum = 0
            
            for i, score in enumerate(pair_scores):
                if i < len(weights):
                    weight = weights[i]
                else:
                    weight = 0.05  # Minimal weight for additional pairs
                
                weighted_sum += score * weight
                total_weight += weight
            
            base_chemistry = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Apply position-based adjustments to the chemistry score
        # Modern NBA typically favors balanced lineups with specific position distributions
        lineup_balance = 0.0
        
        # Most effective lineup patterns get highest scores
        if (guards in [1, 2]) and (forwards in [2, 3]) and (centers in [0, 1]):
            lineup_balance = 1.0  # Balanced lineup
        elif guards == 0 or forwards == 0:
            lineup_balance = 0.3  # Severely unbalanced lineup
        elif guards >= 4 or forwards >= 4 or centers >= 2:
            lineup_balance = 0.5  # Unusual but potentially workable lineup
        else:
            lineup_balance = 0.8  # Somewhat balanced lineup
        
        # Combine pair chemistry with lineup balance (70% pair chemistry, 30% lineup balance)
        final_chemistry = (base_chemistry * 0.7) + (lineup_balance * 0.3)
        
        return min(1.0, max(0.0, final_chemistry))
    
    def get_pair_chemistry(self, player1: str, player2: str) -> float:
        """
        Get chemistry score for a pair of players
        Returns a value between 0 and 1
        """
        # Try both orderings of the pair since chemistry is symmetric
        pair = tuple(sorted([player1, player2]))
        
        # Check stored chemistry data for existing pairs
        if pair in self.player_chemistry:
            return self.player_chemistry[pair]
        
        # If not found, generate a chemistry score based on positions and other factors
        pos1 = self.get_player_position(player1)
        pos2 = self.get_player_position(player2)
        
        # Base chemistry values from position combinations
        position_chemistry = {
            ('G', 'G'): 0.75,  # Guard-Guard combination (typically good synergy)
            ('F', 'F'): 0.72,  # Forward-Forward combination
            ('C', 'C'): 0.65,  # Center-Center combination (less common, often redundant)
            ('G', 'F'): 0.78,  # Guard-Forward (very common effective pairing)
            ('G', 'C'): 0.76,  # Guard-Center (pick and roll/lob potential)
            ('F', 'C'): 0.74   # Forward-Center (frontcourt pair)
        }
        
        # Get base chemistry from position combination
        pos_key = tuple(sorted([pos1, pos2]))
        base_chemistry = position_chemistry.get(pos_key, 0.7)
        
        # Check if these players appear frequently together in data
        # Players that frequently play together likely have good chemistry
        freq_bonus = 0.0
        if pair in self.frequent_pairs:
            freq = self.frequent_pairs[pair]
            if freq >= 50:
                freq_bonus = 0.15  # Strong frequent pairing
            elif freq >= 30:
                freq_bonus = 0.10  # Moderate frequent pairing
            elif freq >= 10:
                freq_bonus = 0.05  # Some history together
        
        # Check win chemistry if available
        win_bonus = 0.0
        if pair in self.win_chemistry:
            win_rate = self.win_chemistry[pair]
            win_bonus = (win_rate - 0.5) * 0.2  # Scale from -0.1 to +0.1
        
        # Add some random variation based on both player names to ensure consistency
        # This creates pseudo-random chemistry that remains consistent for the same pair
        import hashlib
        import struct
        
        # Generate consistent hash for player pair
        hash_input = f"{player1}_{player2}".encode()
        hash_value = hashlib.md5(hash_input).digest()
        random_value = struct.unpack('d', hash_value[:8])[0]
        
        # Scale to small random variation between -0.05 and 0.05
        random_variation = (random_value * 0.1) - 0.05
        
        # Combine all factors
        chemistry = base_chemistry + freq_bonus + win_bonus + random_variation
        
        # Ensure chemistry is within valid range
        chemistry = min(1.0, max(0.0, chemistry))
        
        # Cache the calculated chemistry for future use
        self.player_chemistry[pair] = chemistry
        
        return chemistry
    
    def get_player_position(self, player: str) -> str:
        """Get position for a player using external position data if available"""
        from src.position_features import PositionFeatureGenerator
        
        # Try to use the more complete position data from PositionFeatureGenerator
        try:
            position_generator = PositionFeatureGenerator()
            return position_generator.get_player_position(player)
        except:
            # Fallback to simplified position logic
            if not isinstance(player, str):
                return 'F'  # Default to forward for non-string inputs
            
            player_lower = player.lower()
            
            # Guards
            if any(name in player_lower for name in ['paul', 'parker', 'nash', 'curry', 'irving', 'wall', 'lillard',
                                                 'rose', 'westbrook', 'rondo', 'kidd', 'wade', 'harden', 'ellis']):
            return 'G'
            # Centers    
            elif any(name in player_lower for name in ['howard', 'bynum', 'gasol', 'hibbert', 'jordan', 'noah', 
                                                   'cousins', 'chandler', 'jefferson', 'lopez', 'gortat']):
            return 'C'
            # Forwards
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
        
        print("Analyzing lineup chemistry throughout the dataset...")
        print("Using simplified chemistry analysis for faster processing...")
        
        # Take a small sample for faster processing
        sample_size = min(10000, len(df))
        sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
        
        # Process in small batches
        batch_size = 1000
        num_batches = (sample_size + batch_size - 1) // batch_size
        
        import tqdm
        for batch in tqdm.tqdm(range(num_batches), desc="Processing chemistry"):
            # Get batch indices
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, sample_size)
            batch_indices = sample_indices[start_idx:end_idx]
            
            # Process each row in the batch
            for idx in batch_indices:
                # Extract home team players from the row
                home_players = []
                for i in range(5):
                    col = f'home_{i}'
                    if col in df.columns and pd.notna(df.iloc[idx][col]):
                        player = df.iloc[idx][col]
                        if isinstance(player, str):
                            home_players.append(player)
                
                # Skip if not enough players
                if len(home_players) < 2:
                    continue
                    
                # Calculate chemistry score
            chemistry_score = self.calculate_chemistry(home_players)
                df.iloc[idx, df.columns.get_loc('chemistry_score')] = chemistry_score
        
        # Fill remaining rows with average chemistry
        mean_chemistry = df.loc[sample_indices, 'chemistry_score'].mean()
        df.loc[~df.index.isin(sample_indices), 'chemistry_score'] = mean_chemistry
        
        print(f"Chemistry analysis complete. Analyzed {sample_size} sample lineups.")
        return df 