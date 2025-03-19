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
        # More comprehensive position list for players from 2007-2015
        positions = {
            # Guards
            'Chris Paul': 'G', 'Tony Parker': 'G', 'Kobe Bryant': 'G',
            'Steve Nash': 'G', 'Dwyane Wade': 'G', 'Russell Westbrook': 'G',
            'Rajon Rondo': 'G', 'Jason Kidd': 'G', 'Ray Allen': 'G',
            'Manu Ginobili': 'G', 'Stephen Curry': 'G', 'James Harden': 'G',
            'Derrick Rose': 'G', 'Deron Williams': 'G', 'Chauncey Billups': 'G',
            'Kyle Lowry': 'G', 'John Wall': 'G', 'Kyrie Irving': 'G',
            'Mike Conley': 'G', 'Brandon Jennings': 'G', 'Jose Calderon': 'G',
            'Ty Lawson': 'G', 'Jeremy Lin': 'G', 'Markel Brown': 'G',
            'Ricky Rubio': 'G', 'Eric Bledsoe': 'G', 'Lou Williams': 'G',
            'Goran Dragic': 'G', 'Isaiah Thomas': 'G', 'Jamal Crawford': 'G',
            'Monta Ellis': 'G', 'Jason Terry': 'G', 'J.J. Redick': 'G',
            'Danny Green': 'G', 'Kemba Walker': 'G', 'Brandon Knight': 'G',
            'Eric Gordon': 'G', 'Victor Oladipo': 'G', 'Nick Young': 'G',
            'Wesley Matthews': 'G', 'Klay Thompson': 'G', 'DeMar DeRozan': 'G',
            'Jimmy Butler': 'G', 'Kyle Korver': 'G', 'J.R. Smith': 'G',
            'Joe Johnson': 'G', 'Courtney Lee': 'G', 'Ben Gordon': 'G',
            'Damian Lillard': 'G', 'Jeff Teague': 'G', 'George Hill': 'G',
            
            # Forwards
            'LeBron James': 'F', 'Kevin Durant': 'F', 'Carmelo Anthony': 'F',
            'Paul Pierce': 'F', 'Kevin Garnett': 'F', 'Dirk Nowitzki': 'F',
            'Tim Duncan': 'F', 'Chris Bosh': 'F', 'Pau Gasol': 'F',
            'Amar\'e Stoudemire': 'F', 'Blake Griffin': 'F', 'David West': 'F',
            'Zach Randolph': 'F', 'Lamar Odom': 'F', 'Rudy Gay': 'F',
            'Paul George': 'F', 'Josh Smith': 'F', 'Luol Deng': 'F',
            'Andre Iguodala': 'F', 'Thaddeus Young': 'F', 'Tayshaun Prince': 'F',
            'Udonis Haslem': 'F', 'Mirza Teletovic': 'F', 'Shawn Marion': 'F',
            'Danilo Gallinari': 'F', 'David Lee': 'F', 'Carlos Boozer': 'F',
            'Andrei Kirilenko': 'F', 'LaMarcus Aldridge': 'F', 'Boris Diaw': 'F',
            'Al Horford': 'F', 'Serge Ibaka': 'F', 'Metta World Peace': 'F',
            'Kawhi Leonard': 'F', 'Trevor Ariza': 'F', 'Nicolas Batum': 'F',
            'Chandler Parsons': 'F', 'Jared Dudley': 'F', 'Jeff Green': 'F',
            'Josh McRoberts': 'F', 'Tobias Harris': 'F', 'Michael Beasley': 'F',
            'Wilson Chandler': 'F', 'Harrison Barnes': 'F', 'Kenneth Faried': 'F',
            'Amir Johnson': 'F', 'Ersan Ilyasova': 'F', 'Ryan Anderson': 'F',
            'Draymond Green': 'F', 'Carl Landry': 'F', 'Luis Scola': 'F',
            
            # Centers
            'Dwight Howard': 'C', 'Andrew Bynum': 'C', 'Marc Gasol': 'C',
            'Tyson Chandler': 'C', 'Joakim Noah': 'C', 'Al Jefferson': 'C',
            'Nene Hilario': 'C', 'Brook Lopez': 'C', 'Roy Hibbert': 'C',
            'DeMarcus Cousins': 'C', 'DeAndre Jordan': 'C', 'Chris Andersen': 'C',
            'Omer Asik': 'C', 'Robin Lopez': 'C', 'Marcin Gortat': 'C',
            'Kendrick Perkins': 'C', 'Emeka Okafor': 'C', 'Anderson Varejao': 'C',
            'Samuel Dalembert': 'C', 'Ian Mahinmi': 'C', 'JaVale McGee': 'C',
            'Nikola Pekovic': 'C', 'Greg Monroe': 'C', 'Tiago Splitter': 'C',
            'Andrew Bogut': 'C', 'Justin Hamilton': 'C', 'Timofey Mozgov': 'C',
            'Nikola Vucevic': 'C', 'Spencer Hawes': 'C', 'Bismack Biyombo': 'C',
            'Kosta Koufos': 'C', 'Mason Plumlee': 'C', 'Miles Plumlee': 'C',
            'Marreese Speights': 'C', 'Brendan Haywood': 'C', 'Brandan Wright': 'C',
            'Andrea Bargnani': 'C', 'Joel Anthony': 'C', 'Nazr Mohammed': 'C',
            'Lavoy Allen': 'C', 'Festus Ezeli': 'C', 'Zaza Pachulia': 'C',
            'Steven Adams': 'C', 'Jermaine O\'Neal': 'C', 'Jonas Valanciunas': 'C'
        }
        
        # Create a default function to infer positions from player names
        # This will be used for players not in our dictionary
        def infer_position(player: str) -> str:
            player_lower = player.lower()
            # Common point guard last names or naming patterns
            if any(name in player_lower for name in ['paul', 'parker', 'nash', 'curry', 'irving', 'wall', 'lillard', 'rose']):
                return 'G'
            # Common center indicators
            elif any(name in player_lower for name in ['howard', 'bynum', 'gasol', 'hibbert', 'jordan', 'noah', 'cousins']):
                return 'C'
            # Common forward indicators
            elif any(name in player_lower for name in ['james', 'durant', 'anthony', 'nowitzki', 'griffin', 'duncan', 'garnett']):
                return 'F'
            # Fallback to forward as the most common/versatile position
            else:
                return 'F'
        
        # Skip data loading for simplicity and reliability
        # We'll use our comprehensive predefined dictionary instead
        return positions
    
    def create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create position-based features for the dataset
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate position counts for home team lineups
            df['num_guards'] = 0
            df['num_forwards'] = 0
            df['num_centers'] = 0
            
        # Process each home player column
        for i in range(5):
            col = f'home_{i}'
            if col in df.columns:
                # Count player positions
                df[f'{col}_pos'] = df[col].apply(self.get_player_position)
                df['num_guards'] += (df[f'{col}_pos'] == 'G').astype(int)
                df['num_forwards'] += (df[f'{col}_pos'] == 'F').astype(int)
                df['num_centers'] += (df[f'{col}_pos'] == 'C').astype(int)
        
        # Create a lineup balance feature
        # 0 = very unbalanced, 1 = perfectly balanced
        df['lineup_balance'] = 0.0
        
        # A balanced lineup typically has 2G-2F-1C or similar distributions
        balanced_mask = (
            # 2G-2F-1C (most common balanced lineup)
            ((df['num_guards'] == 2) & (df['num_forwards'] == 2) & (df['num_centers'] == 1)) |
            # 2G-3F-0C (small ball lineup)
            ((df['num_guards'] == 2) & (df['num_forwards'] == 3) & (df['num_centers'] == 0)) |
            # 1G-3F-1C (forward-heavy)
            ((df['num_guards'] == 1) & (df['num_forwards'] == 3) & (df['num_centers'] == 1))
        )
        
        # Assign lineup balance scores
        df.loc[balanced_mask, 'lineup_balance'] = 1.0  # Balanced lineups
        df.loc[~balanced_mask, 'lineup_balance'] = 0.5  # Less balanced but still functional
        
        # Extreme imbalance gets the lowest score
        extreme_imbalance = (
            (df['num_guards'] == 0) |  # No guards
            (df['num_forwards'] == 0) |  # No forwards
            (df['num_guards'] == 5) |  # All guards
            (df['num_forwards'] == 5) |  # All forwards
            (df['num_centers'] >= 3)  # Too many centers
        )
        df.loc[extreme_imbalance, 'lineup_balance'] = 0.0
        
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