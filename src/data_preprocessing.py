import pandas as pd
import numpy as np
from typing import Tuple
import os
import glob

class DataPreprocessor:
    def __init__(self):
        self.player_encoder = None
        self.team_encoder = None
        
        # Corrected allowed features based on metadata
        self.allowed_features = [
            'game',          # ✓
            'season',        # ✓
            'home_team',     # ✓
            'away_team',     # ✓
            'starting_min',  # ✓
            'home_0',        # ✓
            'home_1',        # ✓
            'home_2',        # ✓
            'home_3',        # ✓
            # home_4 is our target
            'away_0',        # ✓
            'away_1',        # ✓
            'away_2',        # ✓
            'away_3',        # ✓
            'away_4'         # ✓
        ]
        # Removed 'end_min' as it cannot be used in the model
    
    def load_data(self, data_dir: str) -> pd.DataFrame:
        """Load multiple NBA matchups data files from a directory"""
        try:
            csv_files = glob.glob(os.path.join(data_dir, 'matchups-*.csv'))
            
            if not csv_files:
                raise Exception("No matchups CSV files found in the specified directory")
            
            all_data = []
            for file in csv_files:
                season_data = pd.read_csv(file)
                # Only keep allowed features
                season_data = season_data[self.allowed_features + ['home_4']]  # Include target
                
                season = os.path.basename(file).split('-')[1].split('.')[0]
                season_data['season'] = season
                all_data.append(season_data)
                print(f"Loaded {len(season_data)} rows from season {season}")
            
            df = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal: Loaded {len(df)} rows from {len(csv_files)} seasons")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables like team names and player IDs"""
        from sklearn.preprocessing import LabelEncoder
        
        df = df.copy()
        
        # Encode teams
        self.team_encoder = LabelEncoder()
        df['home_team_encoded'] = self.team_encoder.fit_transform(df['home_team'])
        df['away_team_encoded'] = self.team_encoder.transform(df['away_team'])
        
        # Encode players
        self.player_encoder = LabelEncoder()
        all_players = pd.concat([
            df['home_0'], df['home_1'], df['home_2'], df['home_3'], df['home_4'],
            df['away_0'], df['away_1'], df['away_2'], df['away_3'], df['away_4']
        ]).unique()
        
        self.player_encoder.fit(all_players)
        
        # Encode all player columns
        player_columns = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                         'away_0', 'away_1', 'away_2', 'away_3', 'away_4']
        
        for col in player_columns:
            df[f'{col}_encoded'] = self.player_encoder.transform(df[col])
        
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from allowed data"""
        # Remove duration calculation since end_min can't be used
        
        # Create lineup combination features
        df['home_starters_avg'] = df[['home_0_encoded', 'home_1_encoded', 
                                    'home_2_encoded', 'home_3_encoded']].mean(axis=1)
        df['away_starters_avg'] = df[['away_0_encoded', 'away_1_encoded', 
                                    'away_2_encoded', 'away_3_encoded', 'away_4_encoded']].mean(axis=1)
        
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling"""
        features = df[df['home_4'].notna()].copy()
        
        # Basic feature columns
        feature_cols = [
            'home_team_encoded', 'away_team_encoded',
            'home_0_encoded', 'home_1_encoded', 'home_2_encoded', 'home_3_encoded',
            'away_0_encoded', 'away_1_encoded', 'away_2_encoded', 'away_3_encoded', 'away_4_encoded',
            'starting_min'
        ]
        
        # Add chemistry and position features if they exist
        if 'num_guards' in df.columns:
            feature_cols.extend(['num_guards', 'num_forwards', 'num_centers'])
        
        if 'chemistry_score' in df.columns:
            feature_cols.append('chemistry_score')
        
        X = features[feature_cols]
        y = features['home_4_encoded']
        
        return X, y

    def process_pipeline(self, data_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Run the complete preprocessing pipeline"""
        df = self.load_data(data_dir)
        if df is not None:
            df = self.encode_categorical(df)
            df = self.create_features(df)
            X, y = self.prepare_data(df)
            return X, y
        return None, None 