from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bars

class LineupPredictor:
    def __init__(self):
        # Optimized for speed
        self.model = RandomForestClassifier(
            n_estimators=50,           # Reduced trees
            max_depth=10,              # Limited depth
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,                 # Use all CPU cores
            class_weight='balanced',   
            verbose=0                  # Reduced verbosity
        )
        self.scaler = StandardScaler()
        self.min_samples_per_player = 100  # Increased to reduce classes further
        # Add these lines to store encoders
        self.team_encoder = None
        self.player_encoder = None
        self.feature_names = None
    
    def set_encoders(self, team_encoder, player_encoder):
        """Set the encoders after preprocessing"""
        self.team_encoder = team_encoder
        self.player_encoder = player_encoder
    
    def remove_rare_players(self, X: pd.DataFrame, y: pd.Series):
        """Remove players that appear less than min_samples_per_player times"""
        print("Filtering rare players...")
        player_counts = y.value_counts()
        valid_players = player_counts[player_counts >= self.min_samples_per_player].index
        
        mask = y.isin(valid_players)
        X_filtered = X[mask].copy()
        y_filtered = y[mask].copy()
        
        print(f"Original samples: {len(y)}")
        print(f"Filtered samples: {len(y_filtered)}")
        print(f"Original unique players: {len(player_counts)}")
        print(f"Filtered unique players: {len(valid_players)}")
        
        return X_filtered, y_filtered
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model with progress bars and faster processing"""
        print("Starting training pipeline...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Step 1: Remove rare players
        X, y = self.remove_rare_players(X, y)
        
        # Step 2: Take a subset of data for faster processing
        if len(X) > 50000:  # If we have more than 50k samples
            print("Taking a subset of data for faster processing...")
            X_subset, _, y_subset, _ = train_test_split(
                X, y, train_size=50000, random_state=42, stratify=y
            )
            X, y = X_subset, y_subset
        
        print("Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert to float32 to save memory
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_val_scaled = X_val_scaled.astype(np.float32)
        
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        print("Making predictions...")
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        print("\nModel Performance:")
        print("Training Accuracy:", accuracy_score(y_train, train_pred))
        print("Validation Accuracy:", accuracy_score(y_val, val_pred))
        print("\nDetailed Validation Report:")
        print(classification_report(y_val, val_pred))
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return self.model

    def predict_fifth_player(self, game_data: pd.DataFrame) -> str:
        """Predict the optimal fifth player for a given game situation"""
        X_scaled = self.scaler.transform(game_data)
        return self.model.predict(X_scaled)[0]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for X"""
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on player positions"""
        # Add this method to DataPreprocessor class
        df['home_guards'] = df[['home_0', 'home_1', 'home_2', 'home_3']].apply(
            lambda x: sum(self.is_guard(player) for player in x), axis=1
        )
        df['home_forwards'] = df[['home_0', 'home_1', 'home_2', 'home_3']].apply(
            lambda x: sum(self.is_forward(player) for player in x), axis=1
        )
        return df 