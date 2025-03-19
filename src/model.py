from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bars

class LineupPredictor:
    def __init__(self):
        # Optimized Random Forest classifier configuration for balanced speed and accuracy
        self.model = RandomForestClassifier(
            n_estimators=150,          # Reduced for faster test runs but still good accuracy
            max_depth=20,              # Deeper trees to capture more complex patterns
            min_samples_split=4,       # Require fewer samples to split a node for better fit
            min_samples_leaf=1,        # Allow smaller leaf nodes for more detailed predictions
            max_features='sqrt',       # Use square root of features for each split (helps with high-dimensional data)
            bootstrap=True,            # Use bootstrap samples
            random_state=42,           # Set seed for reproducibility
            n_jobs=-1,                 # Use all CPU cores for faster training
            class_weight='balanced_subsample',  # Better weight balancing for skewed class distributions
            verbose=0                  # Minimal console output during training
        )
        # Feature scaling to normalize input data
        self.scaler = StandardScaler()
        # Minimum number of samples required to include a player in predictions
        self.min_samples_per_player = 10  # Lowered threshold to include more players in model
        # Storage for categorical encoders
        self.team_encoder = None
        self.player_encoder = None
        self.feature_names = None
        # Track feature importance for better insights
        self.feature_importance = None
        # Store player frequency in the dataset for weighted predictions
        self.player_frequency = {}
    
    def set_encoders(self, team_encoder, player_encoder):
        """Set the encoders after preprocessing"""
        # Store encoder objects for later use in predictions
        self.team_encoder = team_encoder
        self.player_encoder = player_encoder
    
    def remove_rare_players(self, X: pd.DataFrame, y: pd.Series):
        """Remove players that appear less than min_samples_per_player times"""
        print("Filtering rare players...")
        # Count occurrences of each player in the target variable
        player_counts = y.value_counts()
        # Store player frequencies for later prediction adjustments
        total_players = len(y)
        for player, count in player_counts.items():
            self.player_frequency[player] = count / total_players
        
        # Keep only players that meet the minimum sample threshold
        valid_players = player_counts[player_counts >= self.min_samples_per_player].index
        
        # Create mask for rows with valid players
        mask = y.isin(valid_players)
        X_filtered = X[mask].copy()
        y_filtered = y[mask].copy()
        
        # Report filtering statistics
        print(f"Original samples: {len(y)}")
        print(f"Filtered samples: {len(y_filtered)}")
        print(f"Original unique players: {len(player_counts)}")
        print(f"Filtered unique players: {len(valid_players)}")
        
        return X_filtered, y_filtered
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model with improved hyperparameter settings"""
        print("Starting training pipeline...")
        
        # Store feature names for consistent prediction input
        self.feature_names = X.columns.tolist()
        
        # Step 1: Remove rare players to improve model quality
        X, y = self.remove_rare_players(X, y)
        
        # Step 2: Skip SMOTE and imbalance handling for faster processing
        
        # Step 3: Split data into training and validation sets with stratification
        print("Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Step 4: Scale features to normalize the data
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert to float32 to reduce memory usage
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_val_scaled = X_val_scaled.astype(np.float32)
        
        # Step 5: Train the model on the prepared data
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Step 6: Evaluate model performance
        print("Evaluating model...")
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        # Report accuracy metrics
        print("\nModel Performance:")
        print("Training Accuracy:", accuracy_score(y_train, train_pred))
        print("Validation Accuracy:", accuracy_score(y_val, val_pred))
        print("\nDetailed Validation Report:")
        print(classification_report(y_val, val_pred))
        
        # Analyze feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        return self.model

    def predict_fifth_player(self, game_data: pd.DataFrame) -> dict:
        """Predict the optimal fifth player for a given game situation"""
        # Scale input data using the same scaler as during training
        X_scaled = self.scaler.transform(game_data)
        
        # Get raw probability distribution for all possible players
        raw_probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Process and normalize probabilities with enhanced logic
        player_probs = {}
        
        # Check if probabilities exist for predictions
        if sum(raw_probabilities) == 0:
            return {}  # No valid predictions
        
        # Process all non-zero probabilities
        for idx, prob in enumerate(raw_probabilities):
            if prob > 0.0001:  # Filter extremely low probabilities
                try:
                    # Convert encoded index back to player name
                    player_idx = self.model.classes_[idx]
                    player = self.player_encoder.inverse_transform([player_idx])[0]
                    
                    # Apply multiple adjustments to improve prediction quality
                    
                    # 1. Scale probabilities based on prediction confidence
                    # This makes confident predictions more pronounced
                    confidence_boost = np.sqrt(prob)  # Square root makes the distribution less extreme
                    
                    # 2. Apply frequency adjustment to boost common players slightly
                    # Very rare players are less likely to be correct predictions
                    player_freq = self.player_frequency.get(player_idx, 0.001)
                    freq_factor = 0.5 + 0.5 * min(1.0, player_freq * 1000)
                    
                    # 3. Combine adjustments (with higher weight on model prediction)
                    final_prob = (confidence_boost * 0.8) + (freq_factor * 0.2)
                    
                    # Store adjusted probability
                    player_probs[player] = final_prob * 100  # Convert to percentage
                except Exception as e:
                    # Skip problematic predictions
                    continue
        
        # Return top predictions sorted by probability
        sorted_predictions = dict(sorted(player_probs.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:5])
        
        return sorted_predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for X with enhanced prediction logic"""
        # Validate that all required features are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Reorder columns to match training data format
        X = X[self.feature_names]
        
        # Scale the input data for consistency with training
        X_scaled = self.scaler.transform(X)
        
        # Get raw probability predictions from model
        raw_probs = self.model.predict_proba(X_scaled)
        
        # Apply softmax normalization to sharpen predictions
        def softmax(x, temperature=1.0):
            # Temperature controls the sharpness (lower temp = sharper predictions)
            x = x / temperature
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        # Apply softmax with a temperature to control prediction sharpness
        sharpened_probs = softmax(raw_probs, temperature=0.6)  # Temperature < 1 makes confident predictions more pronounced
        
        # Scale to percentage
        scaled_probs = sharpened_probs * 100
        
        # Clip to valid range
        np.clip(scaled_probs, 0, 100, out=scaled_probs)
        
        return scaled_probs

    def create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on player positions"""
        # Count guards in home team lineup (players 0-3)
        df['home_guards'] = df[['home_0', 'home_1', 'home_2', 'home_3']].apply(
            lambda x: sum(self.is_guard(player) for player in x), axis=1
        )
        # Count forwards in home team lineup (players 0-3)
        df['home_forwards'] = df[['home_0', 'home_1', 'home_2', 'home_3']].apply(
            lambda x: sum(self.is_forward(player) for player in x), axis=1
        )
        return df 