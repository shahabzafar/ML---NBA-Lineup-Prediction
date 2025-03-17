from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

class TeamAnalyzer:
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize team performance analyzer
        Args:
            historical_data: DataFrame containing historical game data
        """
        self.data = historical_data
        self.team_metrics = {}
        self.lineup_success_rates = {}
        
    def calculate_team_success_metrics(self, team: str, season: int) -> Dict:
        """
        Calculate various team success metrics
        Args:
            team: Team abbreviation
            season: Season year
        Returns:
            Dict containing team performance metrics
        """
        try:
            # Filter data for team and season
            team_data = self.data[
                (self.data['season'] == str(season)) &
                ((self.data['home_team'] == team) | (self.data['away_team'] == team))
            ]
            
            # Calculate metrics
            total_games = len(team_data)
            home_games = len(team_data[team_data['home_team'] == team])
            away_games = total_games - home_games
            
            # Store metrics
            metrics = {
                'total_games': total_games,
                'home_games': home_games,
                'away_games': away_games,
                'lineup_variations': self._count_lineup_variations(team_data, team),
                'avg_chemistry': self._calculate_avg_chemistry(team_data, team)
            }
            
            self.team_metrics[(team, season)] = metrics
            return metrics
            
        except Exception as e:
            print(f"Error calculating team metrics: {e}")
            return {}
    
    def analyze_lineup_success(self, lineup: List[str], team: str, season: int) -> Dict:
        """
        Analyze historical success rate of lineup combinations
        Args:
            lineup: List of player names
            team: Team abbreviation
            season: Season year
        Returns:
            Dict containing lineup success metrics
        """
        try:
            # Get games where these players played together
            lineup_games = self._find_lineup_games(lineup, team, season)
            
            if lineup_games.empty:
                return {'success_rate': 0.0, 'games_played': 0}
            
            # Calculate success metrics
            success_rate = self._calculate_lineup_success_rate(lineup_games, team)
            
            metrics = {
                'success_rate': success_rate,
                'games_played': len(lineup_games),
                'avg_minutes_together': self._calculate_avg_minutes(lineup_games, lineup)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing lineup success: {e}")
            return {'success_rate': 0.0, 'games_played': 0}
    
    def _count_lineup_variations(self, team_data: pd.DataFrame, team: str) -> int:
        """Count unique lineup combinations used by team"""
        unique_lineups = set()
        
        for _, row in team_data.iterrows():
            if row['home_team'] == team:
                players = [row[f'home_{i}'] for i in range(5)]
            else:
                players = [row[f'away_{i}'] for i in range(5)]
            unique_lineups.add(tuple(sorted(players)))
            
        return len(unique_lineups)
    
    def _calculate_avg_chemistry(self, team_data: pd.DataFrame, team: str) -> float:
        """Calculate average team chemistry score"""
        if 'chemistry_score' in team_data.columns:
            return team_data['chemistry_score'].mean()
        return 0.0
    
    def _find_lineup_games(self, lineup: List[str], team: str, season: int) -> pd.DataFrame:
        """Find games where the lineup played together"""
        team_data = self.data[
            (self.data['season'] == str(season)) &
            ((self.data['home_team'] == team) | (self.data['away_team'] == team))
        ]
        
        # Filter for games where all players in lineup played
        for player in lineup:
            player_mask = False
            for i in range(5):
                player_mask |= (team_data[f'home_{i}'] == player) | (team_data[f'away_{i}'] == player)
            team_data = team_data[player_mask]
            
        return team_data
    
    def _calculate_lineup_success_rate(self, games: pd.DataFrame, team: str) -> float:
        """Calculate success rate for lineup"""
        # This is a placeholder - implement actual success metrics
        return len(games) / 100.0
    
    def _calculate_avg_minutes(self, games: pd.DataFrame, lineup: List[str]) -> float:
        """Calculate average minutes played together"""
        if 'duration' in games.columns:
            return games['duration'].mean()
        return 0.0 