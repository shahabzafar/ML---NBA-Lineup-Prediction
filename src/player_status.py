from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

class PlayerStatusTracker:
    def __init__(self, data_dir=None):
        """
        Initialize player status tracker using matchup data
        Args:
            data_dir: Directory containing matchup CSV files
        """
        # Directory with matchup data files
        self.data_dir = data_dir
        # Dictionary to store comprehensive player status information
        self.player_status = {}
        # Dictionary to track which teams each player has played for
        self.player_teams = {}
        # Dictionary to track game appearances for each player
        self.player_appearances = {}
        # Dictionary to track consecutive absences from lineups
        self.consecutive_absences = {}
        
        if data_dir:
            print(f"Extracting player status data from matchup files in: {data_dir}")
            self.extract_from_matchups(data_dir)
        else:
            print("No data directory provided for player status tracking")
    
    def extract_from_matchups(self, data_dir):
        """Extract player status information from all matchup data files - optimized version"""
        player_seasons = {}  # Track seasons each player appeared in
        player_teams = {}    # Track teams each player played for
        start_time = time.time()
        
        # Process each matchup file (each represents one season)
        for year in range(2007, 2016):
            file_path = os.path.join(data_dir, f'matchups-{year}.csv')
            if not os.path.exists(file_path):
                print(f"Matchup file not found for year {year}")
                continue
                
            print(f"Processing {year} matchup data for player status...")
            
            try:
                # Read only necessary columns to improve performance
                needed_columns = ['home_team', 'away_team']
                for i in range(5):
                    needed_columns.extend([f'home_{i}', f'away_{i}'])
                
                # Process file in chunks to handle large datasets efficiently
                chunk_size = 10000
                chunk_num = 0
                
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, 
                                        usecols=lambda x: x in needed_columns or x == 'game'):
                    chunk_num += 1
                    print(f"  Processing chunk {chunk_num} for year {year}...")
                    
                    # Process home team players
                    for i in range(5):
                        home_col = f'home_{i}'
                        if home_col in chunk.columns:
                            # Get unique non-null players in this chunk
                            home_players = chunk[home_col].dropna().unique()
                            
                            # Process each player
                            for player in home_players:
                                if not isinstance(player, str):
                                    continue
                                    
                                # Update seasons information
                                if player not in player_seasons:
                                    player_seasons[player] = set()
                                player_seasons[player].add(year)
                                
                                # Initialize team tracking if needed
                                if player not in player_teams:
                                    player_teams[player] = set()
                                
                                # Add team affiliations using vectorized operations
                                player_rows = chunk[chunk[home_col] == player]
                                if 'home_team' in player_rows:
                                    teams = player_rows['home_team'].dropna().unique()
                                    for team in teams:
                                        if isinstance(team, str):
                                            player_teams[player].add(team)
                    
                    # Process away team players
                    for i in range(5):
                        away_col = f'away_{i}'
                        if away_col in chunk.columns:
                            # Get unique non-null players in this chunk
                            away_players = chunk[away_col].dropna().unique()
                            
                            # Process each player
                            for player in away_players:
                                if not isinstance(player, str):
                                    continue
                                    
                                # Update seasons information
                                if player not in player_seasons:
                                    player_seasons[player] = set()
                                player_seasons[player].add(year)
                                
                                # Initialize team tracking if needed
                                if player not in player_teams:
                                    player_teams[player] = set()
                                
                                # Add team affiliations using vectorized operations
                                player_rows = chunk[chunk[away_col] == player]
                                if 'away_team' in player_rows:
                                    teams = player_rows['away_team'].dropna().unique()
                                    for team in teams:
                                        if isinstance(team, str):
                                            player_teams[player].add(team)
            
            except Exception as e:
                print(f"Error processing matchup file for {year}: {str(e)}")
                continue
            
            # Report progress after each year
            elapsed = time.time() - start_time
            print(f"Processed year {year} data in {elapsed:.2f} seconds. Found {len(player_seasons)} players so far.")
        
        # Construct player status data for all identified players
        for player, seasons in player_seasons.items():
            self.player_status[player] = {
                'status': 'active',  # All players in dataset considered active
                'seasons': sorted(list(seasons)),
                'teams': sorted(list(player_teams.get(player, [])))
            }
        
        # Store team affiliations for all players
        self.player_teams = player_teams
        
        # Report final results
        total_time = time.time() - start_time
        print(f"Completed extraction in {total_time:.2f} seconds")
        print(f"Extracted status data for {len(self.player_status)} players from matchup files")
    
    def is_player_eligible(self, player: str, season: int, game_time: float = 0) -> bool:
        """Check if a player is eligible to play at given time"""
        # If player not in database, assume eligible (handles new players)
        if player not in self.player_status:
            print(f"  {player} not in status database, assuming eligible")
            return True
        
        player_data = self.player_status[player]
        
        # Check if player was active in the specified season
        if season not in player_data.get('seasons', []):
            print(f"  {player} not active in season {season}")
            return False
        
        return True
    
    def get_player_teams(self, player: str) -> List[str]:
        """Get teams a player has played for"""
        # Return sorted list of teams the player has been associated with
        if player in self.player_teams:
            return sorted(list(self.player_teams[player]))
        return []
    
    def get_player_status_info(self, player: str) -> Dict:
        """Get comprehensive player status information"""
        # Return complete player status data if available
        if player in self.player_status:
            return self.player_status[player]
        # Return default empty values if player not found
        return {'status': 'unknown', 'seasons': [], 'injuries': [], 'teams': []}
    
    def get_players_by_season(self, season: int):
        """Get all players active in a specific season"""
        players = []
        # Find all players who were active in the specified season
        for player, data in self.player_status.items():
            if season in data.get('seasons', []):
                players.append(player)
        return sorted(players)
    
    def get_players_by_team(self, team: str, season: int = None) -> List[str]:
        """Get all players who played for a specific team, optionally filtered by season"""
        players = []
        # Find players who played for the specified team
        for player, data in self.player_status.items():
            if team in data.get('teams', []):
                # Apply optional season filter
                if season is None or season in data.get('seasons', []):
                    players.append(player)
        return sorted(players) 