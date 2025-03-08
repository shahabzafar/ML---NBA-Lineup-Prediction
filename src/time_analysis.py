import pandas as pd
import numpy as np
from typing import Dict, List

class TimeAnalyzer:
    def __init__(self):
        self.time_patterns: Dict[str, Dict[str, float]] = {}
    
    def analyze_time_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze patterns in player usage based on game time"""
        df = df.copy()
        
        # Create time bins (e.g., 5-minute intervals)
        df['time_bin'] = pd.cut(df['starting_min'], 
                               bins=range(0, 49, 5),
                               labels=[f'{i}-{i+5}' for i in range(0, 45, 5)])
        
        # Analyze player frequency by time bin
        time_analysis = df.groupby(['time_bin', 'home_4']).size().reset_index(name='count')
        time_analysis = time_analysis.sort_values(['time_bin', 'count'], ascending=[True, False])
        
        # Store the patterns for prediction
        for time_bin in time_analysis['time_bin'].unique():
            bin_data = time_analysis[time_analysis['time_bin'] == time_bin]
            self.time_patterns[time_bin] = dict(zip(bin_data['home_4'], 
                                                  bin_data['count'] / bin_data['count'].sum()))
        
        return time_analysis
    
    def get_time_based_probabilities(self, time_bin: str) -> Dict[str, float]:
        """Get player probabilities for a given time bin"""
        if time_bin not in self.time_patterns:
            # Create more granular time bins
            minute = float(time_bin.split('-')[0])
            if minute >= 40:
                return self.time_patterns.get('40-45', {})
            elif minute >= 30:
                return self.time_patterns.get('30-35', {})
            elif minute >= 20:
                return self.time_patterns.get('20-25', {})
            else:
                return self.time_patterns.get('0-5', {})
        
        return self.time_patterns[time_bin] 