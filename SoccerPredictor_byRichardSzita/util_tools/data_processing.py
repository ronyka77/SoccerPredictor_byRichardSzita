from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    @staticmethod
    def convert_to_float(value: Union[str, float, int]) -> float:
        """Convert string values to float, handling percentage signs"""
        try:
            if isinstance(value, str):
                return float(value.replace('%', '').strip())
            return float(value)
        except (ValueError, TypeError):
            return np.nan

    @staticmethod
    def extract_percentage_and_ratio(value: str, key: str, home_away: int) -> tuple:
        """Extract percentage and ratio from string values"""
        # Reference to existing implementation:
        # SoccerPredictor_byRichardSzita/data_tools/feature_engineering_for_predictions.py
        startLine: 25
        endLine: 48

    @staticmethod
    def flatten_stats(row: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested statistics dictionaries"""
        # Reference to existing implementation:
        # SoccerPredictor_byRichardSzita/data_tools/feature_engineering_for_predictions.py
        startLine: 51
        endLine: 72 