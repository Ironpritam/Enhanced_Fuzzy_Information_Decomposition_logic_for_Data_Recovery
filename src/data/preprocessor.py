"""
Data Preprocessing & Noise Injection Utilities for FID Data Recovery.
Supports ForestFires, PeMS-Bay, and UNSW-NB15 benchmark datasets.
"""

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional

class DataPreprocessor:
    def __init__(
        self,
        df: pd.DataFrame,
        sel_feature: Optional[str] = None,
        sigma: float = 0.2,
        OP: float = 1.0,
        categorical_maps: Optional[Dict[str, Dict[Any, int]]] = None,
        label_col: Optional[str] = None
    ):
        self.df_org = df.copy()
        self.df = df.copy()
        self.sel_feature = sel_feature
        self.sigma = sigma
        self.OUTLIER_PERCENTAGE = OP
        self.length = len(self.df)
        self.COUNT_OUTLIER = int(self.OUTLIER_PERCENTAGE / 100.0 * self.length)
        
        np.random.seed(369)
        self.random_indices = np.random.choice(self.df.index, self.COUNT_OUTLIER, replace=False)
        self.features: List[str] = []
        self.categorical_maps = categorical_maps or {}
        self.label_col = label_col

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, List[str]]:
        """Process features, inject Gaussian noise/outliers into selected feature, scale data, and split into train/test."""
        self.df = self.df_org.copy()

        # Map categorical variables
        for col, mapping in self.categorical_maps.items():
            if col in self.df.columns:
                self.df[f'{col}_num'] = self.df[col].map(mapping).fillna(0)
                self.df = self.df.drop(columns=[col], errors='ignore')

        # Initialize labels if not present
        if self.label_col not in self.df.columns or self.label_col is None:
            self.df['label'] = 1  # 1 = Normal, 0 = Compromised/Outlier

        # Inject noise into selected target feature
        if self.sel_feature and self.sel_feature in self.df.columns:
            self.df[self.sel_feature] = self.df[self.sel_feature].astype(float)
            
            for index in self.random_indices:
                if index in self.df.index:
                    row = self.df.loc[index].copy()
                    val = float(row[self.sel_feature])
                    sigma_val = self.sigma * abs(val) if abs(val) > 1e-5 else self.sigma
                    noise = random.gauss(0, sigma_val)
                    row[self.sel_feature] = val + noise
                    row['label'] = 0
                    self.df.loc[index] = row

        # Select numerical feature columns
        non_feature_cols = {'label'}
        if self.label_col:
            non_feature_cols.add(self.label_col)
            
        self.features = [col for col in self.df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(self.df[col])]

        # Shuffle and split
        self.df = self.df.sample(frac=1, random_state=369).reset_index(drop=True)

        X = self.df[self.features].values
        y = self.df['label']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.20, random_state=369, shuffle=False
        )
        
        return X_scaled, X_train, X_test, y_train, y_test, self.features

class EnhancedDataPreprocessor(DataPreprocessor):
    @staticmethod
    def get_dataset_config(dataset_name: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Return preset configurations for benchmark datasets."""
        configs = {
            'forestfires': {
                'categorical_maps': {
                    'month': {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                              'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12},
                    'day': {'sun': 0, 'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6}
                },
                'label_col': None
            },
            'pems': {
                'categorical_maps': {},
                'label_col': None
            }
        }

        if dataset_name == 'unsw' and df is not None:
            configs['unsw'] = {
                'categorical_maps': {
                    'proto': {proto: i for i, proto in enumerate(df['proto'].unique())} if 'proto' in df else {},
                    'service': {service: i for i, service in enumerate(df['service'].unique())} if 'service' in df else {},
                    'attack_cat': {cat: i for i, cat in enumerate(df['attack_cat'].unique())} if 'attack_cat' in df else {}
                },
                'label_col': 'label'
            }
            return configs['unsw']

        return configs.get(dataset_name, {})

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        sel_feature: Optional[str] = None,
        sigma: float = 0.2,
        OP: float = 1.0
    ):
        config = self.get_dataset_config(dataset_name, df)
        super().__init__(
            df=df,
            sel_feature=sel_feature,
            sigma=sigma,
            OP=OP,
            categorical_maps=config.get('categorical_maps', None),
            label_col=config.get('label_col', None)
        )
