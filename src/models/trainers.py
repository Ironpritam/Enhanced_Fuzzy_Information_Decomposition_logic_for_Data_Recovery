"""
Downstream Machine Learning Classifiers & Deep AutoEncoder Trainer.
Used to evaluate data quality retention before and after FID recovery.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import register_keras_serializable
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any, Optional

from ..core.memory import load_model, save_model

@register_keras_serializable()
class AutoEncoder(Model):
    def __init__(self, output_units: int, neck: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.output_units = output_units
        self.neck = neck
        self.threshold = None
        
        self.encoder = tf.keras.Sequential([
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(neck, activation='relu')
        ])
        
        self.decoder = tf.keras.Sequential([
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(output_units, activation='sigmoid')
        ])

    def predict_label(self, inputs: np.ndarray) -> np.ndarray:
        """Classify input sample as normal (1.0) or outlier (0.0) based on reconstruction error threshold."""
        encoded = self.encoder(inputs)
        raw = self.decoder(encoded)
        test_errors = tf.keras.losses.msle(raw, inputs)
        thresh = self.threshold if self.threshold is not None else 0.05
        ae_preds = (pd.Series(test_errors.numpy()) > thresh).map(lambda x: 0.0 if x else 1.0).astype('float32')
        return ae_preds.values

    def call(self, inputs):
        encoded = self.encoder(inputs)
        return self.decoder(encoded)

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_units': self.output_units,
            'neck': self.neck
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ModelTrainer:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.ae_threshold: Optional[float] = None

    def train_or_load(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        dataset_name: str,
        feature: str,
        model_type: str,
        sigma: float,
        OP: float
    ):
        """Train model or restore from cache directory."""
        cached_model = load_model(dataset_name, feature, model_type, sigma, OP)
        if cached_model is not None:
            self.models[f"{model_type}_{feature}"] = cached_model
            if model_type == 'ae':
                self.ae_threshold = getattr(cached_model, 'threshold', None)
            return

        if model_type == 'rf':
            self.train_rf(X_train, y_train, feature)
        elif model_type == 'knn':
            self.train_knn(X_train, y_train, feature)
        elif model_type == 'ae':
            self.train_autoencoder(X_train, y_train, feature, OP)

        model = self.models[f"{model_type}_{feature}"]
        save_model(model, dataset_name, feature, model_type, sigma, OP)

    def train_rf(self, X_train: np.ndarray, y_train: np.ndarray, feature_name: str):
        """Train Random Forest classifier."""
        rf = RandomForestClassifier(n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        self.models[f"rf_{feature_name}"] = rf

    def train_knn(self, X_train: np.ndarray, y_train: np.ndarray, feature_name: str):
        """Train K-Nearest Neighbors classifier."""
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        self.models[f"knn_{feature_name}"] = knn

    def train_autoencoder(self, X_train: np.ndarray, y_train: np.ndarray, feature_name: str, OP: float):
        """Train Deep AutoEncoder model and determine reconstruction threshold."""
        model = AutoEncoder(output_units=X_train.shape[1])
        model.compile(optimizer='adam', loss='msle', metrics=['mse'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

        model.fit(
            X_train, X_train,
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            verbose=0,
            callbacks=[early_stop]
        )

        reconstructions = model.predict(X_train, verbose=0)
        reconstruction_errors = tf.keras.losses.msle(reconstructions, X_train).numpy()
        self.ae_threshold = float(np.percentile(reconstruction_errors, 100 - OP))
        model.threshold = self.ae_threshold

        self.models[f"ae_{feature_name}"] = model
