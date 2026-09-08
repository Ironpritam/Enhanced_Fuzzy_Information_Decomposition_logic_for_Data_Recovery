"""
Hardware-accelerated Fuzzy Information Decomposition (FID) Engine for Data Recovery.
Supports base FID, Pearson correlation weighting, Distance correlation weighting,
and Shallow Neural Network (MLP) weighting.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Union, List, Tuple

tf.random.set_seed(36)

class FIDRecovery:
    def __init__(self, small_dataset_threshold: int = 5000):
        self.small_dataset_threshold = small_dataset_threshold
        self.device = self._detect_hardware()
        self.strategy = self._initialize_strategy()
        
        # Pre-compile TF functions to avoid retracing
        self.compute_fid_tf = tf.function(self.compute_fid, reduce_retracing=True)

    def _detect_hardware(self) -> str:
        """Detect physical computing device (TPU, GPU, or CPU)."""
        try:
            if tf.config.list_physical_devices('TPU'):
                return 'tpu'
            elif tf.config.list_physical_devices('GPU'):
                return 'gpu'
        except Exception:
            pass
        return 'cpu'

    def _initialize_strategy(self):
        """Initialize TensorFlow distribution strategy based on detected hardware."""
        if self.device == 'tpu':
            try:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)
                return tf.distribute.TPUStrategy(resolver)
            except Exception:
                return tf.distribute.get_strategy()
        elif self.device == 'gpu':
            return tf.distribute.MirroredStrategy()
        else:
            return tf.distribute.get_strategy()

    def _should_use_gpu(self, data_size: int) -> bool:
        """Heuristic decision whether to leverage GPU/TPU or CPU fallback."""
        if self.device == 'cpu':
            return False
        return data_size > self.small_dataset_threshold

    def calculate_correlation_weights(self, X: np.ndarray, target_feature_index: int, method: str = 'pearson') -> np.ndarray:
        """Calculate pairwise correlation weights between target feature and all features."""
        if not self._should_use_gpu(X.shape[0]):
            df_local = pd.DataFrame(X)
            correlation_series = df_local.corr(method=method)[target_feature_index]
            feature_weights = correlation_series.abs().values
            feature_weights[target_feature_index] = 1.0
            return feature_weights / np.sum(feature_weights)

        if method == 'pearson':
            return self._gpu_pearson_correlation(X, target_feature_index)
        else:
            return self._cpu_correlation(X, target_feature_index, method)

    def _gpu_pearson_correlation(self, X: np.ndarray, target_feature_index: int) -> np.ndarray:
        """GPU-optimized Pearson correlation calculation."""
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        target = X_tensor[:, target_feature_index]

        centered = X_tensor - tf.reduce_mean(X_tensor, axis=0)
        target_centered = target - tf.reduce_mean(target)

        cov = tf.reduce_mean(centered * target_centered[:, tf.newaxis], axis=0)
        var_target = tf.reduce_mean(tf.square(target_centered))
        var_features = tf.reduce_mean(tf.square(centered), axis=0)

        corr = cov / (tf.sqrt(var_features * var_target) + 1e-8)
        corr = tf.where(tf.math.is_finite(corr), corr, tf.zeros_like(corr))
        corr = tf.tensor_scatter_nd_update(corr, [[target_feature_index]], [1.0])
        
        weights = tf.abs(corr)
        sum_w = tf.reduce_sum(weights)
        norm_weights = tf.where(sum_w > 0, weights / sum_w, weights)
        return norm_weights.numpy()

    def _cpu_correlation(self, X: np.ndarray, target_feature_index: int, method: str) -> np.ndarray:
        """CPU fallback for Pearson / Distance correlation."""
        if method == 'distance':
            return self.calculate_distance_correlation_weights(X, target_feature_index)
        else:
            df_local = pd.DataFrame(X)
            correlation_series = df_local.corr(method=method)[target_feature_index]
            feature_weights = correlation_series.abs().values
            feature_weights[target_feature_index] = 1.0
            return feature_weights / np.sum(feature_weights)

    def calculate_distance_correlation_weights(self, X: np.ndarray, target_feature_index: int) -> np.ndarray:
        """Compute non-linear distance correlation weights between target feature and all columns."""
        if not self._should_use_gpu(X.shape[0]):
            target_feature = X[:, target_feature_index]
            weights = []
            for i in range(X.shape[1]):
                if i == target_feature_index:
                    weights.append(1.0)
                else:
                    weights.append(self._cpu_distance_correlation(X[:, i], target_feature))
            weights = np.array(weights)
            return weights / np.sum(weights)
        else:
            return self._gpu_distance_correlation_weights(X, target_feature_index)

    def _cpu_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Distance correlation CPU implementation."""
        def _dcov(a_val, b_val):
            a = squareform(pdist(a_val[:, None], 'euclidean'))
            b = squareform(pdist(b_val[:, None], 'euclidean'))
            A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
            return np.sqrt(np.maximum(0.0, (A * B).mean()))
        
        d_xy = _dcov(x, y)
        d_xx = _dcov(x, x)
        d_yy = _dcov(y, y)
        denom = np.sqrt(d_xx * d_yy)
        return float(d_xy / denom) if denom > 0 else 0.0

    def _gpu_distance_correlation_weights(self, X: np.ndarray, target_feature_index: int) -> np.ndarray:
        """Optimized GPU batch distance correlation calculation."""
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        n_features = X_tensor.shape[1]
        target = X_tensor[:, target_feature_index]

        # Compute Euclidean distance matrices
        target_dist = tf.abs(tf.expand_dims(target, 0) - tf.expand_dims(target, 1))
        target_mean_col = tf.reduce_mean(target_dist, axis=0, keepdims=True)
        target_mean_row = tf.reduce_mean(target_dist, axis=1, keepdims=True)
        target_mean_all = tf.reduce_mean(target_dist)
        A_target = target_dist - target_mean_col - target_mean_row + target_mean_all

        cov_target = tf.reduce_mean(tf.square(A_target))

        weights = []
        for i in range(n_features):
            if i == target_feature_index:
                weights.append(1.0)
            else:
                feat = X_tensor[:, i]
                feat_dist = tf.abs(tf.expand_dims(feat, 0) - tf.expand_dims(feat, 1))
                feat_mean_col = tf.reduce_mean(feat_dist, axis=0, keepdims=True)
                feat_mean_row = tf.reduce_mean(feat_dist, axis=1, keepdims=True)
                feat_mean_all = tf.reduce_mean(feat_dist)
                A_feat = feat_dist - feat_mean_col - feat_mean_row + feat_mean_all

                cov_feat = tf.reduce_mean(tf.square(A_feat))
                cov_cross = tf.reduce_mean(A_feat * A_target)
                
                denom = tf.sqrt(tf.maximum(1e-8, cov_feat * cov_target))
                dcorr = cov_cross / denom
                weights.append(float(tf.clip_by_value(dcorr, 0.0, 1.0).numpy()))

        weights = np.array(weights)
        return weights / np.sum(weights)

    def calculate_shallow_nn_weights(self, X: np.ndarray, target_feature_index: int) -> np.ndarray:
        """Calculate feature contribution weights using a lightweight Multi-Layer Perceptron regressor."""
        n_features = X.shape[1]
        feature_indices = [i for i in range(n_features) if i != target_feature_index]
        
        X_other = X[:, feature_indices]
        y_target = X[:, target_feature_index]
        
        # Build shallow MLP regressor
        inputs = tf.keras.Input(shape=(len(feature_indices),))
        h = tf.keras.layers.Dense(10, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(1)(h)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_other, y_target, epochs=30, batch_size=64, verbose=0)
        
        # Estimate feature importances via absolute dense layer weights sum
        first_layer_weights = np.abs(model.layers[1].get_weights()[0])  # Shape: (n_other, 10)
        importances = np.sum(first_layer_weights, axis=1)  # Shape: (n_other,)
        
        weights = np.zeros(n_features)
        for idx, orig_idx in enumerate(feature_indices):
            weights[orig_idx] = importances[idx]
        weights[target_feature_index] = 1.0
        
        sum_w = np.sum(weights)
        return weights / sum_w if sum_w > 0 else weights

    def compute_fid(self, org_tensor: tf.Tensor, comp_indices: tf.Tensor) -> tf.Tensor:
        """Vectorized base FID computation via TensorFlow graph."""
        mask = tf.ones_like(org_tensor, dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(
            mask,
            tf.expand_dims(comp_indices, 1),
            tf.zeros_like(comp_indices, dtype=tf.bool)
        )

        observed_data = tf.boolean_mask(org_tensor, mask)
        a, b = tf.reduce_min(observed_data), tf.reduce_max(observed_data)
        t = tf.size(comp_indices)
        l = (b - a) / tf.cast(t, tf.float32)

        s = tf.range(1, t + 1, dtype=tf.float32)
        V = (a + (s - 1) * l + a + s * l) / 2.0

        diff = tf.abs(tf.expand_dims(observed_data, 0) - tf.expand_dims(V, 1))
        weights = tf.where(diff <= l, 1.0 - diff / l, 0.0)

        sum_weights = tf.reduce_sum(weights, axis=1)
        weighted_sum = tf.reduce_sum(weights * tf.expand_dims(observed_data, 0), axis=1)
        
        R = tf.where(
            sum_weights > 0,
            weighted_sum / sum_weights,
            tf.reduce_mean(observed_data)
        )

        return tf.tensor_scatter_nd_update(org_tensor, tf.expand_dims(comp_indices, 1), R)

    def FID(
        self,
        org_data: np.ndarray,
        compromised_indices: np.ndarray,
        X: Optional[np.ndarray] = None,
        target_feature_index: Optional[int] = None,
        method: Optional[str] = None
    ) -> np.ndarray:
        """Main entry point selecting CPU or GPU accelerated FID recovery."""
        if not self._should_use_gpu(len(org_data)):
            return self._cpu_FID(org_data, compromised_indices, X, target_feature_index, method)
        else:
            return self._gpu_FID(org_data, compromised_indices, X, target_feature_index, method)

    def _cpu_FID(
        self,
        org_data: np.ndarray,
        compromised_indices: np.ndarray,
        X: Optional[np.ndarray] = None,
        target_feature_index: Optional[int] = None,
        method: Optional[str] = None
    ) -> np.ndarray:
        """CPU implementation of Base FID and Correlation-Weighted FID."""
        if method is None:
            compromised_indices_set = set(compromised_indices)
            observed_data = [val for ind, val in enumerate(org_data) if ind not in compromised_indices_set]

            if len(observed_data) == 0:
                return np.copy(org_data)

            mean_val = float(np.mean(observed_data))
            a, b = min(observed_data), max(observed_data)
            t = len(compromised_indices)
            if t == 0:
                return np.copy(org_data)
            
            l = (b - a) / t if t > 0 else 1.0
            V = [(a + (s - 1) * l + a + s * l) / 2.0 for s in range(1, t + 1)]

            R = []
            for v in V:
                contrib = [1 - abs(i - v) / l if abs(i - v) <= l else 0 for i in observed_data]
                sum_w = sum(contrib)
                if sum_w == 0:
                    R.append(mean_val)
                else:
                    weighted_sum = sum(w * val for w, val in zip(contrib, observed_data))
                    R.append(weighted_sum / sum_w)

            recovered_data = np.copy(org_data)
            for i, idx in enumerate(compromised_indices):
                recovered_data[idx] = R[i]
            return recovered_data
        else:
            if method == 'pearson':
                feature_weights = self.calculate_correlation_weights(X, target_feature_index, 'pearson')
            elif method == 'distance':
                feature_weights = self.calculate_distance_correlation_weights(X, target_feature_index)
            elif method == 'shallow_nn':
                feature_weights = self.calculate_shallow_nn_weights(X, target_feature_index)
            else:
                raise ValueError(f"Unknown recovery method: {method}")

            compromised_indices_set = set(compromised_indices)
            observed_data = [val for ind, val in enumerate(org_data) if ind not in compromised_indices_set]
            observed_X = np.delete(X, compromised_indices, axis=0)

            if len(observed_data) == 0:
                return np.copy(org_data)

            mean_val = float(np.mean(observed_data))
            a, b = min(observed_data), max(observed_data)
            t = len(compromised_indices)
            if t == 0:
                return np.copy(org_data)

            l = (b - a) / t if t > 0 else 1.0
            V = [(a + (s - 1) * l + a + s * l) / 2.0 for s in range(1, t + 1)]

            R = []
            for v in V:
                contrib = [1 - abs(i - v) / l if abs(i - v) <= l else 0 for i in observed_data]
                sum_w = sum(contrib)
                weighted_obs = 0.0
                for i in range(X.shape[1]):
                    if i != target_feature_index:
                        weighted_obs += feature_weights[i] * np.sum(observed_X[:, i] * contrib)
                if sum_w == 0:
                    R.append(mean_val)
                else:
                    R.append(weighted_obs / sum_w)

            recovered_data = np.copy(org_data)
            for i, idx in enumerate(compromised_indices):
                recovered_data[idx] = R[i]
            return recovered_data

    def _gpu_FID(
        self,
        org_data: np.ndarray,
        compromised_indices: np.ndarray,
        X: Optional[np.ndarray] = None,
        target_feature_index: Optional[int] = None,
        method: Optional[str] = None
    ) -> np.ndarray:
        """GPU implementation of FID recovery."""
        if method is None:
            org_tensor = tf.convert_to_tensor(org_data, dtype=tf.float32)
            comp_indices = tf.convert_to_tensor(compromised_indices, dtype=tf.int32)
            return self.compute_fid_tf(org_tensor, comp_indices).numpy()
        else:
            return self._gpu_correlation_FID(org_data, compromised_indices, X, target_feature_index, method)

    def _gpu_correlation_FID(
        self,
        org_data: np.ndarray,
        compromised_indices: np.ndarray,
        X: np.ndarray,
        target_feature_index: int,
        method: str
    ) -> np.ndarray:
        """GPU tensor flow implementation of Correlation-Weighted FID."""
        org_tensor = tf.convert_to_tensor(org_data, dtype=tf.float32)
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        comp_indices = tf.convert_to_tensor(compromised_indices, dtype=tf.int32)

        if method == 'pearson':
            weights = self._gpu_pearson_correlation(X, target_feature_index)
        elif method == 'distance':
            weights = self._gpu_distance_correlation_weights(X, target_feature_index)
        elif method == 'shallow_nn':
            weights = self.calculate_shallow_nn_weights(X, target_feature_index)

        weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)

        observed_mask = tf.ones_like(org_tensor, dtype=tf.bool)
        observed_mask = tf.tensor_scatter_nd_update(
            observed_mask,
            tf.expand_dims(comp_indices, 1),
            tf.zeros_like(comp_indices, dtype=tf.bool)
        )
        observed_data = tf.boolean_mask(org_tensor, observed_mask)
        observed_X = tf.boolean_mask(X_tensor, observed_mask)

        a, b = tf.reduce_min(observed_data), tf.reduce_max(observed_data)
        t = tf.size(comp_indices)
        l = (b - a) / tf.cast(t, tf.float32)
        s = tf.range(1, t + 1, dtype=tf.float32)
        V = (a + (s - 1) * l + a + s * l) / 2.0

        diff = tf.abs(tf.expand_dims(observed_data, 0) - tf.expand_dims(V, 1))
        contrib_weights = tf.where(diff <= l, 1.0 - diff / l, 0.0)

        weighted_contributions = tf.einsum(
            'ji,tj->ti',
            observed_X * weights_tensor[None, :],
            contrib_weights
        )

        mask = tf.one_hot(target_feature_index, X.shape[1], on_value=0.0, off_value=1.0)
        weighted_observed = tf.reduce_sum(weighted_contributions * mask, axis=1)

        sum_weights = tf.reduce_sum(contrib_weights, axis=1)
        R = tf.where(sum_weights > 0, weighted_observed / sum_weights, tf.reduce_mean(observed_data))

        return tf.tensor_scatter_nd_update(org_tensor, tf.expand_dims(comp_indices, 1), R).numpy()
