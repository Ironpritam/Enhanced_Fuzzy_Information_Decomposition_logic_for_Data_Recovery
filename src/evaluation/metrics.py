"""
Evaluation Metrics & Visualization Suite for FID Data Recovery.
Calculates reconstruction fidelity (MAE, RMSE, Pearson r), statistical property
preservation (Jensen-Shannon Divergence, KS Test), and downstream ML accuracy retention.
"""

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ks_2samp
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from typing import Dict, Any, List, Optional, Tuple

class RecoveryEvaluator:
    def __init__(self, original_data: np.ndarray, feature_name: str = "target"):
        self.original_data = np.asarray(original_data)
        self.feature_name = feature_name
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate_recovery(
        self,
        recovered_data: np.ndarray,
        compromised_indices: np.ndarray,
        method_name: str
    ) -> Dict[str, Any]:
        """Evaluate recovery metrics for compromised indices and full dataset distribution."""
        recovered_data = np.asarray(recovered_data)
        self.results[method_name] = {}
        
        # 1. Direct Reconstruction Metrics
        self.results[method_name]['reconstruction'] = self._calculate_reconstruction_metrics(
            self.original_data, recovered_data, compromised_indices
        )
        
        # 2. Statistical Property Preservation
        self.results[method_name]['statistics'] = self._compare_distributions(
            self.original_data, recovered_data
        )

        return self.results[method_name]

    def _calculate_reconstruction_metrics(
        self,
        original: np.ndarray,
        recovered: np.ndarray,
        compromised_idx: np.ndarray
    ) -> Dict[str, float]:
        """Compute MAE, RMSE, and Pearson correlation coefficient on compromised points."""
        if len(compromised_idx) == 0:
            return {'MAE': 0.0, 'RMSE': 0.0, 'Pearson_r': 1.0}

        comp_orig = original[compromised_idx]
        comp_rec = recovered[compromised_idx]

        mae = float(mean_absolute_error(comp_orig, comp_rec))
        rmse = float(np.sqrt(mean_squared_error(comp_orig, comp_rec)))

        try:
            r_val, _ = pearsonr(comp_orig, comp_rec)
            r_val = float(r_val) if np.isfinite(r_val) else 0.0
        except Exception:
            r_val = 0.0

        return {
            'MAE': mae,
            'RMSE': rmse,
            'Pearson_r': r_val
        }

    def _compare_distributions(
        self,
        original: np.ndarray,
        recovered: np.ndarray
    ) -> Dict[str, float]:
        """Compute Jensen-Shannon Divergence and Kolmogorov-Smirnov Test statistic."""
        stats = {}
        
        # Jensen-Shannon Divergence via histograms
        try:
            p, _ = np.histogram(original, bins=50, density=True)
            q, _ = np.histogram(recovered, bins=50, density=True)
            p += 1e-8
            q += 1e-8
            js_div = float(jensenshannon(p, q))
        except Exception:
            js_div = np.nan

        # Kolmogorov-Smirnov Test
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ks_res = ks_2samp(original, recovered)
                ks_stat = float(ks_res.statistic)
        except Exception:
            ks_stat = np.nan

        stats['Jensen-Shannon'] = js_div
        stats['KS_Statistic'] = ks_stat
        return stats

    def generate_report(self) -> pd.DataFrame:
        """Compile evaluation dictionary into a structured pandas DataFrame report."""
        rows = []
        for method, metrics in self.results.items():
            row = {'Method': method}
            row.update(metrics.get('reconstruction', {}))
            row.update(metrics.get('statistics', {}))
            rows.append(row)
        return pd.DataFrame(rows)

def visualize_results_jittered(
    df: pd.DataFrame,
    dataset_name: str = "Benchmark",
    feature: str = "Feature",
    jitter_strength: float = 0.0019
) -> plt.Figure:
    """
    Plot correlation trends vs Sigma and Outlier Percentage.
    Applies subtle jittering to FID variants to eliminate line overlap.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{dataset_name} - Feature: {feature}", fontsize=14, y=1.02)

    df_plot = df.copy()
    if 'method' in df_plot.columns and 'correlation' in df_plot.columns:
        df_plot['correlation_jittered'] = df_plot['correlation'] + df_plot['method'].apply(
            lambda m: np.random.uniform(-jitter_strength, jitter_strength) if 'FID' in str(m) and str(m) != 'FID' else 0
        )
    else:
        df_plot['correlation_jittered'] = df_plot.get('correlation', 0.0)

    # Subplot 1: Sigma vs Correlation
    if 'sigma' in df_plot.columns:
        sns.lineplot(
            data=df_plot,
            x='sigma',
            y='correlation_jittered',
            hue='method',
            marker='o',
            linewidth=2,
            ax=axes[0]
        )
        axes[0].set_title('Sigma (Noise Scale) vs Correlation')
        axes[0].set_xlabel('Sigma (σ)')
        axes[0].set_ylabel('Pearson Correlation (r)')
        axes[0].grid(True, alpha=0.3)

    # Subplot 2: Outlier Percentage vs Correlation
    if 'outlier_percentage' in df_plot.columns:
        sns.lineplot(
            data=df_plot,
            x='outlier_percentage',
            y='correlation_jittered',
            hue='method',
            marker='s',
            linewidth=2,
            ax=axes[1]
        )
        axes[1].set_title('Outlier Percentage vs Correlation')
        axes[1].set_xlabel('Outlier Percentage (OP %)')
        axes[1].set_ylabel('Pearson Correlation (r)')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_recovery_comparison(
    original: np.ndarray,
    recovered: np.ndarray,
    compromised_idx: np.ndarray,
    method_name: str = "FID Recovery"
) -> plt.Figure:
    """Plot time-series trajectory and distribution histogram overlay."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 1. Time Series
    axes[0].plot(original, label='Original Data', color='black', alpha=0.7)
    axes[0].plot(recovered, label=f'Recovered ({method_name})', color='tab:blue', linestyle='--')
    axes[0].scatter(compromised_idx, original[compromised_idx], color='red', label='Compromised Noise Injected', s=25, zorder=5)
    axes[0].set_title(f'Time Series Data Recovery: {method_name}')
    axes[0].set_ylabel('Feature Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Distribution Density
    sns.kdeplot(original, ax=axes[1], label='Original Density', color='black', fill=True, alpha=0.2)
    sns.kdeplot(recovered, ax=axes[1], label=f'Recovered Density ({method_name})', color='tab:blue', fill=True, alpha=0.2)
    axes[1].set_title('Probability Density Overlay')
    axes[1].set_xlabel('Feature Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
