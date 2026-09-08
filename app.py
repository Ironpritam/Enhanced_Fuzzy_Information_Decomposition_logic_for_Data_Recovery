"""
Streamlit Web Dashboard for Enhanced Fuzzy Information Decomposition (FID) Data Recovery.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.fid_engine import FIDRecovery
from src.data.preprocessor import DataPreprocessor, EnhancedDataPreprocessor
from src.evaluation.metrics import RecoveryEvaluator, plot_recovery_comparison, visualize_results_jittered
from main import generate_sample_forestfires

st.set_page_config(
    page_title="Enhanced FID Data Recovery Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for dark glassmorphism styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 25px;
    }
    .card {
        background-color: #1A202C;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2D3748;
        margin-bottom: 20px;
    }
    .badge {
        background-color: #3182CE;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Enhanced Fuzzy Information Decomposition (FID)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hardware-Accelerated Non-Linear Data Recovery & Reconstruction Engine</div>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🎛️ Configuration & Controls")

dataset_source = st.sidebar.radio("Data Source", ["Preset: ForestFires Benchmark", "Upload Custom CSV"])

if dataset_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_name = os.path.splitext(uploaded_file.name)[0]
    else:
        st.sidebar.info("Using synthetic benchmark dataset until file is uploaded.")
        df = generate_sample_forestfires()
        dataset_name = "forestfires"
else:
    if os.path.exists("forestfires.csv"):
        df = pd.read_csv("forestfires.csv")
    else:
        df = generate_sample_forestfires()
    dataset_name = "forestfires"

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.error("No numeric columns found in dataset!")
    st.stop()

default_feat_idx = numeric_cols.index("temp") if "temp" in numeric_cols else 0
sel_feature = st.sidebar.selectbox("Select Target Feature to Noise & Recover", numeric_cols, index=default_feat_idx)

sigma = st.sidebar.slider("Gaussian Noise Scale (σ)", min_value=0.05, max_value=1.0, value=0.3, step=0.05)
op = st.sidebar.slider("Outlier Percentage (OP %)", min_value=1.0, max_value=30.0, value=15.0, step=1.0)

# Hardware status
fid_engine = FIDRecovery()
device_status = fid_engine.device.upper()
st.sidebar.markdown(f"**Hardware Engine Detected:** `<span class='badge'>{device_status}</span>`", unsafe_allow_html=True)

# Preprocess & noise inject
preprocessor = EnhancedDataPreprocessor(
    df=df,
    dataset_name=dataset_name,
    sel_feature=sel_feature,
    sigma=sigma,
    OP=op
)

X_scaled, X_train, X_test, y_train, y_test, features = preprocessor.prepare_data()
feature_index = features.index(sel_feature) if sel_feature in features else 0
compromised_indices = np.where(y_test == 0)[0]
original_data = X_scaled[:, feature_index]

# Main layout tabs
tab1, tab2, tab3 = st.tabs(["🚀 Live Recovery Benchmark", "📈 Visual Analytics & Trajectories", "🧠 Architecture & Theory"])

with tab1:
    st.subheader("Data Recovery Experiment Execution")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(df))
    col2.metric("Target Attribute", sel_feature)
    col3.metric("Noise Scale (σ)", f"{sigma}")
    col4.metric("Compromised Rows", f"{len(compromised_indices)} ({op}%)")

    st.markdown("---")

    if st.button("▶ Run Full FID Recovery Benchmark", type="primary"):
        with st.spinner("Executing GPU/CPU Accelerated FID algorithms..."):
            methods = [
                ("Base FID", None),
                ("FID + Pearson", "pearson"),
                ("FID + Distance", "distance"),
                ("FID + Shallow NN", "shallow_nn")
            ]

            evaluator = RecoveryEvaluator(original_data, feature_name=sel_feature)
            recovered_dict = {}

            for label, m_type in methods:
                rec = fid_engine.FID(
                    org_data=original_data,
                    compromised_indices=compromised_indices,
                    X=X_scaled,
                    target_feature_index=feature_index,
                    method=m_type
                )
                recovered_dict[label] = rec
                evaluator.evaluate_recovery(rec, compromised_indices, label)

            st.session_state['recovered_dict'] = recovered_dict
            st.session_state['evaluator'] = evaluator
            st.session_state['original_data'] = original_data
            st.session_state['compromised_indices'] = compromised_indices

    if 'evaluator' in st.session_state:
        st.success("Recovery Benchmark Completed Successfully!")
        report_df = st.session_state['evaluator'].generate_report()
        
        st.dataframe(report_df.style.highlight_max(axis=0, subset=['Pearson_r']).highlight_min(axis=0, subset=['MAE', 'RMSE', 'Jensen-Shannon']), use_container_width=True)

with tab2:
    st.subheader("Visual Recovery Analytics")
    if 'recovered_dict' in st.session_state:
        selected_method = st.selectbox("Select Method to Visualize Trajectory", list(st.session_state['recovered_dict'].keys()))
        rec_data = st.session_state['recovered_dict'][selected_method]
        
        fig = plot_recovery_comparison(
            original=st.session_state['original_data'],
            recovered=rec_data,
            compromised_idx=st.session_state['compromised_indices'],
            method_name=selected_method
        )
        st.pyplot(fig)
    else:
        st.info("Please run the Recovery Benchmark in Tab 1 first to display visual analytics.")

with tab3:
    st.subheader("Fuzzy Information Decomposition (FID) Theoretical Overview")
    st.markdown("""
    ### Mathematical Formulation
    The **Fuzzy Information Decomposition (FID)** algorithm converts multi-attribute data into fuzzy intervals:
    
    $$V_s = \\frac{(a + (s-1)l) + (a + sl)}{2}, \\quad l = \\frac{b - a}{t}$$
    
    Where:
    - $a, b$: Minimum and Maximum bounds of observed feature values
    - $t$: Count of compromised/outlier indices
    - $l$: Interval step length
    
    Contribution weights for each fuzzy interval $v \\in V$:
    $$w_i = 1 - \\frac{|x_i - v|}{l} \\quad \\text{if } |x_i - v| \\le l \\text{ else } 0$$
    
    In **Enhanced FID**, inter-attribute weights $W_{feat}$ are incorporated using **Pearson Covariance**, **Distance Correlation**, or **Multi-Layer Perceptron (Shallow NN)** regressor dense layer weights.
    """)
