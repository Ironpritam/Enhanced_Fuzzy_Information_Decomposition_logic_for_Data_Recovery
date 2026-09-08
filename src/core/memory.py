import os
import pickle
import hashlib
import gc
from datetime import datetime
import tensorflow as tf

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model_cache")

def clear_gpu_memory():
    """Clear TensorFlow Keras session and force garbage collection."""
    tf.keras.backend.clear_session()
    gc.collect()

def get_model_hash(dataset_name: str, feature: str, model_type: str, sigma: float, OP: float) -> str:
    """Generate a unique hash for model caching based on training hyperparameters."""
    identifier = f"{dataset_name}_{feature}_{model_type}_sig{sigma}_op{OP}"
    return hashlib.md5(identifier.encode()).hexdigest()

def save_model(model, dataset_name: str, feature: str, model_type: str, sigma: float, OP: float) -> str:
    """Save model checkpoint and metadata into local cache directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_hash = get_model_hash(dataset_name, feature, model_type, sigma, OP)
    
    if hasattr(model, 'save'):
        # Keras model
        filepath = os.path.join(CACHE_DIR, f"{model_hash}.h5")
        model.save(filepath)
    else:
        # Scikit-learn or custom Python model
        filepath = os.path.join(CACHE_DIR, f"{model_hash}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
    # Save metadata
    meta_path = os.path.join(CACHE_DIR, f"{model_hash}.meta")
    meta_data = {
        'dataset_name': dataset_name,
        'feature': feature,
        'model_type': model_type,
        'sigma': sigma,
        'OP': OP,
        'timestamp': datetime.now().isoformat()
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_data, f)
        
    return filepath

def load_model(dataset_name: str, feature: str, model_type: str, sigma: float, OP: float):
    """Load model checkpoint from cache directory if present."""
    model_hash = get_model_hash(dataset_name, feature, model_type, sigma, OP)
    h5_path = os.path.join(CACHE_DIR, f"{model_hash}.h5")
    pkl_path = os.path.join(CACHE_DIR, f"{model_hash}.pkl")
    
    if os.path.exists(h5_path):
        return tf.keras.models.load_model(h5_path)
    elif os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return None

def inspect_model_cache():
    """List all cached models and their metadata."""
    if not os.path.exists(CACHE_DIR):
        return []
    cached = []
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith('.meta'):
            meta_path = os.path.join(CACHE_DIR, fname)
            with open(meta_path, 'rb') as f:
                cached.append(pickle.load(f))
    return cached
