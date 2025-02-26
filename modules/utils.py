# modules/utils.py
# Utility functions for the application

import pickle
import base64
import platform
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import io
from typing import Dict, Any, List, Optional, Union, Tuple

def get_system_info() -> Dict[str, str]:
    """
    Collect system information including OS, processor, Python and TensorFlow versions.
    
    Returns:
    --------
    info : Dict[str, str]
        Dictionary containing system information
    """
    info = {
        "Platform": f"{platform.system()} {platform.release()}",
        "Processor": platform.processor() or platform.machine(),
        "Python version": platform.python_version(),
        "TensorFlow version": tf.__version__,
        "NumPy version": np.__version__
    }
    
    # Check if running on macOS with Apple Silicon
    is_mac_m_series = platform.system() == 'Darwin' and platform.machine().startswith('arm')
    if is_mac_m_series:
        info["Apple Silicon"] = "Yes"
        
        # Check if Metal GPU is available
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            info["Metal GPU"] = "Available"
        else:
            info["Metal GPU"] = "Not available"
    
    # Check for CUDA GPU
    if platform.system() != 'Darwin':
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            gpu_details = []
            for device in physical_devices:
                device_details = tf.config.experimental.get_device_details(device)
                if device_details and 'device_name' in device_details:
                    gpu_details.append(device_details['device_name'])
            
            if gpu_details:
                info["GPU"] = ", ".join(gpu_details)
            else:
                info["GPU"] = f"Available ({len(physical_devices)})"
        else:
            info["GPU"] = "Not available"
    
    return info

def configure_gpu() -> None:
    """
    Configure GPU settings for optimal performance.
    """
    try:
        # Configure for M-series Macs if available
        if platform.system() == 'Darwin' and platform.machine().startswith('arm'):
            os.environ['TF_METAL_ENABLED'] = '1'
        
        # Configure memory growth on all available GPUs
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Configured {len(physical_devices)} GPU(s) for memory growth")
    except Exception as e:
        print(f"Error configuring GPU: {str(e)}")

def get_csv_download_link(df: pd.DataFrame, filename: str) -> str:
    """
    Generate a download link for a DataFrame as a CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to download
    filename : str
        Filename without extension
        
    Returns:
    --------
    href : str
        HTML link for downloading
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {filename}.csv</a>'
    return href

def get_pickle_download_link(data: Any, filename: str) -> str:
    """
    Generate a download link for an object as a pickle file.
    
    Parameters:
    -----------
    data : Any
        Data to download
    filename : str
        Filename without extension
        
    Returns:
    --------
    href : str
        HTML link for downloading
    """
    pickle_byte_obj = pickle.dumps(data)
    b64 = base64.b64encode(pickle_byte_obj).decode()
    href = f'<a href="data:file/pickle;base64,{b64}" download="{filename}.pickle">Download {filename}.pickle</a>'
    return href

def save_experiment_results(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    experiment_params: Dict[str, Any],
    filename: str
) -> None:
    """
    Save experiment results to a pickle file.
    
    Parameters:
    -----------
    results : Dict
        Experiment results
    experiment_params : Dict
        Parameters used for the experiment
    filename : str
        Filename to save to
    """
    data = {
        'results': results,
        'experiment_params': experiment_params
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_experiment_results(file_obj: Union[str, io.BytesIO]) -> Optional[Dict[str, Any]]:
    """
    Load experiment results from a pickle file.
    
    Parameters:
    -----------
    file_obj : Union[str, io.BytesIO]
        File path or uploaded file object
        
    Returns:
    --------
    data : Optional[Dict[str, Any]]
        Loaded data or None if loading failed
    """
    try:
        if isinstance(file_obj, str):
            with open(file_obj, 'rb') as f:
                data = pickle.load(f)
        else:
            data = pickle.load(file_obj)
        return data
    except Exception as e:
        print(f"Error loading experiment results: {e}")
        return None

def estimate_experiment_time(
    activations: List[str],
    random_methods: List[str],
    n_experiments: int,
    epochs: int,
    n_samples: int
) -> Tuple[float, str]:
    """
    Estimate the time required to run experiments.
    
    Parameters:
    -----------
    activations : List[str]
        List of activation functions
    random_methods : List[str]
        List of random data generation methods
    n_experiments : int
        Number of experiments per combination
    epochs : int
        Number of training epochs
    n_samples : int
        Number of data samples
        
    Returns:
    --------
    estimated_time : float
        Estimated time in seconds
    time_str : str
        Formatted time string
    """
    # Basic time estimate based on empirical testing
    # These are rough estimates and will vary by hardware
    base_time_per_epoch = 0.2  # seconds per epoch for 1000 samples
    sample_factor = n_samples / 1000
    
    # Calculate total number of experiments
    total_experiments = len(activations) * len(random_methods) * n_experiments
    
    # Estimate total time
    estimated_time = total_experiments * epochs * base_time_per_epoch * sample_factor
    
    # Format time string
    if estimated_time < 60:
        time_str = f"{estimated_time:.1f} seconds"
    elif estimated_time < 3600:
        minutes = estimated_time / 60
        time_str = f"{minutes:.1f} minutes"
    else:
        hours = estimated_time / 3600
        time_str = f"{hours:.1f} hours"
    
    return estimated_time, time_str

def parse_hidden_layers(hidden_layers_str: str) -> List[int]:
    """
    Parse a comma-separated string of hidden layer sizes.
    
    Parameters:
    -----------
    hidden_layers_str : str
        Comma-separated string of integers
        
    Returns:
    --------
    hidden_layers : List[int]
        List of hidden layer sizes
    """
    try:
        return [int(x.strip()) for x in hidden_layers_str.split(',')]
    except ValueError:
        # Default if parsing fails
        return [64, 32, 16]