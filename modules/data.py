# modules/data.py
# Functions for generating synthetic data for experiments

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict, Union, Any

def generate_random_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_outputs: int = 3,
    random_method: str = 'gaussian',
    noise: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for regression tasks with multiple outputs.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of input features to generate
    n_outputs : int
        Number of output variables to generate
    random_method : str
        Method to generate random data: 'gaussian', 'uniform', 'exponential', 'beta', 'gamma'
    noise : float
        Amount of noise to add to the outputs
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target matrix of shape (n_samples, n_outputs)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate features using different random distributions
    if random_method == 'gaussian':
        X = np.random.randn(n_samples, n_features)
    elif random_method == 'uniform':
        X = np.random.uniform(-2, 2, size=(n_samples, n_features))
    elif random_method == 'exponential':
        X = np.random.exponential(scale=1.0, size=(n_samples, n_features))
    elif random_method == 'beta':
        X = np.random.beta(2, 5, size=(n_samples, n_features))
    elif random_method == 'gamma':
        X = np.random.gamma(2, 2, size=(n_samples, n_features))
    else:
        raise ValueError(f"Unknown random method: {random_method}. Choose from: 'gaussian', 'uniform', 'exponential', 'beta', 'gamma'")
    
    # Create multiple outputs with different non-linear relationships
    y = np.zeros((n_samples, n_outputs))
    
    # Output 1: Quadratic relationship
    y[:, 0] = 0.3 * X[:, 0]**2 + 0.7 * X[:, 1] + 0.2 * X[:, 2] * X[:, 3] + noise * np.random.randn(n_samples)
    
    if n_outputs > 1:
        # Output 2: Sinusoidal relationship
        y[:, 1] = 0.5 * np.sin(X[:, 0]) + 0.8 * np.cos(X[:, 1]) - 0.3 * X[:, 4] + noise * np.random.randn(n_samples)
    
    if n_outputs > 2:
        # Output 3: Exponential relationship
        y[:, 2] = 0.2 * np.exp(0.5 * X[:, 2]) + 0.4 * X[:, 5] - 0.1 * X[:, 6]**2 + noise * np.random.randn(n_samples)
    
    # Add more complex relationships for additional outputs
    if n_outputs > 3:
        # Output 4: Logarithmic relationship
        y[:, 3] = 0.4 * np.log(np.abs(X[:, 1]) + 1) + 0.3 * X[:, 7] + 0.2 * np.sqrt(np.abs(X[:, 8])) + noise * np.random.randn(n_samples)
    
    if n_outputs > 4:
        # Output 5: Polynomial relationship
        y[:, 4] = 0.1 * X[:, 0]**3 - 0.2 * X[:, 1]**2 + 0.3 * X[:, 9] - 0.4 + noise * np.random.randn(n_samples)
    
    return X, y

def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    validation_split: bool = False,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Prepare data for model training by normalizing features and splitting into train/test/validation sets.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target matrix
    test_size : float
        Proportion of data to use for testing
    validation_split : bool
        Whether to create a validation set in addition to train and test
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    data_dict : dict
        Dictionary containing the prepared data sets and the scaler
    """
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if validation_split:
        # First split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Then split train+val into train and val
        val_size = test_size / (1 - test_size)  # Adjusted validation size relative to train set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler
        }
    else:
        # Split into train and test only
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler
        }

def get_relationship_descriptions(n_outputs: int) -> List[str]:
    """
    Get descriptions of the relationships used for each output.
    
    Parameters:
    -----------
    n_outputs : int
        Number of outputs
        
    Returns:
    --------
    descriptions : List[str]
        List of descriptions for each output
    """
    descriptions = [
        "Quadratic Relationship (y = a*x₁² + b*x₂ + c*x₃*x₄)",
        "Sinusoidal Relationship (y = a*sin(x₁) + b*cos(x₂) - c*x₅)",
        "Exponential Relationship (y = a*exp(b*x₃) + c*x₆ - d*x₇²)",
        "Logarithmic Relationship (y = a*log(|x₂|+1) + b*x₈ + c*√|x₉|)",
        "Polynomial Relationship (y = a*x₁³ - b*x₂² + c*x₁₀ - d)"
    ]
    
    return descriptions[:n_outputs]

def generate_baseline_predictions(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Generate baseline predictions using the mean values of training data.
    
    Parameters:
    -----------
    y_train : np.ndarray
        Training target values
    y_test : np.ndarray
        Test target values
        
    Returns:
    --------
    y_baseline : np.ndarray
        Baseline predictions for test data
    """
    # Use mean of training data as baseline prediction
    y_mean = np.mean(y_train, axis=0)
    y_baseline = np.full_like(y_test, y_mean)
    
    return y_baseline