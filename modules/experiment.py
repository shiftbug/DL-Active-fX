# modules/experiment.py
# Functions for conducting experiments and analyzing results

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import time
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import tensorflow as tf

from modules.data import generate_random_data, prepare_data, generate_baseline_predictions
from modules.models import create_model, train_model

def run_single_experiment(
    activation: str,
    random_method: str,
    n_samples: int,
    n_features: int, 
    n_outputs: int,
    hidden_layers: List[int],
    dropout_rate: float,
    learning_rate: float,
    noise_level: float,
    test_size: float,
    epochs: int,
    batch_size: int,
    experiment_id: int,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Run a single experiment with specified parameters.
    
    Parameters:
    -----------
    activation : str
        Activation function to use
    random_method : str
        Method to generate random data
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of input features
    n_outputs : int
        Number of output variables
    hidden_layers : List[int]
        List of hidden layer sizes
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for Adam optimizer
    noise_level : float
        Amount of noise to add to generated data
    test_size : float
        Proportion of data to use for testing
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    experiment_id : int
        Identifier for the experiment (used as random seed)
    verbose : int
        Verbosity level for training
        
    Returns:
    --------
    metrics : Dict[str, Any]
        Dictionary containing experiment results and metrics
    """
    # Generate data
    X, y = generate_random_data(
        n_samples=n_samples,
        n_features=n_features,
        n_outputs=n_outputs,
        random_method=random_method,
        noise=noise_level,
        seed=experiment_id  # Use experiment_id as seed for reproducibility
    )
    
    # Prepare data
    data = prepare_data(X, y, test_size=test_size, random_state=experiment_id)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Create model
    model = create_model(
        input_dim=n_features,
        output_dim=n_outputs,
        activation=activation,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Train model and measure time
    start_time = time.time()
    history, detailed_history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    training_time = time.time() - start_time
    
    # Evaluate on test data
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics for each output
    metrics = {}
    
    # Overall metrics
    metrics['overall_mse'] = mean_squared_error(y_test, y_pred)
    metrics['overall_r2'] = r2_score(y_test, y_pred)
    
    # Per-output metrics
    per_output_metrics = []
    for i in range(n_outputs):
        output_metrics = {
            'mse': mean_squared_error(y_test[:, i], y_pred[:, i]),
            'r2': r2_score(y_test[:, i], y_pred[:, i])
        }
        per_output_metrics.append(output_metrics)
    
    metrics['per_output'] = per_output_metrics
    metrics['training_time'] = training_time
    metrics['history'] = history
    metrics['detailed_history'] = detailed_history.detailed_history
    
    # Compare with baseline (mean prediction)
    y_baseline = generate_baseline_predictions(y_train, y_test)
    metrics['baseline_mse'] = mean_squared_error(y_test, y_baseline)
    metrics['baseline_r2'] = r2_score(y_test, y_baseline)
    
    # Calculate improvement over baseline
    metrics['improvement_over_baseline'] = 1 - (metrics['overall_mse'] / metrics['baseline_mse'])
    
    return metrics

def run_experiments(
    activations: List[str],
    random_methods: List[str],
    n_experiments: int,
    n_samples: int,
    n_features: int,
    n_outputs: int,
    hidden_layers: List[int],
    dropout_rate: float,
    learning_rate: float,
    noise_level: float,
    test_size: float,
    epochs: int,
    batch_size: int,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Run multiple experiments with different combinations of parameters.
    
    Parameters:
    -----------
    activations : List[str]
        List of activation functions to test
    random_methods : List[str]
        List of random data generation methods to test
    n_experiments : int
        Number of experiments to run for each combination
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of input features
    n_outputs : int
        Number of output variables
    hidden_layers : List[int]
        List of hidden layer sizes
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for Adam optimizer
    noise_level : float
        Amount of noise to add to generated data
    test_size : float
        Proportion of data to use for testing
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    progress_callback : Optional[Callable[[int, int], None]]
        Callback function to update progress, takes current and total experiments
        
    Returns:
    --------
    results : Dict[str, Dict[str, List[Dict[str, Any]]]]
        Nested dictionary of results organized by random method and activation
    """
    results = {}
    experiment_counter = 0
    total_experiments = len(activations) * len(random_methods) * n_experiments
    
    # Configure GPU memory growth if available
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass  # No GPU or other issue
    
    for random_method in random_methods:
        results[random_method] = {}
        
        for activation in activations:
            results[random_method][activation] = []
            
            for exp_id in range(n_experiments):
                # Update progress
                experiment_counter += 1
                if progress_callback:
                    progress_callback(experiment_counter, total_experiments)
                
                # Run experiment
                try:
                    metrics = run_single_experiment(
                        activation=activation,
                        random_method=random_method,
                        n_samples=n_samples,
                        n_features=n_features,
                        n_outputs=n_outputs,
                        hidden_layers=hidden_layers,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate,
                        noise_level=noise_level,
                        test_size=test_size,
                        epochs=epochs,
                        batch_size=batch_size,
                        experiment_id=exp_id
                    )
                    
                    results[random_method][activation].append(metrics)
                except Exception as e:
                    # Log error but continue with other experiments
                    print(f"Error in experiment {experiment_counter}/{total_experiments}: {e}")
                    # Add a placeholder with error information
                    results[random_method][activation].append({
                        'error': str(e),
                        'overall_mse': float('nan'),
                        'overall_r2': float('nan'),
                        'improvement_over_baseline': float('nan'),
                        'training_time': 0.0
                    })
    
    return results

def create_summary_dataframes(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    n_outputs: int
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Create summary dataframes from experiment results.
    
    Parameters:
    -----------
    results : Dict
        Nested dictionary of results from run_experiments
    n_outputs : int
        Number of output variables
        
    Returns:
    --------
    comparison_dfs : Dict[str, pd.DataFrame]
        Dictionary of dataframes for each random method
    cross_method_df : pd.DataFrame
        Dataframe comparing across different random methods
    """
    comparison_dfs = {}
    cross_method_data = []
    
    for random_method in results:
        data = []
        
        for activation in results[random_method]:
            # Calculate average metrics across experiments
            exp_results = results[random_method][activation]
            
            # Skip experiments with errors
            valid_results = [r for r in exp_results if 'error' not in r]
            if not valid_results:
                continue
                
            # Make sure all values are scalar (not lists) to avoid Arrow errors
            avg_metrics = {
                'Activation': activation,
                'Overall MSE': float(np.mean([float(r['overall_mse']) for r in valid_results])),
                'Overall R²': float(np.mean([float(r['overall_r2']) for r in valid_results])),
                'Baseline MSE': float(np.mean([float(r['baseline_mse']) for r in valid_results])),
                'Improvement Over Baseline (%)': float(np.mean([float(r['improvement_over_baseline'] * 100) for r in valid_results])),
                'Training Time (s)': float(np.mean([float(r['training_time']) for r in valid_results]))
            }
            
            # Add per-output metrics
            for i in range(n_outputs):
                avg_metrics[f'Output {i+1} MSE'] = float(np.mean([float(r['per_output'][i]['mse']) for r in valid_results]))
                avg_metrics[f'Output {i+1} R²'] = float(np.mean([float(r['per_output'][i]['r2']) for r in valid_results]))
            
            data.append(avg_metrics)
            
            # Add to cross-method data - ensure all values are scalar
            cross_method_data.append({
                'Random Method': str(random_method).capitalize(),
                'Activation': str(activation),
                'Mean Overall MSE': float(avg_metrics['Overall MSE']),
                'Mean Overall R²': float(avg_metrics['Overall R²']),
                'Mean Improvement Over Baseline (%)': float(avg_metrics['Improvement Over Baseline (%)']),
                'Mean Training Time (s)': float(avg_metrics['Training Time (s)'])
            })
        
        # Create dataframe for this random method
        if data:
            df = pd.DataFrame(data)
            # Sort by overall MSE
            df = df.sort_values('Overall MSE')
            comparison_dfs[random_method] = df
    
    # Create cross-method comparison dataframe
    cross_method_df = pd.DataFrame(cross_method_data)
    
    return comparison_dfs, cross_method_df

def create_summary_table(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    n_outputs: int
) -> pd.DataFrame:
    """
    Create a summary table of the best activations for each random method.
    
    Parameters:
    -----------
    results : Dict
        Nested dictionary of results from run_experiments
    n_outputs : int
        Number of output variables
        
    Returns:
    --------
    summary_df : pd.DataFrame
        Summary dataframe
    """
    summary_data = []
    
    for random_method in results:
        # Find best overall activation
        method_data = {}
        for activation in results[random_method]:
            valid_results = [r for r in results[random_method][activation] if 'error' not in r]
            if not valid_results:
                continue
                
            method_data[activation] = {
                'mse': float(np.mean([float(r['overall_mse']) for r in valid_results])),
                'time': float(np.mean([float(r['training_time']) for r in valid_results])),
                'improvement': float(np.mean([float(r['improvement_over_baseline'] * 100) for r in valid_results]))
            }
            
            # Add per-output metrics
            for i in range(n_outputs):
                output_key = f'output_{i}_mse'
                method_data[activation][output_key] = float(np.mean([float(r['per_output'][i]['mse']) for r in valid_results]))
        
        if not method_data:
            continue
            
        # Find best for each metric
        best_overall = min(method_data.items(), key=lambda x: x[1]['mse'])[0]
        fastest = min(method_data.items(), key=lambda x: x[1]['time'])[0]
        best_improvement = max(method_data.items(), key=lambda x: x[1]['improvement'])[0]
        
        # Find best for each output
        best_outputs = []
        for i in range(n_outputs):
            output_key = f'output_{i}_mse'
            best_output_i = min(method_data.items(), key=lambda x: x[1][output_key])[0]
            best_outputs.append(best_output_i)
        
        # Add to summary data - ensure all values are strings or scalar
        row_data = {
            'Random Method': str(random_method).capitalize(),
            'Best Activation Overall': str(best_overall),
            'Best Improvement': str(best_improvement),
            'Fastest Activation': str(fastest)
        }
        
        # Add output-specific data
        for i in range(n_outputs):
            row_data[f'Best for Output {i+1}'] = str(best_outputs[i])
        
        summary_data.append(row_data)
    
    return pd.DataFrame(summary_data)