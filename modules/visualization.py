# modules/visualization.py
# Functions for visualizing experiment results

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Any, Tuple, Optional, Union
import io
import base64
from matplotlib.colors import LinearSegmentedColormap

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def plot_learning_curves(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    activation: str,
    random_method: str,
    experiment_idx: int = 0
) -> plt.Figure:
    """
    Plot learning curves (loss and MAE) from experiment results.
    
    Parameters:
    -----------
    results : Dict
        Nested dictionary of results from run_experiments
    activation : str
        Activation function to plot
    random_method : str
        Random method to plot
    experiment_idx : int
        Index of experiment to plot
        
    Returns:
    --------
    fig : plt.Figure
        Figure containing the plots
    """
    metrics = results[random_method][activation][experiment_idx]
    
    if 'error' in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error in experiment: {metrics['error']}", 
                horizontalalignment='center', fontsize=14)
        ax.axis('off')
        return fig
        
    history = metrics['history']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot training and validation loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title(f'Loss Curves - {activation} on {random_method.capitalize()} Data', fontsize=14)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot training and validation MAE
    axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_title(f'MAE Curves - {activation} on {random_method.capitalize()} Data', fontsize=14)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_predictions_vs_actual(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    activation: str,
    random_method: str,
    experiment_idx: int = 0,
    output_idx: int = 0,
    epoch_idx: int = -1
) -> Optional[plt.Figure]:
    """
    Plot predictions vs actual values for a specific output.
    
    Parameters:
    -----------
    results : Dict
        Nested dictionary of results from run_experiments
    activation : str
        Activation function to plot
    random_method : str
        Random method to plot
    experiment_idx : int
        Index of experiment to plot
    output_idx : int
        Index of output variable to plot
    epoch_idx : int
        Index of epoch to use for predictions (-1 for last)
        
    Returns:
    --------
    fig : plt.Figure
        Figure containing the plot
    """
    metrics = results[random_method][activation][experiment_idx]
    
    if 'error' in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error in experiment: {metrics['error']}", 
                horizontalalignment='center', fontsize=14)
        ax.axis('off')
        return fig
        
    detailed_history = metrics['detailed_history']
    
    if not detailed_history['predictions']:
        return None
    
    # Use the last epoch's predictions by default
    epoch_data = detailed_history['predictions'][epoch_idx]
    
    predictions = epoch_data['predictions'][:, output_idx]
    true_values = epoch_data['true_values'][:, output_idx]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the true vs predicted values
    ax.scatter(true_values, predictions, alpha=0.6, s=70)
    
    # Add perfect prediction line
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    lims = [min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val)]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Calculate metrics for this output
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    
    # Add correlation coefficient and MSE to the plot
    ax.text(0.05, 0.95, f'MSE: {mse:.4f}\nR²: {r2:.4f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    relationship_types = ['Quadratic', 'Sinusoidal', 'Exponential', 'Logarithmic', 'Polynomial']
    relationship = relationship_types[output_idx] if output_idx < len(relationship_types) else f"Output {output_idx+1}"
    
    ax.set_xlabel('True Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)
    ax.set_title(f'Predicted vs Actual - {activation} on {random_method.capitalize()} Data\n'
                f'{relationship} Relationship (Epoch {epoch_data["epoch"]})', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_batch_losses(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    activation: str,
    random_method: str,
    experiment_idx: int = 0,
    moving_avg_window: int = 50
) -> Optional[plt.Figure]:
    """
    Plot batch-level losses with moving average.
    
    Parameters:
    -----------
    results : Dict
        Nested dictionary of results from run_experiments
    activation : str
        Activation function to plot
    random_method : str
        Random method to plot
    experiment_idx : int
        Index of experiment to plot
    moving_avg_window : int
        Window size for moving average
        
    Returns:
    --------
    fig : plt.Figure
        Figure containing the plot
    """
    metrics = results[random_method][activation][experiment_idx]
    
    if 'error' in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error in experiment: {metrics['error']}", 
                horizontalalignment='center', fontsize=14)
        ax.axis('off')
        return fig
        
    detailed_history = metrics['detailed_history']
    
    if not detailed_history['batch_losses']:
        return None
    
    # Get batch losses
    batch_losses = detailed_history['batch_losses']
    
    # Calculate moving average
    moving_avg = np.convolve(batch_losses, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot raw losses with low alpha for readability
    ax.plot(batch_losses, 'b-', alpha=0.3, label='Batch Loss', linewidth=1)
    
    # Plot moving average
    ax.plot(np.arange(len(moving_avg)) + moving_avg_window-1, moving_avg, 'r-', 
            label=f'Moving Average (window={moving_avg_window})', linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Batch', fontsize=14)
    ax.set_ylabel('Loss (MSE)', fontsize=14)
    ax.set_title(f'Batch Losses - {activation} on {random_method.capitalize()} Data', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    return fig

def create_heatmaps(
    cross_method_df: pd.DataFrame,
    results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Create heatmaps comparing activations and random methods.
    
    Parameters:
    -----------
    cross_method_df : pd.DataFrame
        Dataframe comparing across different random methods
    results : Dict
        Nested dictionary of results from run_experiments
        
    Returns:
    --------
    fig_mse : plt.Figure
        MSE heatmap
    fig_improvement : plt.Figure
        Improvement heatmap
    fig_time : plt.Figure
        Training time heatmap
    """
    # Prepare data for heatmaps
    heatmap_data_mse = pd.pivot_table(
        cross_method_df,
        values='Mean Overall MSE',
        index='Activation',
        columns='Random Method'
    )
    
    # Create another dataframe for improvement over baseline
    heatmap_data_improvement = pd.pivot_table(
        cross_method_df,
        values='Mean Improvement Over Baseline (%)',
        index='Activation',
        columns='Random Method'
    )
    
    # Create training time data
    training_time_data = []
    for random_method in results:
        for activation in results[random_method]:
            valid_results = [r for r in results[random_method][activation] if 'error' not in r]
            if not valid_results:
                continue
                
            mean_time = np.mean([r['training_time'] for r in valid_results])
            training_time_data.append({
                'Random Method': random_method.capitalize(),
                'Activation': activation,
                'Mean Training Time': mean_time
            })
    
    training_time_df = pd.DataFrame(training_time_data)
    heatmap_data_time = pd.pivot_table(
        training_time_df,
        values='Mean Training Time',
        index='Activation',
        columns='Random Method'
    )
    
    # Create colormap for MSE (lower is better)
    blues_r = sns.color_palette("YlGnBu_r", as_cmap=True)
    
    # Create MSE heatmap
    fig_mse, ax_mse = plt.subplots(figsize=(12, 9))
    sns.heatmap(heatmap_data_mse, annot=True, cmap=blues_r, fmt='.4f', ax=ax_mse)
    ax_mse.set_title('MSE Performance Heatmap: Activation vs. Random Method\n(Lower is Better)', fontsize=16)
    plt.tight_layout()
    
    # Create improvement heatmap
    fig_improvement, ax_improvement = plt.subplots(figsize=(12, 9))
    sns.heatmap(heatmap_data_improvement, annot=True, cmap='YlGnBu', fmt='.1f', ax=ax_improvement)
    ax_improvement.set_title('Improvement Over Baseline (%): Activation vs. Random Method\n(Higher is Better)', fontsize=16)
    plt.tight_layout()
    
    # Create training time heatmap
    fig_time, ax_time = plt.subplots(figsize=(12, 9))
    sns.heatmap(heatmap_data_time, annot=True, cmap='Reds_r', fmt='.2f', ax=ax_time)
    ax_time.set_title('Training Time (seconds): Activation vs. Random Method\n(Lower is Better)', fontsize=16)
    plt.tight_layout()
    
    return fig_mse, fig_improvement, fig_time

def plot_activation_functions() -> plt.Figure:
    """
    Plot common activation functions for visualization.
    
    Returns:
    --------
    fig : plt.Figure
        Figure containing the plots
    """
    x = np.linspace(-5, 5, 1000)
    
    # Define activation functions
    def relu(x):
        return np.maximum(0, x)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def swish(x):
        return x * sigmoid(x)
    
    def mish(x):
        return x * np.tanh(np.log(1 + np.exp(x)))
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot each activation function
    axes[0].plot(x, relu(x), 'b-', linewidth=2)
    axes[0].set_title('ReLU', fontsize=14)
    
    axes[1].plot(x, sigmoid(x), 'g-', linewidth=2)
    axes[1].set_title('Sigmoid', fontsize=14)
    
    axes[2].plot(x, tanh(x), 'r-', linewidth=2)
    axes[2].set_title('Tanh', fontsize=14)
    
    axes[3].plot(x, elu(x), 'c-', linewidth=2)
    axes[3].set_title('ELU', fontsize=14)
    
    axes[4].plot(x, swish(x), 'm-', linewidth=2)
    axes[4].set_title('Swish', fontsize=14)
    
    axes[5].plot(x, mish(x), 'y-', linewidth=2)
    axes[5].set_title('Mish', fontsize=14)
    
    # Add grid and set limits for each subplot
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-2, 5])
    
    plt.tight_layout()
    fig.suptitle('Common Activation Functions', fontsize=18, y=1.02)
    return fig

def get_figure_as_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Parameters:
    -----------
    fig : plt.Figure
        Figure to convert
        
    Returns:
    --------
    base64_str : str
        Base64 encoded string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str