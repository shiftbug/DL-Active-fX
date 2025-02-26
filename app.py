# app.py
# Main Streamlit application for Deep Learning Activation Function Comparison

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple, Optional

# Import from modules
from modules.data import generate_random_data, get_relationship_descriptions
from modules.models import get_available_activations
from modules.experiment import run_experiments, create_summary_dataframes, create_summary_table
from modules.visualization import (
    plot_learning_curves, plot_predictions_vs_actual, plot_batch_losses,
    create_heatmaps, plot_activation_functions
)
from modules.utils import (
    get_system_info, configure_gpu, get_csv_download_link, 
    get_pickle_download_link, load_experiment_results,
    estimate_experiment_time, parse_hidden_layers
)

# Set page config
st.set_page_config(
    page_title="Deep Learning Activation Function Comparison",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

if 'experiment_params' not in st.session_state:
    st.session_state.experiment_params = None

def main():
    """Main application function"""
    st.title("🧠 Deep Learning Activation Function Comparison")
    
    st.markdown("""
    This app allows you to experiment with different activation functions across various data distributions and 
    neural network architectures. Customize parameters, run experiments, and visualize detailed results to better 
    understand how activation functions impact model performance.
    """)
    
    # Sidebar for system info and settings
    sidebar_section()
    
    # Create tabs
    tabs = st.tabs([
        "Experiment Setup", 
        "Results Analysis", 
        "Learning Visualization", 
        "Activation Functions",
        "Load Previous Results"
    ])
    
    # Fill each tab with content
    with tabs[0]:
        experiment_setup_tab()
    
    with tabs[1]:
        results_analysis_tab()
    
    with tabs[2]:
        learning_visualization_tab()
    
    with tabs[3]:
        activation_functions_tab()
    
    with tabs[4]:
        load_results_tab()

def sidebar_section():
    """Sidebar with system information and app settings"""
    st.sidebar.header("System Information")
    
    # Get system info
    system_info = get_system_info()
    
    # Display system info
    for key, value in system_info.items():
        st.sidebar.text(f"{key}: {value}")
    
    st.sidebar.markdown("---")
    
    # App settings
    st.sidebar.header("App Settings")
    
    # Theme selection
    theme = st.sidebar.selectbox(
        "Color Theme",
        ["Default", "Dark", "Light"],
        index=0
    )
    
    # Apply theme (placeholder for now)
    if theme != "Default":
        st.sidebar.info(f"{theme} theme selected (customization coming soon)")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This app helps compare different activation functions for deep learning 
    regression tasks. It generates synthetic data with various relationships
    and tests different activation functions across multiple neural network
    configurations.
    
    Created with Streamlit and TensorFlow.
    """)

def experiment_setup_tab():
    """Content for experiment setup tab"""
    st.header("Experiment Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Generation Settings")
        
        random_methods = st.multiselect(
            "Random Data Methods",
            ["gaussian", "uniform", "exponential", "beta", "gamma"],
            default=["gaussian", "uniform", "exponential"]
        )
        
        n_samples = st.number_input("Number of samples", min_value=100, max_value=10000, value=1000, step=100)
        n_features = st.number_input("Number of features", min_value=2, max_value=50, value=10, step=1)
        n_outputs = st.number_input("Number of outputs", min_value=1, max_value=5, value=3, step=1)
        
        with st.expander("Output Relationship Types"):
            st.markdown("This app generates data with the following relationship types:")
            relationship_descriptions = get_relationship_descriptions(5)
            for i, desc in enumerate(relationship_descriptions[:n_outputs]):
                st.markdown(f"- **Output {i+1}**: {desc}")
        
        noise_level = st.slider("Noise level", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
        test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
        
    with col2:
        st.subheader("Model Settings")
        
        # Get available activations with descriptions
        available_activations = get_available_activations()
        activation_options = list(available_activations.keys())
        
        # Display activation functions with tooltips
        st.markdown("**Activation Functions**")
        activation_cols = st.columns(2)
        
        activations = []
        for i, act_name in enumerate(activation_options):
            col_idx = i % 2
            with activation_cols[col_idx]:
                if st.checkbox(act_name, value=(act_name in ['relu', 'tanh', 'sigmoid', 'elu', 'selu']), 
                               help=available_activations[act_name]):
                    activations.append(act_name)
        
        if not activations:
            st.warning("Please select at least one activation function.")
            activations = ['relu']  # Default if none selected
        
        # Allow customization of hidden layers
        hidden_layers_str = st.text_input("Hidden Layers (comma-separated)", "64, 32, 16")
        hidden_layers = parse_hidden_layers(hidden_layers_str)
        
        dropout_rate = st.slider("Dropout rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
        learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f", step=0.0001)
        
    st.subheader("Experiment Configuration")
    
    col3, col4 = st.columns(2)
    
    with col3:
        n_experiments = st.number_input("Number of experiments per combination", min_value=1, max_value=100, value=3, step=1)
        epochs = st.number_input("Training epochs", min_value=10, max_value=500, value=50, step=10)
        
    with col4:
        batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=32, step=8)
    
    # Calculate total number of experiments
    total_experiments = len(activations) * len(random_methods) * n_experiments
    st.info(f"Total experiments to run: {total_experiments}")
    
    # Estimate time
    estimated_time, time_str = estimate_experiment_time(
        activations, random_methods, n_experiments, epochs, n_samples
    )
    st.info(f"Estimated running time: {time_str}")
    
    # Run button
    run_button = st.button("Run Experiments")
    
    if run_button:
        # Set up progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress callback
        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress)
            status_text.info(f"Running experiment {current}/{total}...")
        
        # Configure GPU
        configure_gpu()
        
        # Run experiments
        start_time = time.time()
        
        try:
            results = run_experiments(
                activations=activations,
                random_methods=random_methods,
                n_experiments=n_experiments,
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
                progress_callback=update_progress
            )
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.experiment_params = {
                'activations': activations,
                'random_methods': random_methods,
                'n_experiments': n_experiments,
                'n_samples': n_samples,
                'n_features': n_features,
                'n_outputs': n_outputs,
                'hidden_layers': hidden_layers,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'noise_level': noise_level,
                'test_size': test_size,
                'epochs': epochs,
                'batch_size': batch_size,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            status_text.success(f"Experiments completed in {total_time:.2f} seconds!")
            
            # Automatically switch to results tab
            st.rerun()
            
        except Exception as e:
            status_text.error(f"Error running experiments: {str(e)}")

def results_analysis_tab():
    """Content for results analysis tab"""
    st.header("Results Analysis")
    
    if 'results' not in st.session_state or st.session_state.results is None:
        st.info("Run experiments first or load previous results to see analysis.")
        return
    
    if 'experiment_params' not in st.session_state or st.session_state.experiment_params is None:
        st.warning("Experiment parameters are missing. Please run experiments again or reload your results file.")
        return
    
    # Display debug info (for troubleshooting)
    with st.expander("Debug Info (Results Structure)", expanded=False):
        st.write("Results keys:", list(st.session_state.results.keys()) if isinstance(st.session_state.results, dict) else "Not a dictionary")
        st.write("Params keys:", list(st.session_state.experiment_params.keys()) if isinstance(st.session_state.experiment_params, dict) else "Not a dictionary")
    
    try:
        results = st.session_state.results
        params = st.session_state.experiment_params
        
        # Ensure required parameters exist
        if 'n_outputs' not in params:
            st.error("Missing required parameter: n_outputs")
            return
            
        n_outputs = params['n_outputs']
        
        # Create summary dataframes
        comparison_dfs, cross_method_df = create_summary_dataframes(results, n_outputs)
        
        # Display experiment parameters
        with st.expander("Experiment Parameters"):
            # Convert lists to strings to avoid Arrow errors
            formatted_params = {}
            for k, v in params.items():
                if k != 'timestamp':
                    if isinstance(v, list):
                        formatted_params[k] = [', '.join(map(str, v))]
                    else:
                        formatted_params[k] = [str(v)]
                        
            param_df = pd.DataFrame(formatted_params)
            st.write(param_df.T)
            if 'timestamp' in params:
                st.write(f"Experiment run at: {params['timestamp']}")
        
        # Display cross-method comparison
        st.subheader("Cross-Method Comparison")
        if cross_method_df.empty:
            st.warning("No valid results data to display.")
            return
            
        st.dataframe(cross_method_df)
        
        # Create heatmaps
        st.subheader("Performance Heatmaps")
        fig_mse, fig_improvement, fig_time = create_heatmaps(cross_method_df, results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_mse)
        with col2:
            st.pyplot(fig_improvement)
        
        st.pyplot(fig_time)
        
        # Create summary table
        st.subheader("Summary of Best Activations")
        summary_df = create_summary_table(results, n_outputs)
        st.dataframe(summary_df)
        
        # Display per-method results
        st.subheader("Detailed Results by Data Distribution")
        
        for random_method in comparison_dfs:
            with st.expander(f"{random_method.capitalize()} Data Results"):
                st.dataframe(comparison_dfs[random_method])
        
        # Download options
        st.subheader("Download Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(get_csv_download_link(cross_method_df, "cross_method_comparison"), unsafe_allow_html=True)
        with col2:
            st.markdown(get_csv_download_link(summary_df, "activation_comparison_summary"), unsafe_allow_html=True)
        with col3:
            download_data = {
                'results': results,
                'experiment_params': params
            }
            st.markdown(get_pickle_download_link(download_data, "experiment_results"), unsafe_allow_html=True)
        
        for random_method in comparison_dfs:
            st.markdown(get_csv_download_link(comparison_dfs[random_method], f"activation_comparison_{random_method}"), unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error analyzing results: {str(e)}")
        st.info("Try running experiments again or loading a different results file.")

def learning_visualization_tab():
    """Content for learning visualization tab"""
    st.header("Learning Visualization")
    
    if 'results' not in st.session_state or st.session_state.results is None:
        st.info("Run experiments first or load previous results to see visualizations.")
        return
        
    if 'experiment_params' not in st.session_state or st.session_state.experiment_params is None:
        st.warning("Experiment parameters are missing. Please run experiments again or reload your results file.")
        return
    
    try:
        results = st.session_state.results
        params = st.session_state.experiment_params
        
        # Ensure results is a properly formatted dictionary
        if not isinstance(results, dict) or len(results) == 0:
            st.error("Results data is not in the correct format.")
            return
        
        # Get available random methods and activations from results
        available_random_methods = list(results.keys())
        if not available_random_methods:
            st.error("No random methods found in results.")
            return
            
        random_method = available_random_methods[0]  # Default value
        
        available_activations = list(results[random_method].keys())
        if not available_activations:
            st.error(f"No activation functions found for {random_method}.")
            return
            
        activation = available_activations[0]  # Default value
        
        # Calculate max experiment index
        max_exp_idx = max(0, len(results[random_method][activation]) - 1) if results[random_method][activation] else 0
        
        # Get output dimension
        n_outputs = params.get('n_outputs', 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            random_method = st.selectbox("Select Data Distribution", available_random_methods)
        
        # Update activations based on selected random method
        available_activations = list(results[random_method].keys())
        
        with col2:
            activation = st.selectbox("Select Activation Function", available_activations)
        
        # Recalculate max experiment index based on selection
        max_exp_idx = max(0, len(results[random_method][activation]) - 1)
        
        col3, col4 = st.columns(2)
        
        with col3:
            experiment_idx = st.number_input("Experiment Index", min_value=0, max_value=max_exp_idx, value=0, step=1)
        
        with col4:
            output_idx = st.number_input("Output Index", min_value=0, max_value=n_outputs-1, value=0, step=1)
        
        # Learning curves
        st.subheader("Learning Curves")
        fig_learning = plot_learning_curves(results, activation, random_method, experiment_idx)
        st.pyplot(fig_learning)
        
        # Predictions vs Actual
        st.subheader("Predictions vs Actual Values")
        fig_predictions = plot_predictions_vs_actual(
            results, activation, random_method, experiment_idx, output_idx
        )
        if fig_predictions:
            st.pyplot(fig_predictions)
        else:
            st.info("No prediction data available for this experiment.")
        
        # Batch Losses
        st.subheader("Batch-level Learning Dynamics")
        moving_avg_window = st.slider("Moving Average Window", min_value=5, max_value=100, value=50, step=5)
        fig_batch = plot_batch_losses(
            results, activation, random_method, experiment_idx, moving_avg_window
        )
        if fig_batch:
            st.pyplot(fig_batch)
        else:
            st.info("No batch loss data available for this experiment.")
        
        # Show improvement over baseline
        if 'error' not in results[random_method][activation][experiment_idx]:
            metrics = results[random_method][activation][experiment_idx]
            improvement = metrics['improvement_over_baseline'] * 100
            
            st.metric("Improvement Over Baseline", f"{improvement:.2f}%", 
                    delta=f"{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%")
            
            # Display baseline metrics
            st.info(f"Baseline MSE (mean prediction): {metrics['baseline_mse']:.4f}")
        else:
            st.error(f"Error in experiment: {results[random_method][activation][experiment_idx]['error']}")
            
    except Exception as e:
        st.error(f"Error visualizing learning curves: {str(e)}")
        st.info("Try running experiments again or loading a different results file.")

def activation_functions_tab():
    """Content for activation functions information tab"""
    st.header("Activation Functions")
    
    st.markdown("""
    Activation functions are a crucial component of neural networks. They introduce non-linearity, 
    allowing neural networks to learn complex patterns. Different activation functions have different 
    properties that may make them more suitable for certain tasks.
    """)
    
    # Plot activation functions
    fig = plot_activation_functions()
    st.pyplot(fig)
    
    # Information about activation functions
    st.subheader("Activation Function Characteristics")
    
    activations_info = get_available_activations()
    
    info_cols = st.columns(2)
    
    for i, (name, desc) in enumerate(activations_info.items()):
        col_idx = i % 2
        with info_cols[col_idx]:
            st.markdown(f"**{name}**")
            st.markdown(desc)
            
            # Add specific characteristics
            if name == 'relu':
                st.markdown("- Fast computation")
                st.markdown("- May suffer from 'dying ReLU' problem")
                st.markdown("- No upper bound on activation")
            elif name == 'sigmoid':
                st.markdown("- Outputs between 0 and 1")
                st.markdown("- Can cause vanishing gradient problem")
                st.markdown("- Historically used in output layers for binary classification")
            elif name == 'tanh':
                st.markdown("- Outputs between -1 and 1")
                st.markdown("- Zero-centered, helping optimization")
                st.markdown("- Can still suffer from vanishing gradient")
            elif name == 'elu':
                st.markdown("- Addresses dying ReLU problem")
                st.markdown("- Smoother gradient near zero")
                st.markdown("- Computationally more expensive than ReLU")
            elif name == 'selu':
                st.markdown("- Self-normalizing properties")
                st.markdown("- Good for deep networks")
                st.markdown("- Requires specific initialization")
            elif name == 'swish':
                st.markdown("- Smooth activation function")
                st.markdown("- Performs well in deeper models")
                st.markdown("- Introduced by Google Research")
            st.markdown("---")

def load_results_tab():
    """Content for loading previous results tab"""
    st.header("Load Previous Results")
    
    uploaded_file = st.file_uploader("Upload experiment results (.pickle file)", type="pickle")
    
    if uploaded_file is not None:
        # Load results
        loaded_results = load_experiment_results(uploaded_file)
        
        if loaded_results is not None:
            try:
                # Check if the loaded data has the expected structure
                if isinstance(loaded_results, dict) and 'results' in loaded_results and 'experiment_params' in loaded_results:
                    # Extract results and parameters
                    st.session_state.results = loaded_results['results']
                    
                    # Ensure parameters are DataFrame-friendly
                    experiment_params = loaded_results['experiment_params']
                    # Convert any lists in parameters to ensure they display correctly in DataFrames
                    for key, value in experiment_params.items():
                        if isinstance(value, list) and key != 'activations' and key != 'random_methods':
                            experiment_params[key] = str(value)
                            
                    st.session_state.experiment_params = experiment_params
                    
                    st.success("Results loaded successfully! Go to the Results Analysis or Learning Visualization tabs to explore.")
                    # Add a button to navigate to the Results Analysis tab
                    if st.button("Go to Results Analysis"):
                        st.rerun()
                        
                # Handle older format where the result is just the results dictionary without experiment_params
                elif isinstance(loaded_results, dict) and 'experiment_params' not in loaded_results:
                    st.warning("Experiment parameters not found in uploaded file. Please enter them manually.")
                    
                    n_outputs = st.number_input("Number of outputs in experiment", min_value=1, max_value=5, value=3, step=1)
                    activations = st.multiselect(
                        "Activation Functions used",
                        ["relu", "tanh", "sigmoid", "elu", "selu", "gelu", "swish", "mish", "linear"],
                        default=["relu", "tanh", "sigmoid", "elu", "selu"]
                    )
                    random_methods = st.multiselect(
                        "Random Data Methods used",
                        ["gaussian", "uniform", "exponential", "beta", "gamma"],
                        default=["gaussian", "uniform", "exponential"]
                    )
                    
                    params = {
                        'activations': activations,
                        'random_methods': random_methods,
                        'n_outputs': n_outputs,
                        'timestamp': 'Loaded from file (no timestamp available)'
                    }
                    
                    # Ensure parameters are DataFrame-friendly
                    for key, value in params.items():
                        if isinstance(value, list):
                            params[key] = [str(item) for item in value]
                            
                    # Store results and parameters
                    st.session_state.results = loaded_results
                    st.session_state.experiment_params = params
                    
                    st.success("Results loaded successfully! Go to the Results Analysis or Learning Visualization tabs to explore.")
                    # Add a button to navigate to the Results Analysis tab
                    if st.button("Go to Results Analysis"):
                        st.rerun()
                else:
                    st.error("The uploaded file does not contain valid experiment results.")
            except Exception as e:
                st.error(f"Error processing results: {str(e)}")
                st.session_state.results = None
                st.session_state.experiment_params = None

if __name__ == "__main__":
    main()