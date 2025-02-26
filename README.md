# Deep Learning Activation Function Comparison

A Streamlit application for comparing the performance of different activation functions across various data distributions for regression tasks.

## 🧠 Overview

This application allows you to experiment with different activation functions across various data distributions and neural network architectures. It generates synthetic data with various non-linear relationships and tests how different activation functions perform on these datasets.

## ✨ Features

- **Data Generation**: Create synthetic data with various distributions (Gaussian, uniform, exponential, etc.) and non-linear relationships
- **Model Comparison**: Test multiple activation functions (ReLU, Tanh, Sigmoid, ELU, SELU, etc.) on the generated data
- **Detailed Analysis**: View comprehensive results including MSE, R², improvement over baseline, and training time
- **Visualizations**: Explore learning curves, predictions vs actual values, and batch-level learning dynamics
- **Result Export**: Download experiment results and analysis for further exploration

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Required packages (installed automatically by the launcher):
  - streamlit
  - tensorflow
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

### Installation & Running

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/dl-activation-comparison.git
   cd dl-activation-comparison
   ```

2. Run the application:
   ```
   python run.py
   ```

The launcher will automatically check for and install any missing dependencies, then start the Streamlit application in your default web browser.

## 📊 Using the Application

### Step 1: Configure Experiment

In the Experiment Setup tab:
1. Select the data generation methods you want to test
2. Choose the activation functions you want to compare
3. Configure model architecture (hidden layers, dropout rate, etc.)
4. Set the number of experiments to run for each combination

### Step 2: Run Experiments

Click the "Run Experiments" button to start the training process. A progress bar will show the status of your experiments.

### Step 3: Analyze Results

Navigate to the Results Analysis tab to view:
- Cross-method comparison table
- Performance heatmaps (MSE, improvement over baseline, training time)
- Summary of best activations by data distribution and output type

### Step 4: Visualize Learning

The Learning Visualization tab allows you to:
- Examine learning curves for specific experiments
- Compare predictions vs actual values
- Explore batch-level learning dynamics

### Step 5: Export Results

Download your results in various formats:
- CSV files for tables and summaries
- Pickle files for complete experiment data that can be reloaded later

## 📋 Project Structure

```
dl_activation_comparison/
│
├── run.py                # Launcher script
├── app.py                # Main Streamlit application
│
├── modules/
│   ├── __init__.py       # Makes modules directory a package
│   ├── data.py           # Data generation functions
│   ├── models.py         # Model creation and training
│   ├── experiment.py     # Experiment orchestration
│   ├── visualization.py  # Plotting and visualization
│   └── utils.py          # Helper functions and utilities
│
└── README.md             # Project documentation
```

## 📝 Notes

- For Apple Silicon Macs, the application will automatically configure TensorFlow Metal for GPU acceleration
- Large experiments with many combinations may take a significant amount of time to run
- For reproducibility, experiment seeds are based on the experiment ID

## 🧩 Extending the Application

- Add new activation functions in `modules/models.py`
- Implement new data generation methods in `modules/data.py`
- Create additional visualizations in `modules/visualization.py`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.# DL-Active-fX
