# modules/models.py
# Model creation and training functions

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional, Union

class DetailedHistory(Callback):
    """
    Custom callback to capture detailed training progress including:
    - Per-epoch metrics
    - Per-batch metrics
    - Training times
    - Predictions on validation data
    """
    def __init__(self, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        super(DetailedHistory, self).__init__()
        self.validation_data = validation_data
        self.epoch_times = []
        self.batch_times = []
        self.batch_losses = []
        self.detailed_history = {
            'epoch_loss': [],
            'epoch_mae': [],
            'val_loss': [],
            'val_mae': [],
            'epoch_times': [],
            'batch_times': [],
            'batch_losses': [],
            'predictions': [],
            'weights': []
        }
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        # Measure epoch time
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        self.detailed_history['epoch_times'].append(epoch_time)
        
        # Store metrics
        for metric, value in logs.items():
            if metric not in self.detailed_history:
                self.detailed_history[metric] = []
            self.detailed_history[metric].append(value)
            
        # Store model weights (optional - can be memory intensive)
        if epoch % 5 == 0:  # Store weights every 5 epochs to save memory
            weights = []
            for layer in self.model.layers:
                if len(layer.weights) > 0:
                    layer_weights = [w.numpy() for w in layer.weights]
                    weights.append({
                        'name': layer.name,
                        'weights': layer_weights
                    })
            self.detailed_history['weights'].append({
                'epoch': epoch,
                'weights': weights
            })
        
        # Get predictions on validation data
        if self.validation_data and epoch % 5 == 0:  # Every 5 epochs
            x_val, y_val = self.validation_data
            predictions = self.model.predict(x_val, verbose=0)
            self.detailed_history['predictions'].append({
                'epoch': epoch,
                'predictions': predictions,
                'true_values': y_val
            })
    
    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()
        
    def on_batch_end(self, batch, logs=None):
        # Measure batch time
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        self.batch_losses.append(logs.get('loss'))
        
        # Store only the last 1000 batch times and losses to prevent memory issues
        if len(self.batch_times) > 1000:
            self.batch_times = self.batch_times[-1000:]
            self.batch_losses = self.batch_losses[-1000:]
        
        # Periodically update the detailed history
        if batch % 100 == 0:
            self.detailed_history['batch_times'] = self.batch_times.copy()
            self.detailed_history['batch_losses'] = self.batch_losses.copy()

def get_activation_function(activation_name: str) -> Any:
    """
    Get the activation function by name.
    
    Parameters:
    -----------
    activation_name : str
        Name of the activation function
        
    Returns:
    --------
    function or str
        The activation function or its name if it's a built-in function
    """
    # TensorFlow built-in activations
    built_in = ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'linear']
    
    if activation_name in built_in:
        return activation_name
    
    # Custom activations
    if activation_name == 'swish':
        return tf.nn.swish
    elif activation_name == 'gelu':
        return tf.keras.activations.gelu
    elif activation_name == 'mish':
        # Custom implementation of Mish: f(x) = x * tanh(softplus(x))
        def mish(x):
            return x * tf.math.tanh(tf.math.softplus(x))
        return mish
    elif activation_name == 'snake':
        # Snake activation: sin(alpha * x) / (alpha * x)
        def snake(x, alpha=0.5):
            return x + tf.math.sin(alpha * x) / (alpha)
        return snake
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")

def create_model(
    input_dim: int,
    output_dim: int = 3,
    activation: str = 'relu',
    hidden_layers: List[int] = [64, 32, 16],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> Model:
    """
    Create a neural network model using the functional API with customizable architecture.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    output_dim : int
        Number of output variables
    activation : str
        Activation function to use in hidden layers
    hidden_layers : List[int]
        List of hidden layer sizes
    dropout_rate : float
        Dropout rate for regularization
    learning_rate : float
        Learning rate for Adam optimizer
        
    Returns:
    --------
    model : tf.keras.Model
        Compiled model
    """
    # Get activation function
    activation_fn = get_activation_function(activation)
    
    # Create model
    inputs = Input(shape=(input_dim,))
    x = inputs
    
    # Add hidden layers
    for units in hidden_layers:
        x = Dense(units, activation=activation_fn)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
    
    # Output layer always uses linear activation for regression
    outputs = Dense(output_dim, activation='linear')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0
) -> Tuple[Dict[str, List[float]], DetailedHistory]:
    """
    Train a model and return training history.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to train
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation targets
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    verbose : int
        Verbosity level for training
        
    Returns:
    --------
    history : dict
        Training history
    detailed_history : DetailedHistory
        Detailed training metrics
    """
    # Create detailed history callback
    detailed_history = DetailedHistory(validation_data=(X_val, y_val))
    
    # Train model and measure time
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=verbose,
        callbacks=[detailed_history]
    )
    
    # Add training time to history
    training_time = time.time() - start_time
    history.history['training_time'] = training_time
    
    return history.history, detailed_history

def get_available_activations() -> Dict[str, str]:
    """
    Get a dictionary of available activation functions with descriptions.
    
    Returns:
    --------
    activations : Dict[str, str]
        Dictionary mapping activation names to descriptions
    """
    return {
        'relu': 'Rectified Linear Unit: f(x) = max(0, x)',
        'tanh': 'Hyperbolic Tangent: f(x) = tanh(x)',
        'sigmoid': 'Sigmoid: f(x) = 1 / (1 + exp(-x))',
        'elu': 'Exponential Linear Unit: f(x) = x if x > 0 else α * (exp(x) - 1)',
        'selu': 'Scaled ELU: Self-normalizing variant of ELU',
        'gelu': 'Gaussian Error Linear Unit: f(x) = x * Φ(x)',
        'swish': 'Swish: f(x) = x * sigmoid(x)',
        'mish': 'Mish: f(x) = x * tanh(softplus(x))',
        'linear': 'Linear: f(x) = x',
        'snake': 'Snake: f(x) = x + sin(alpha * x) / alpha'
    }