# src/utils/modeling/Neural_Network.py

import torch
import torch.optim as optim
import pandas as pd
import os
import yaml
import json

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from subprocess import PIPE
from datetime import datetime
from utils.tensorboard_util import log_model_histograms, log_training_metrics


class NeuralNetwork(nn.Module):

    def __init__(self, config):
        """
        Initialize the Neural Network with configurable depth, hidden layer size, and input/output sizes.

        Args:
            config (dict): A dictionary containing the hyperparameters:
                - input_size (int): Number of input features.
                - output_size (int): Number of output features.
                - hidden_layer_width (int): Number of neurons in hidden layers.
                - depth (int): Number of hidden layers.
        """

        super(NeuralNetwork, self).__init__()

        layers = []
        input_size = config.get('input_size', 10) # Default input size
        output_size = config.get('output_size', 1) # Default output size
        hidden_layer_width = config.get('hidden_layer_width', 64)
        depth = config.get('depth', 2)

        # Create the hidden layers
        for _ in range(depth):
            layers.append(nn.Linear(input_size, hidden_layer_width))
            layers.append(nn.ReLU())
            input_size = hidden_layer_width

        # Create the output layer
        layers.append(nn.Linear(input_size, output_size))

        # Define the model
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output predictions.
        """

        return self.model(x)

class AirbnbNightlyPriceRegressionDataset(Dataset):

    def __init__(self, csv_file):
        
        self.data = pd.read_csv(csv_file) # Loads the data
        # Assume 'Category' is the column name for the previous label
        
        # Select only numerical columns
        numeric_columns = self.data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        self.data = self.data[numeric_columns]
        #self.features = torch.tensor(self.data.drop('Price_Night', axis=1).values, dtype=torch.float32)
        #self.labels = torch.tensor(self.data['Price_Night'].values, dtype=torch.float32).view(-1, 1)
        self.features = torch.tensor(self.data.drop('beds', axis=1).values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['beds'].values, dtype=torch.float32).view(-1, 1)
        
        print('Label: ', self.data['beds'].name)
        print('Feature Names:', list(self.data.drop('beds', axis=1).columns))

    def __len__(self):
        '''Returns the length of the dataset'''
        
        return len(self.data)

    def __getitem__(self, idx):
        '''Returns a tuple of features and labels for a given index'''
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.features[idx]
        label = self.labels[idx]

        return features, label
   


def create_data_loaders(dataset, train_size=0.8, batch_size=64, shuffle=True, random_seed=42):

    # Calculate the sizes of train, validation and test sets
    train_size = int(train_size * len(dataset))
    validation_size = len(dataset) - train_size

    # Split the dataset into train and validation sets
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(random_seed))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, validation_loader   

'''def get__nn__config(config_file='src/utils/config/nn_config.yaml') -> None:

    with open(config_file, 'r') as file:
        nn_config = yaml.safe_load(file)

    return nn_config'''



class TabularModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def save_model(model, hyperparameters, performance_metrics, folder: str) -> None:
    """Save the trained model, hyperparameters, and performance metrics."""

    # Create directory if it doesnt exist
    os.makedirs(folder, exist_ok=True)

    # Check if the model is a PyTorch module
    if isinstance(model, torch.nn.Module):
        # Save model
        model_filename = os.path.join(folder, 'model.pt')
        torch.save(model.state_dict(), model_filename)

    # Save the hyperparameters
    hyperparameters_filename = os.path.join(folder, 'hyperparameters.json')
    with open(hyperparameters_filename, 'w') as json_file:
        json.dump(hyperparameters, json_file, indent=4)
        
    # Save the performance metrics
    metrics_filename = os.path.join(folder, 'metrics.json')
    with open(metrics_filename, 'w') as json_file:
        json.dump(performance_metrics, json_file, indent=4)

    print(f"Model, hyperparameter and metrics saved to {folder}")

    return None

def generate_nn_configs() -> None:

    '''Generate different configurations for the neural network'''

    configs = []

    # Define the range of hyperparameters to explore
    learning_rates = [0.001, 0.01, 0.1]
    hidden_layer_widths = [32, 64, 128]
    depths = [2, 3]
    optimizers = ['adam', 'sgd']

    # Generate configurations
    for lr in learning_rates:
        for hidden_size in hidden_layer_widths:
            for depth in depths:
                for optimizer in optimizers:
                    config = {
                        'learning_rate': lr,
                        'hidden_layer_width': hidden_size,
                        'depth': depth,
                        'optimizer': optimizer}
                    configs.append(config)

    return configs

def find_best_nn(train_function, validation_loader):
    """
    Train models with different configurations to find the best one.

    Args:
        train_loader (DataLoader): Training data loader.
        validation_loader (DataLoader): Validation data loader.

    Returns:
        tuple: Best model, metrics, and hyperparameters.
    """

    configs = generate_nn_configs()
    best_model, best_metrics, best_hyperparameters = None, None, None
    best_performance = float('inf')

    for config in configs:
        print(f"Training with config: {config}")

        # Train the model with current configuration
        model, metrics, hyperparameters = train_function(train_loader, validation_loader, config)

        # Check if the current model performs better than the previous best model
        if metrics['validation_loss'] < best_performance:
            best_performance = metrics['validation_loss']
            best_model, best_metrics, best_hyperparameters = model, metrics, hyperparameters

    return best_model, best_metrics, best_hyperparameters



class Trainer:
    def __init__(self, model, criterion, optimizer, writer) -> None:
        
        self.model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer # For TensorBoard
        self.total_loss = 0.0 # Shared variables for loss
        self.total_rmse = 0.0 # Shared variables for RMSE

        self.metrics = {
            "train_loss": [],
            "train_rmse": [],
            "validation_loss": [],
            "validation_rmse": []
        }

    def train_model(self, train_loader, validation_loader, num_epochs, device):

        self.model.to(device)

        for epoch in range(num_epochs):

            # Train model for one epoch
            train_loss, train_rmse = self.train_one_epoch(train_loader, device)
            validation_loss = self.validate_one_epoch(validation_loader, device)
            
            # Log weight and bias histograms on TensorBoard
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, epoch)
            self.writer.add_scalar('Training loss', train_loss, epoch)

            # Validate the model after training one epoch
            self.metrics['train_loss'].append(train_loss)
            self.metrics['validation_loss'].append(validation_loss)

            # Append metrics
            self.metrics["train_loss"].append(train_loss)
            self.metrics["train_rmse"].append(train_rmse)

            # Log metrics to TensorBoard
            log_model_histograms(self.writer, self.model, epoch)
            log_training_metrics(self.writer, epoch, train_loss, validation_loss, train_rmse)
            self.writer.add_scalars("Loss", {"Train": train_loss, "Validation": validation_loss}, epoch)
            self.writer.add_scalar("RMSE", train_rmse, epoch)
    
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Validation Loss = {validation_loss:.4}")
        
        return self.metrics

    def initilize_model(self, config):
        """
        Initialize a neural network model, optimizer, and criterion based on the config.

        Args:
            config (dict): Hyperparameter configuration.

        Returns:
            model: Initialized neural network model.
            optimizer: Optimizer object.
            criterion: Loss function.
        """

        model = NeuralNetwork(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) if config['optimizer'] == 'adam' else torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
        criterion = nn.MSELoss()

        return model, optimizer, criterion

    def train_one_epoch(self, train_loader, device):

        self.model.train() # Set model to training mode
        self.metrics["batch_loss"], self.metrics["batch_rmse"] = 0.0, 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = self.model(features)
            
            # Compute the loss
            loss = self.criterion(outputs, labels)
            self.metrics["batch_loss"] += loss.item()

            # Compute RMSE
            rmse = torch.sqrt(torch.mean((outputs - labels) ** 2))
            self.metrics["batch_rmse"] += rmse.item()
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Calculate average training loss for the epoch
        average_loss = self.metrics["batch_loss"] / len(train_loader)
        average_rmse = self.metrics["batch_rmse"] / len(train_loader)

        return average_loss, average_rmse

    def validate_one_epoch(self, validation_loader, device):

        self.model.eval() # Set model to evaluation mode
        total_loss, total_rmse = 0.0, 0.0

        with torch.no_grad(): # No gradients needed for validation
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = self.model(features)
                
                # Compute the loss
                validation_loss = self.criterion(outputs, labels)
                total_loss += validation_loss.item()

                # Compute RMSE
                rmse = torch.sqrt(torch.mean((outputs - labels) ** 2))
                total_rmse += rmse.item()
        
        # Calculate average validation loss for the epoch
        average_loss = total_loss / len(validation_loader)
        average_rmse = total_rmse / len(validation_loader)

        return total_loss / len(validation_loader)
    
    def train_and_save(self, train_loader, validation_loader, config):
        """
        Train a model with given hyperparameters, evaluate it, and save results.

        Args:
            train_loader (DataLoader): Training data loader.
            validation_loader (DataLoader): Validation data loader.
            config (dict): Hyperparameter configuration.

        Returns:
            trained_model: The trained PyTorch model.
            training_hyperparameters: Hyperparameters used for training.
            metrics: Metrics collected during training.
        """

        # Initialize model, optimizer and criterion
        model, optimizer, criterion = self.initilize_model(config)

        # Train the model
        config['num_epochs'] = 50
        config['device'] = 'cpu'  # or 'cuda' if using GPU
        metrics = self.train_model(train_loader, validation_loader, config['num_epochs'] , config['device'] )
        
        # Evaluate the model
        validation_loss, validation_rmse = self.evaluate_model(model, validation_loader, criterion)

        # Log metrics to TensorBoard
        for epoch in range(config['num_epochs']):

            train_loss = metrics["train_loss"][epoch]
            train_rmse = metrics["train_rmse"][epoch]

            self.writer.add_scalars("Loss", {"Train": train_loss, "Validation": validation_loss}, epoch)
            self.writer.add_scalars("RMSE", {"Train": train_rmse, "Validation": validation_rmse}, epoch)
        
        # Log hyperparameters and best metric
        self.writer.add_hparams(hparam_dict=config, metric_dict={"hparam/best_metric": validation_loss})

        # Save results
        metrics['validation_loss'] = validation_loss
        self.save_hyperparameters_model_metrics(model, config, metrics)

        return model, config, metrics

    def evaluate_model(self, model, validation_loader, criterion) -> tuple:
        """
        Evaluate the model on validation data.

        Args:
            model: Neural network model.
            validation_loader (DataLoader): Validation data loader.
            criterion: Loss function.

        Returns:
            float: Validation loss, Validation RMSE.
        """

        model.eval()
        total_loss, total_rmse = 0.0, 0.0

        with torch.no_grad(): # No gradient needed
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                outputs = model(inputs)
                
                # Compute the loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                # Compute RMSE
                rmse = torch.sqrt(torch.mean((outputs - targets) ** 2))
                total_rmse += rmse.item()

        # Calculate average validation loss for the epoch
        average_loss = total_loss / len(validation_loader)
        average_rmse = total_rmse / len(validation_loader)

        return average_loss, average_rmse

    def save_hyperparameters_model_metrics(self, model, config, metrics):
        """
        Save the trained model, hyperparameters, and metrics.

        Args:
            model: Trained PyTorch model.
            config (dict): Hyperparameter configuration.
            metrics (dict): Training and validation metrics.
        """

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        folder = os.path.join('models', 'neural_networks', 'regression', current_datetime)
        os.makedirs(folder, exist_ok=True)

        # Save the model
        torch.save(model.state_dict(), os.path.join(folder, 'model.pth'))
        # Save hyperparameters
        with open(os.path.join(folder, 'hyperparameters.json'), 'w') as f:
            json.dump(config, f, indent=4)
        # Save metrics
        with open(os.path.join(folder, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Model, hyperparameters and metrics saved in {folder}")



# END OF FILE

"""
Purpose: Contains the neural network model, dataset class, training logic, and utility functions.

Functionality:

Dataset Class (AirbnbNightlyPriceRegressionDataset):
- Defines how the tabular data is loaded, processed, and fed into the model.

Model Class (TabularModel):
- Defines the architecture of the neural network for price prediction.

Training Class (Trainer):
- Manages the training loop, including forward and backward passes, loss computation, and optimizer updates.

Model Saving (save_model):
- Saves the trained model, its hyperparameters, and evaluation metrics.

Hyperparameter Tuning (find_best_nn):
- Likely used for tuning the best hyperparameters for the model
"""