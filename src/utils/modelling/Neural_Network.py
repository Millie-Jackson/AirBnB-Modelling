# Neural_Network.py

import torch
import torch.optim as optim
import pandas as pd
import os
import yaml
import json
import time

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from subprocess import Popen, PIPE
from datetime import datetime



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

def get__nn__config(config_file='nn_config.yaml') -> None:

    with open(config_file, 'r') as file:
        nn_config = yaml.safe_load(file)

    return nn_config



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
        # Save moduel
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
    learning_rates = [0.001, 0.01]
    hidden_layer_widths = [32, 64]
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

def find_best_nn(train_function, model_class, train_loader, validation_loader, config):
    '''Train models with different configurations to find the best'''

    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_performance = float('inf')

    configs = generate_nn_configs()

    for i, config in enumerate(configs):
        print(f"Training model {i+1}/{len(configs)} with config: {config}")

        # Train the model with current configuration
        model, metrics, hyperparameters = train_function(
            model_class, train_loader, validation_loader, config=config)
        
        # Save the hyperparameters used for training
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        folder = os.path.join('models', 'neural_networks', 'regression', current_datetime)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, 'hyperparameters.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # Check if the current model performs better than the previous best model
        if metrics['validation_loss'] < best_performance:
            best_model = model
            best_metrics = metrics
            best_hyperparameters = hyperparameters
            best_performance = metrics['validation_loss']

    # Save the best model
    if best_model is not None:
        save_model(best_model, best_hyperparameters, best_metrics, folder)

    return best_model, best_metrics, best_hyperparameters



class Trainer:
    def __init__(self, model, criterion, optimizer, writer, tensorboard_process=None) -> None:
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer
        self.tensorboard_process = tensorboard_process

        self.hyperparameters = None
        self.metrics = None

    def train_model(self, train_loader, num_epochs):
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.criterion.to(device)

        # Initialize hyperparameters
        self.hyperparameter = {
            'num_epochs': num_epochs,
            'optimizer': type(self.optimizer).__name__
        }
        # Initialize metics
        self.metrics = {
            'train_loss': []
        }

        for epoch in range(num_epochs):
            running_loss = 0.0
            self.model.train()

            for batch_idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(device), labels.to(device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(features)
                # Compute the loss
                loss = self.criterion(outputs, labels)
                # Backward pass
                loss.backward()
                # Optimization: Update model parameters
                self.optimizer.step()

                running_loss += loss.item()

            # Calculate average loss for the epoch
            average_loss = running_loss / len(train_loader)
            # Log average loss for the epoch
            self.writer.add_scalar('Loss/Train', average_loss, epoch)
            # Store training loss for each epoch
            self.metrics['train_loss'].append(average_loss)

            #loss =  running_loss / len(train_loader)
        
        return self.model, self.hyperparameters, self.metrics

    def train_and_save(self, train_loader, config):
        
        # Create model
        input_size = 9
        hidden_size = config.get('hidden_layer_width', 64)
        output_size = 1
        model = TabularModel(input_size, hidden_size, output_size)

        # Create loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer_name = config.get('optimiser', 'adam')
        learning_rate = config.get('learning_rate', 0.001)
        num_epochs = config.get('num_epochs', 10)
        depth = config.get('depth', 2)

        # Set hyperparameters
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Start Tensorboard in background
        writer = SummaryWriter('runs')
        tensorboard_process = Popen(["tensorboard", "--logdir=runs"], stdout=PIPE, stderr=PIPE)

        try:
            # Train model
            trainer = Trainer(model, criterion, optimizer, writer)
            #trained_model, training_metrics, training_hyperparameters = trainer.train_model(train_loader, num_epochs)
            trained_model, training_hyperparameters, training_metrics = trainer.train_model(train_loader, num_epochs)

            print("Training Complete")

            # Save the model
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            folder = os.path.join('models', 'neural_networks', 'regression', current_datetime)
            os.makedirs(folder, exist_ok=True)
            save_model(trained_model, training_hyperparameters, training_metrics, folder)

            print("Model Saved")
        except Exception as e:
            print(f"An error occured during training and saving: {e}")
        finally:
            # Close SummaryWriter and TensorBoard
            writer.close()
            if tensorboard_process is not None:
                tensorboard_process.terminate()
                tensorboard_process.wait() # Wait for tensorboard to finish closeing before moving on
        
        return trained_model, training_hyperparameters, training_metrics



# END OF FILE