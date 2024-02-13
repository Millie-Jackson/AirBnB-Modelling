
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import yaml
import json
import time

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from subprocess import Popen, PIPE
from datetime import datetime



class AirbnbnightlyPriceRegressionDataset(Dataset):

    def __init__(self, csv_file):
        
        self.data = pd.read_csv(csv_file) # Loads the data
        
        # Select only numerical columns
        numeric_columns = self.data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        self.data = self.data[numeric_columns]
        self.features = torch.tensor(self.data.drop('Price_Night', axis=1).values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['Price_Night'].values, dtype=torch.float32).view(-1, 1)

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
    
def train(model, train_loader, validation_loader, num_epochs=10, learning_rate=0.001, config=None) -> None:
    
    # Check if there is a config file
    if config:
        # Get hyperparameters
        optimizer_name = config.get('optimiser', 'adam')
        learning_rate = config.get('learning_rate', 0.001)
        hidden_size = config.get('hidden_layer_width', 64)
        depth = config.get('depth', 2)
    
        # Set hyperparameter
        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Initializae model with hidden layers
        input_size = 9
        hidden_layers = []
        for _ in range(depth):
            hidden_layers.append(nn.Linear(input_size, hidden_size))
            hidden_layers.append(nn.ReLU())
            input_size = hidden_size

        # Add the hidden layers to the model
        model.hidden_layers = nn.Sequential(*hidden_layers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    model.to(device)
    criterion.to(device)

    # Record start time for training duration
    start_time = time.time()  

    # Create a SummaryWriter for with a log directory
    writer = SummaryWriter('runs')
    log_dir = 'runs'
    writer = SummaryWriter(log_dir)
    if not os.path.exists(log_dir):
        print(f"Warning: The '{log_dir} directory was not created")
    else:
        print(f"Logs will be stored in '{log_dir}'.")
    
    # Star TensorBoard in the background
    tensorboard_process = Popen(["tensorboard", "--logdir=runs"], stdout=PIPE, stderr=PIPE)

    # Training loop
    for epoch in range(num_epochs):

        # Training
        model.train()
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #average_loss += loss.item()

        # Log the training loss
        average_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/Train', average_loss, epoch)
        #print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4}")
        
        # Validation
        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()

        # Log the validation loss
        average_validation_loss = validation_loss / len(validation_loader)
        writer.add_scalar('Loss/Validation', average_validation_loss, epoch)

        print(f"Validation Loss: {average_validation_loss:.4f}")

    # Close SummaryWriter
    writer.close()
    # Stop TensorBoard
    tensorboard_process.terminate()

    print("Training Complete")

    # Save the model
    # Calculate training duration
    training_duration = time.time() - start_time

    # Create hyperparameters dictionary
    hyperparameters = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'optimizer': optimizer_name,
        'hidden_size': hidden_size,
        'depth': depth
    }

    # Create performance_metrics dictionary
    performance_metrics = {
        'training_loss': average_loss,
        'validation_loss': average_validation_loss,
        'training_duration': training_duration
    }

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    folder = os.path.join('models', 'neural_networks', 'regression', current_datetime)
    save_model(model, hyperparameters, performance_metrics, folder)

    return model, performance_metrics, hyperparameters

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

def find_best_nn(train_function, model_class, train_loader, validation_loader):
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



# END OF FILE