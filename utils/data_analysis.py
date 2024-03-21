# data_analysis.py

import yaml
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Neural_Network import AirbnbNightlyPriceRegressionDataset, create_data_loaders, TabularModel, Trainer, save_model, find_best_nn



# Create instance of import osdataset
dataset = AirbnbNightlyPriceRegressionDataset(csv_file='/home/millie/Documents/GitHub/AirBnB/data/tabular_data/clean_tabular_data.csv')

# Create data loaders
train_loader, validation_loader = create_data_loaders(dataset)

# Load model configuration
with open('utils/nn_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Create instance of model
input_size = 9
hidden_size = 64
output_size = 1

model = TabularModel(input_size, hidden_size, output_size)

# Train and save the model
trainer = Trainer(model, torch.nn.MSELoss(), torch.optim.Adam(model.parameters()), SummaryWriter('runs'))
best_model, best_hyperparameters, best_metrics = trainer.train_and_save(train_loader, config)


# Save the best model
#if best_model is not None:
#    save_model(best_model, best_hyperparameters, best_metrics, 'best_model')



# END OF FILE
'''
Refactoring Suggestions:
Training Code: The training logic in Trainer.train_model and train_and_save is repeated. Refactor this logic into a separate method/function.

Model Configuration: The code to create the model, criterion, optimizer, and writer is similar in both files. Extract this into a helper function.

Model Evaluation: The code to find the best model based on validation metrics is repeated in both files. Extract this into a separate function.

Main Function: The main() function in data_analysis.py contains the overall workflow, including training, model evaluation, and saving. 
    Consider breaking this down into smaller, more focused functions.

Testing: Add unit and intergration testing
'''