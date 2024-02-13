# data_analysis.py

import yaml

from torch.utils.data import DataLoader
from Neural_Network import AirbnbnightlyPriceRegressionDataset, create_data_loaders, TabularModel, train, save_model, find_best_nn

# Create instance of import osdataset
dataset = AirbnbnightlyPriceRegressionDataset(csv_file='/home/millie/Documents/GitHub/AirBnB/data/tabular_data/clean_tabular_data.csv')

# Create data loaders
train_loader, validation_loader = create_data_loaders(dataset)

# Create instance of model
input_size = 9
hidden_size = 64
output_size = 1

model = TabularModel(input_size, hidden_size, output_size)

# Train (and save) the model
with open('utils/nn_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
train(model, train_loader, validation_loader, num_epochs=10, learning_rate=0.001, config=config)

# Find best models
best_model, best_metrics, best_hyperparameters = find_best_nn(train, model, train_loader, validation_loader)

# Save the best model
if best_model is not None:
    save_model(best_model, best_hyperparameters, best_metrics, 'best_model')



# END OF FILE