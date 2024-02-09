
from torch.utils.data import DataLoader
from Neural_Network import AirbnbnightlyPriceRegressionDataset, create_data_loaders, TabularModel, train, save_model

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
train(model, train_loader, validation_loader, num_epochs=10, learning_rate=0.001, config_file='utils/nn_config.yaml')



# END OF FILE