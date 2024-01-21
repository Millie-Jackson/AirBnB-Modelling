from torch.utils.data import DataLoader
from Neural_Network import AirbnbnightlyPriceRegressionDataset, create_data_loaders

# Create instance of dataset
dataset = AirbnbnightlyPriceRegressionDataset

# Create data loaders
train_loader, validation_loader = create_data_loaders(dataset)


'''# Example of using a dataloader
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for batch in train_loader:
    features, labels = batch
    
for batch in validation_loader:
    features, labels = batch'''