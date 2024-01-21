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

'''Example NN
model = TabularModel(imput_size= ,hidden_size=64, output_size=1)
train(model, your_train_dataloader, num_epochs=1)'''