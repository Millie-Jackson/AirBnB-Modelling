import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

class AirbnbnightlyPriceRegressionDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file) # Loads the data
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



class TabularModel(nn.Module):
    def __init__(self, input_size, hidden_size, Output_size):
        super(TabularModel, self). __init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn nn.ReLu()
        self.fc2 = nn.Linear(hidden_size, Output_size)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
def train(model, dataloader, num_epochs):
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parapeters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(dataloader):
            # Forward pass
            outputs = model(features)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epocj [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
                