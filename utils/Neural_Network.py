
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split
#from torch.autograd import Variable



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
    def __init__(self, input_size, hidden_size, output_size):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.re1u = nn.ReLu()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
def train(model, train_loader, validation_loader, num_epochs=10, learning_rate=0.001):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    model.to(device)
    criterion.to(device)

    # Training loop
    for epoch in range(num_epochs):

        # Training
        model.train()
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device, labels.to(device))

            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = running_loss / len(train_loader)
            average_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4}")

        # Validation
        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for features, labels in validation_loader:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()

        average_validation_loss = validation_loss / len(validation_loader)
        print(f"Validation Loss: {average_validation_loss:.4f}")

    print("Training Complete")

# END OF FILE