 # src/ utils/data/data_analysis.py

import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Neural_Network import AirbnbNightlyPriceRegressionDataset, create_data_loaders, TabularModel, Trainer, save_model, find_best_nn



# Create instance of import osdataset
dataset = AirbnbNightlyPriceRegressionDataset(csv_file='/home/millie/Documents/GitHub/AirBnB/data/processed/clean_tabular_data.csv')

# Create data loaders
train_loader, validation_loader = create_data_loaders(dataset)

# Load model configuration
with open('src/utils/config/nn_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Create instance of model
input_size = 9
hidden_size = 64
output_size = 1

model = TabularModel(input_size, hidden_size, output_size)

# Train and save the model
trainer = Trainer(model, torch.nn.MSELoss(), torch.optim.Adam(model.parameters()), SummaryWriter('runs'))
#best_model, best_hyperparameters, best_metrics = trainer.train_and_save(train_loader, config)


# Save the best model
#if best_model is not None:
#    save_model(best_model, best_hyperparameters, best_metrics, 'best_model')



# END OF FILE



'''
Refactoring Suggestions:
Training Code: The training logic in Trainer.train_model and train_and_save is repeated. Refactor this logic into a separate method/function.

Model Configuration: The code to create the model, criterion, optimizer, and writer is similar in both files. Extract this into a helper function.

Model Evaluation: The code to find the best model based on validation metrics is repeated in both files. Extract this into a separate function.
    Write a function that displays the 'answer'

Main Function: The main() function in data_analysis.py contains the overall workflow, including training, model evaluation, and saving. 
    Consider breaking this down into smaller, more focused functions.

Hyperparameter Tuning:
The function find_best_nn() is commented out but suggests the possibility of performing hyperparameter optimization. Un-comment and use it to try different combinations of hyperparameters like learning rate, hidden size, etc., to improve model performance.

Model Evaluation:
You may want to add code for evaluating the model on test data, comparing predictions to actual values, and calculating metrics like Mean Squared Error (MSE).

Model Deployment:
After training and evaluation, the model can be saved and deployed to make real-time predictions for Airbnb prices. Consider writing a separate file for deployment using the trained model.

Enhance Model:
Consider experimenting with different neural network architectures or adding regularization techniques like dropout to prevent overfitting.

TensorBoard Visualization:
Run TensorBoard to visualize the model training process (e.g., loss over time, etc.), which could be helpful for diagnosing the model's performance.

Batch Processing:
Implement batch processing and validation for handling large datasets, ensuring that the model generalizes well on unseen data.

Logging and Monitoring:
Improve logging and monitoring of the model training, and potentially add checkpoints to save model states during training.


Testing: Add unit and intergration testing
'''

"""
Purpose: Handles model training and evaluation.

Functionality:

Dataset Loading:
- Loads the cleaned data (clean_tabular_data.csv) using the AirbnbNightlyPriceRegressionDataset class.

Data Loaders:
- The create_data_loaders function creates training and validation data loaders from the dataset.

Model Configuration:
- Loads a model configuration (nn_config.yaml), which likely contains hyperparameters like learning rate, epochs, etc.

Model Creation:
- A TabularModel neural network model is instantiated, with input_size, hidden_size, and output_size specifying the network architecture.

Model Training:
- The model is trained using the Trainer class, which handles training with a chosen loss function (MSELoss) and optimizer (Adam).
- Optionally, the best model is saved using the save_model function after training.

TensorBoard Integration:
A SummaryWriter is initialized to log metrics for visualization in TensorBoard.

Next Step: Once the model is trained and evaluated, the best model can be saved and used for predictions.
"""