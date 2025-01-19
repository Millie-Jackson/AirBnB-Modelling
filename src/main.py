# src/main.py

import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.data.data_cleaner import DataCleaner
from utils.data.tabular_data import load_airbnb
from utils.modelling.Neural_Network import AirbnbNightlyPriceRegressionDataset, create_data_loaders, TabularModel, Trainer, save_model



def main():

    # Step 1: Define file paths
    raw_data_path = "data/raw/tabular_data/listing.csv"
    processed_data_path = "data/processed/clean_tabular_data.csv"
    config_path = "src/utils/config/nn_config.yaml"
    model_save_path = "models/best_model.pt"

    # Step 2: Clean data
    print("Starting data clean...")
    cleaner = DataCleaner(raw_data_path)
    cleaned_df = cleaner.clean_tabular_data()
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    cleaned_df.to_csv(processed_data_path, index=False)
    print(f"Cleaned data saved to {processed_data_path}")

    # Step 3: Load data
    print("Loading cleaned data...")
    features, labels = load_airbnb(label="Price_Night")

    # Step 4: Set up dataset and data loaders
    print("Creating dataset and data loaders...")
    dataset = AirbnbNightlyPriceRegressionDataset(csv_file=processed_data_path)
    train_loader, validation_loader = create_data_loaders(dataset)

    # Step 5: Load configuration
    print("Loading model configuration...")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    input_size = features.shape[1] # Number of numerical features
    hidden_size = config["hidden_layer_width"]
    output_size = 1
    learning_rate = config["learning_rate"]

    # Step 6: Initialize model, loss, optimizer and trainer
    print("Initializing model...")
    model = TabularModel(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    trainer = Trainer(model, loss_fn, optimizer, SummaryWriter("runs"))

    # Step 7: Train the model
    print("Training model...")
    best_model, best_hyperparameter, best_metric = trainer.train_and_save(train_loader, config)

    # Step 8: Save the best model
    if best_model is not None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        save_model(best_model, best_hyperparameter, best_metric, model_save_path)
        print(f"Best model saved to {model_save_path}")
    else:
        print("No model was saved")
    
    print("Pipeline complete!")



if __name__=="__main__":
    main()