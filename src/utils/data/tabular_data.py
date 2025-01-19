# src/utils/data/tabular_data.py

import pandas as pd
from utils.data.data_cleaner import DataCleaner



'''
Functions:
        remove_rows_with_missing_ratings(): Removes the rows with missing values
        combine_description_strings(): Combines the list of strings from the descriptions column into one string.
        set_default_feature_values(): Replace empty values from guests, beds, bathrooms an bedrooms with 1.
        clean_tabular_data(): Calls all the data cleaning functions on the tabular data.
'''



def load_airbnb(label="Price_Night") -> pd.DataFrame:

    """
    Load the cleaned Airbnb data and return numerical features and the specified column as the label.

    Parameters:
        label (str, optional): The name of the column to be used as the label. Default is "Price_Night".

    Returns:
        tuple: A tuple containing the numerical features (X) and the specified column as the label (y) as pandas DataFrames.

    Raises:
        FileNotFoundError: If the cleaned data file is not found.
        ValueError: If the specified label column is not found in the data.

    Example:
        features, labels = load_airbnb_data()
        # or specify a different label column
        features, labels = load_airbnb_data(label="Another_Label_Column")
    """

    # Load cleaned data
    try:
        df = pd.read_csv("data/processed/clean_tabular_data.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Cant find cleaned data file")

    # Check if the label column is in the data
    if label not in df.columns:
        raise ValueError(f"'{label}' is not a valid label column")

    # Filter out non-numeric columns
    features = df.select_dtypes(include=[int, float])

    # Remove label column from features
    features.drop(columns=[label], inplace=True, errors="ignore")
    labels = df[[label]]

    return features, labels

def main():

    import os
    print("Current Working Directory:", os.getcwd())


    # Load the raw data
    #cleaner = DataCleaner("data/raw/tabular_data/listing.csv")
    cleaner = DataCleaner(os.path.join(os.path.dirname(__file__), "../data/raw/tabular_data/listing.csv"))

    
    # Clean the data
    cleaner_df = cleaner.clean_tabular_data()

    # Save processed data as a .csv
    cleaner_df.to_csv("data/processed/clean_tabular_data.csv", index=False)

    # Extract
    features, labels = load_airbnb()



if __name__ == "__main__":

    main()



# END OF FILE 



"""
This script defines a module for handling tabular Airbnb data cleaning and feature-label extraction for downstream tasks like modeling. Below is a detailed breakdown:

Purpose
The module focuses on:

Data Cleaning: Cleaning raw Airbnb data to prepare it for analysis or modeling.
Feature Extraction: Extracting numerical features and labels from the cleaned data.
Key Functions
load_airbnb()

Purpose: Loads the cleaned Airbnb data, extracts numerical features, and separates a specified column as the label.
Parameters:
label (str): The name of the label column to extract (default is "Price_Night").
Returns: A tuple (features, labels) where:
features contains all numerical columns except the label.
labels contains the values from the specified label column.
Raises:
FileNotFoundError: If the cleaned data file is missing.
ValueError: If the specified label column is not found in the data.
Example:
python
Copy code
features, labels = load_airbnb()
Notes:
Expects the cleaned data to exist in "data/processed/clean_tabular_data.csv".
Non-numerical columns are filtered out automatically.
main()

Purpose: Acts as the entry point for data cleaning and preparation.
Steps:
Loads the raw Airbnb data from "data/raw/tabular_data/listing.csv" using the DataCleaner class.
Cleans the data using the clean_tabular_data() method of DataCleaner.
Saves the cleaned data to "data/processed/clean_tabular_data.csv".
Calls load_airbnb() to extract features and labels.
Notes:
Relies on the DataCleaner class from data_cleaner.py for cleaning operations.
External Dependency
DataCleaner Class:
Assumes a DataCleaner class is implemented in data_cleaner.py.
Responsible for cleaning raw Airbnb tabular data.
Potential Improvements
Error Handling:
Add more robust error handling when calling methods of DataCleaner to ensure failures don't cascade.
Logging:
Implement logging for the various steps (loading raw data, cleaning, saving, etc.) for better traceability.
Dynamic Paths:
Use environment variables or configuration files for path management to avoid hardcoded paths.
Usage Example
python
Copy code
# Command Line
python src/utils/data/tabular_data.py
This will:

Clean raw data from data/raw/tabular_data/listing.csv.
Save processed data as data/processed/clean_tabular_data.csv.
Extract features and labels from the cleaned dataset.
"""

"""
Purpose: Loads the cleaned dataset and prepares it for use in training or model evaluation.
F
unctionality:

load_airbnb():
- Loads the cleaned Airbnb data (clean_tabular_data.csv).
- Selects numerical features and separates the target label (price).

This data is passed to the model training pipeline in data_analysis.py to be used for training.
"""