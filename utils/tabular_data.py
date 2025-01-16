# utils/tabular_data.py

import pandas as pd
from data_cleaner import DataCleaner



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
        df = pd.read_csv("data/processed_data/clean_tabular_data.csv")
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
    #cleaner = DataCleaner("data/raw_data/tabular_data/listing.csv")
    cleaner = DataCleaner(os.path.join(os.path.dirname(__file__), "../data/raw_data/tabular_data/listing.csv"))

    
    # Clean the data
    cleaner_df = cleaner.clean_tabular_data()

    # Save processed data as a .csv
    cleaner_df.to_csv("data/processed_data/clean_tabular_data.csv", index=False)

    # Extract
    features, labels = load_airbnb()



if __name__ == "__main__":

    main()



# END OF FILE 
