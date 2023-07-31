import pandas as pd
import ast

'''
Functions:
        remove_rows_with_missing_ratings(): Removes the rows with missing values
        combine_description_strings(): Combines the list of strings from the descriptions column into one string.
        set_default_feature_values(): Replace empty values from guests, beds, bathrooms an bedrooms with 1.
        clean_tabular_data(): Calls all the data cleaning functions on the tabular data.
'''


def remove_rows_with_missing_ratings(df)-> pd.DataFrame:
    
    """
    Removes rows with missing values in the rating columns of the given dataframe.

    Parameters:
        df (pd.DataFrame): The input dataframe containing rating columns.

    Returns:
        pd.DataFrame: A modified dataframe with rows removed for missing values in the rating columns.
    """

    # Put all the ratings columns in one dataframe
    df_ratings = df[["Accuracy_rating", "Communication_rating", "Location_rating", "Check-in_rating", "Value_rating"]]
  
    # Force values into floats (not needed in airbnb but here for future databases)
    df_ratings = df_ratings.apply(pd.to_numeric, errors='coerce')

    # Remove missing values
    df.dropna(subset=df_ratings.columns, inplace=True)

    return df

def combine_description_strings(df)-> pd.DataFrame:

    """
    Combine and clean the strings in the 'Description' column of the given DataFrame.

    This function removes missing descriptions (NaN),
    removes the prefix "'About this space', 
    and removes empty quotes from the lists in the 'Description' column.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the 'Description' column.

    Returns:
        pd.DataFrame: A modified DataFrame with cleaned 'Description' strings.

    Notes:
        - The changes are applied directly to the original DataFrame, modifying it in place.
        - If the 'Description' column contains actual lists, the prefix removal and empty quote
          removal might not be necessary; verify the format of the 'Description' values beforehand.
    """

    # Removes missing descriptions
    df = df.dropna(subset=["Description"])
    # Removes "About this space" prefix
    df["Description"] = df["Description"].str.replace("'About this space', ", "")
    # Remove empty quotes from the lists
    df["Description"] = df["Description"].str.replace(r"['\"]\s*['\"]", "", regex=True)

    return df

def set_default_feature_values(df)-> pd.DataFrame:

    """
    Sets default values for the feature columns that contain missing values (NaN) in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing feature columns.

    Returns:
        pandas.DataFrame: The modified DataFrame with the default feature values.

    Raises:
        None

    """

    # Put all the feature columns in one dataframe
    df_features = df[["guests", "beds", "bathrooms", "bedrooms"]]

    # Replace all NaN with 1
    df_features = df_features.fillna(1)
    
    return df_features

def clean_tabular_data(df) -> None:

    """
    Clean the tabular data in the given DataFrame.

    This function performs the following operations on the DataFrame:
    1. Makes a copy of the original DataFrame to keep track of changes.
    2. Removes rows with missing values in the rating columns.
    3. Combines and cleans the strings in the 'Description' column.
    4. Sets default feature values to fill missing values in the DataFrame.
    5. Compares if the DataFrame has been modified after the updates.
    6. Displays a message indicating whether the original DataFrame has been updated.
    7. Reindexes the DataFrame and removes the old index.

    Parameters:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: A modified DataFrame with cleaned data.

    Notes:
        - The changes are applied directly to the input DataFrame, modifying it in place.
        - The function may print a message indicating whether the original DataFrame has
          been updated after cleaning.

    Example:
        df = clean_tabular_data(df)

    """

    df_before_update = df.copy()  # Make a copy of the original dataframe

    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)

    # Compare if 'df' has been modified after the update
    is_updated = not df.equals(df_before_update)

    if is_updated:
        print("The original dataframe 'df' has been updated successfully.")
    else:
        print("The original dataframe 'df' remains unchanged.")

    # Re index and remove old index 
    df.reset_index(drop=True, inplace=True)

    return None

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
        features, labels = load_airbnb()
        # or specify a different label column
        features, labels = load_airbnb(label="Another_Label_Column")
    """

    # Load cleaned data
    try:
        df = pd.read_csv("Data/tabular_data/clean_tabular_data.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Cant find cleaned data file")

    # Check if the label column is in the data
    if label not in df.columns:
        raise ValueError(f"'{label}' is not a features")

    # Filter out non-numeric columns
    features = df.select_dtypes(include=[int, float])

    # Remove label column from features
    features.drop(columns=[label], inplace=True, errors="ignore")
    labels = df[[label]]

    return features, labels



if __name__ == "__main__":
    # Load the raw data
    df = pd.read_csv("Data/tabular_data/listing.csv")

    # Clean data
    clean_tabular_data(df)

    # Save processed data as a .csv
    df.to_csv("Data/tabular_data/clean_tabular_data.csv", index=False)

    # Extract
    features, labels = load_airbnb()


# END OF FILE 
