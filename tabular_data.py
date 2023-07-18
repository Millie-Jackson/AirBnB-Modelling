import pandas as pd

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

    # Update original dataframe
    df.update(df_ratings)

    return df

def combine_description_strings(df) -> None:

    """
    Combines the strings in the 'Description' column of the DataFrame into a single string.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the 'Description' column.

    Returns:
        pandas.DataFrame: The modified DataFrame with the combined description strings.

    Raises:
        None

    """
    
    # Put description column into one dataframe
    df_description = df[["Description"]]
    print(df_description)

    # Removes empty descriptions
    df_description = df_description.dropna()

    # Combine lists of strings into one string
    df_description= df_description["Description"].apply(str)

    # Removes "About this space" prefix" and empty quotes
    df_description = df_description.str.replace("'About this space'," ,"")

    return df_description 

def set_default_feature_values(df) -> None:

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

def clean_tabular_data(df):

    df_before_update = df.copy()  # Make a copy of the original dataframe

    remove_rows_with_missing_ratings(df)

    # Compare if 'df' has been modified after the update
    is_updated = not df.equals(df_before_update)

    if is_updated:
        print("The original dataframe 'df' has been updated successfully.")
    else:
        print("The original dataframe 'df' remains unchanged.")

    # Re index and remove old index 
    df.reset_index(drop=True, inplace=True)

    return None



if __name__ == "__main__":
    # Load the raw data
    df = pd.read_csv("Data/tabular_data/listing.csv")

    # Clean data
    clean_tabular_data(df)

    # Save processed data as a .csv
    df.to_csv("Data/tabular_data/clean_tabular_data.csv", index=False)

# END OF FILE 





