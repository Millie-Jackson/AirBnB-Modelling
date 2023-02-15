import pandas as pd

def remove_rows_with_missing_ratings(df) -> df:
    '''Removes the rows with missing values. Takes a dataset as a dataframe. Returns a dataset as a dataframe'''
    return df 

def combine_description_strings(df) -> df:
    '''Combines the list of strings from the descriptions column into one string.
       Removes empty quotes.
       Removes records with empty discriptions.
       Removes "About this space" prefix
       Takes in a dataset as a dataframe.
       Returns a dataset as a dataframe.'''
    return df

def set_default_feature_values(df) -> df:
    '''Replace empty values from guests, beds, bathrooms an bedrooms with 1.
       Takes in a dataset as a dataframe.
       Returns a dataset as a dataframe.'''
    return df

def clean_tabular_data():
    '''Calls all the data cleaning functions on the tabular data.
       Takes in a raw dataframe.
       Returns processed data'''
    
    remove_rows_with_missing_ratings()
    combine_description_strings()
    set_default_feature_values()

    return

if __name__ == "__main__":
    # Load the raw data
    df = pd.read_csv("AirBnB/Data/tabular_data/listing.csv")

    # Clean data
    clean_tabular_data()

    # Save processed data as a .csv

# END OF FILE 