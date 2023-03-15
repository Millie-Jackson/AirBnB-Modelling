import pandas as pd

'''
Functions:
        remove_rows_with_missing_ratings(): Removes the rows with missing values
        combine_description_strings(): Combines the list of strings from the descriptions column into one string.
        set_default_feature_values(): Replace empty values from guests, beds, bathrooms an bedrooms with 1.
        clean_tabular_data(): Calls all the data cleaning functions on the tabular data.
'''


def remove_rows_with_missing_ratings(df) -> None:
    '''
    Removes the rows with missing values.
    Puts all the ratings columns into a separate dataframe
    Forces all values into floats
    Checks for NaN values (bool)
    Counts how many Nan values there are
    Drops all the NaN values
    Resets the index
    # Adds separate dataframe to df
    Takes a dataset as a dataframe.

        Parameters:
                df (dataframe): dataframe of tabular AirBnB data
        Returns:
                df (dataframe): dataframe with all the NaN rating values removed
    '''

    # Put all the ratings columns in one dataframe
    df_ratings = df[["ID", "Accuracy_rating", "Communication_rating", "Location_rating", "Check-in_rating", "Value_rating"]]

    # Force values into floats (not needed in airbnb but here for future databases)
    df_ratings = df_ratings.apply(pd.to_numeric, errors='coerce')

    # Check if there are NaN values in the dataframe (returns bool)
    found_NaN_ratings = df_ratings.isnull().values.any()

    if found_NaN_ratings == True:
        print("The following missing ratings have been found and removed")
        # How many values are NaN
        print(df_ratings.isnull().sum())
        # Drop all the NaN values
        df_ratings = df_ratings.dropna()
        # Reset the index
        df_ratings = df_ratings.reset_index(drop=True)
    else:
        print("No missing ratings to remove")

    # NEED TO ADD df_ratings TO df
    df = pd.merge(df, df_ratings, how="inner", on=["ID", "ID"])

    return df 

def combine_description_strings(df) -> None:
    '''Combines the list of strings from the descriptions column into one string.
       Removes empty quotes.
       Removes records with empty discriptions.
       Removes "About this space" prefix.
       Takes in a dataset as a dataframe.
       Returns a dataset as a dataframe.'''
    
    # Combine lists of strings into one string

    # Removes empty quotes
    #df = df.loc[df["Description"] == "", "Description"] = "TEST"

    # Removes empty descriptions

    # Removes "About this space" prefix"
    #df.loc[df["Description"].str.contains("About this space")]
    #df.loc[df["Description"] == "About this space", "Description"] = "TEST"

    return df 

def set_default_feature_values(df) -> None:
    '''Replace empty values from guests, beds, bathrooms an bedrooms with 1.
       Takes in a dataset as a dataframe.
       Returns a dataset as a dataframe.'''
    
    # Replace empty values with "TEST"
    # Replace empty values with "1"
    #df = df.loc[df["guests"] == "0", "guests"] = "TEST"
    #df = df.loc[df["beds"] == "0", "guests"] = "TEST"
    #df = df.loc[df["bathrooms"] == "0", "guests"] = "TEST"
    #df = df.loc[df["bedrooms"] == "0", "guests"] = "TEST"
    #print(df["guests"] == "")

    return df

def clean_tabular_data(df):
    '''Calls all the data cleaning functions on the tabular data.
       Takes in a raw dataframe.
       Returns processed data'''
    
    remove_rows_with_missing_ratings(df)
    combine_description_strings(df)
    set_default_feature_values(df)

    # Re index and remove old index 
    df = df.reset_index(drop=True, inplace=True)

    return



if __name__ == "__main__":
    # Load the raw data
    # TODO Do I want a default index or the ID as an index? 
    df = pd.read_csv("AirBnB/Data/tabular_data/listing.csv")
    #df = pd.read_csv("AirBnB/Data/tabular_data/listing.csv", index_col=0)

    # Clean data
    clean_tabular_data(df)

    # Save processed data as a .csv
    df.to_csv("AirBnB/Data/tabular_data/clean_tabular_data.csv", index=False)

    #print(df.head(5))
    print(df.info())
    #print(df.describe())

# END OF FILE 