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

        Parameters:
                df (dataframe): dataframe of tabular AirBnB data
        Returns:
                df (dataframe): dataframe with all the NaN rating values removed
    '''

    # Put all the ratings columns in one dataframe
    df_ratings = df[["Accuracy_rating", "Communication_rating", "Location_rating", "Check-in_rating", "Value_rating"]]

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
    else:
        print("No missing ratings to remove")

    return df_ratings 

def combine_description_strings(df) -> None:
    '''Combines the list of strings from the descriptions column into one string.
       Removes empty quotes.
       Removes records with empty discriptions.
       Removes "About this space" prefix.
       Takes in a dataset as a dataframe.
       Returns a dataset as a dataframe.'''
    
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
    '''Replace empty values from guests, beds, bathrooms an bedrooms with 1.
       Takes in a dataset as a dataframe.
       Returns a dataset as a dataframe.'''

    # Put all the feature columns in one dataframe
    df_features = df[["guests", "beds", "bathrooms", "bedrooms"]]

    # Replace all NaN with 1
    df_features = df_features.fillna(1)

    return df_features

def clean_tabular_data(df):
    '''Calls all the data cleaning functions on the tabular data.
       Takes in a raw dataframe.
       Returns processed data'''
    
    #remove_rows_with_missing_ratings(df)
    combine_description_strings(df)
    #set_default_feature_values(df)

    # Merge all cleaned data into one dataframe
    #cleaded_df = pd.merge(remove_rows_with_missing_ratings(df), combine_description_strings(df), set_default_feature_values(df))

    # Re index and remove old index 
    df = df.reset_index(drop=True, inplace=True)

    return



if __name__ == "__main__":
    # Load the raw data
    df = pd.read_csv("AirBnB/Data/tabular_data/listing.csv")

    # Clean data
    clean_tabular_data(df)

    # Save processed data as a .csv
    df.to_csv("AirBnB/Data/tabular_data/clean_tabular_data.csv", index=False)

    #print(df.head(5))
    #print(df.info())
    #print(df.describe())

# END OF FILE 