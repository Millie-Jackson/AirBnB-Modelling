import pandas as pd



def remove_rows_with_missing_ratings(df) -> None:
    '''Removes the rows with missing values.
    Takes a dataset as a dataframe.
    Returns a dataset as a dataframe.'''

    '''# Remove rows with missing values
    df["Missing_Values"] = df[].apply(lambda x : x.drop())'''

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
    #print(df.info())
    print(df.describe())

# END OF FILE 