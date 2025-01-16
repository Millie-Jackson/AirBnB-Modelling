# /utils/data_cleaner.py
import pandas as pd



class DataCleaner:

    """
    A class for cleaning tabular data from a given CSV file.

    This class encapsulates data cleaning operations for tabular data, including removing
    rows with missing values in rating columns, combining and cleaning description strings,
    and setting default feature values for missing entries.

    Args:
        filename (str): The path to the CSV file containing the tabular data.

    Attributes:
        filename (str): The path to the CSV file containing the tabular data.
        df (pandas.DataFrame): The DataFrame containing the tabular data.

    Methods:
        remove_rows_with_missing_ratings(): Removes rows with missing values in rating columns.
        combine_description_strings(): Combines and cleans the strings in the 'Description' column.
        set_default_feature_values(): Sets default values for feature columns with missing values.
        clean_tabular_data(): Performs a comprehensive data cleaning process on the DataFrame.

    Example:
        # Create an instance of DataCleaner
        cleaner = DataCleaner("data.csv")

        # Clean the tabular data
        cleaner.clean_tabular_data()

    """

    def __init__(self, filename):
        self.filename = filename 
        self.df = None

    def remove_rows_with_missing_ratings(self)-> pd.DataFrame:
    
        """
        Removes rows with missing values in the rating columns of the given dataframe.
        
        Returns:
            None
        """

        # Put all the ratings columns in one dataframe
        df_ratings = self.df[["Accuracy_rating", "Communication_rating", "Location_rating", "Check-in_rating", "Value_rating"]]

        # Remove missing values
        self.df.dropna(subset=df_ratings.columns, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self.df
    
    def combine_description_strings(self)-> pd.DataFrame:

        """
        Combine and clean the strings in the 'Description' column of the given DataFrame.
        
        Returns:
            None
        """

        # Removes missing descriptions
        self.df.dropna(subset=["Description"], inplace=True)
        # Removes "About this space" prefix
        self.df["Description"] = self.df["Description"].str.replace("'About this space', ", "")
        # Remove empty quotes from the lists
        self.df["Description"] = self.df["Description"].str.replace(r"['\"]\s*['\"]", "", regex=True)

        return self.df
    
    def set_default_feature_values(self)-> pd.DataFrame:

        """
        Sets default values for the feature columns that contain missing values (NaN) in the DataFrame.
        
        Returns:
            None
        """

        # Put all the feature columns in one dataframe
        feature_columns = ["guests", "beds", "bathrooms", "bedrooms"]

        # Replace all NaN with 1
        for column in feature_columns:
            self.df[column].replace(0, 1, inplace=True)
            self.df[column].fillna(1, inplace=True)

        return self.df
    
    def clean_tabular_data(self) -> pd.DataFrame:

        """
        Clean the tabular data in the given DataFrame.
        
        Returns:
            None
        """

        # Load data
        try:
            self.df = pd.read_csv(self.filename)
            print(f"Looking for data file at: {self.filename}")
        except FileNotFoundError:
            raise FileNotFoundError("Can't find data file")

        df_before_update = self.df.copy()  # Make a copy of the original dataframe

        self.df.drop("Unnamed: 19", axis=1, inplace=True)
        self.df = self.remove_rows_with_missing_ratings()
        self.df = self.combine_description_strings()
        self.df = self.set_default_feature_values()

        # Compare if 'df' has been modified after the update
        is_updated = not self.df.equals(df_before_update)

        if is_updated:
            print("The original dataframe 'df' has been updated successfully.")
        else:
            print("The original dataframe 'df' remains unchanged.")

        # Re index and remove old index 
        self.df.reset_index(drop=True, inplace=True)

        return self.df
    

    
# END OF FILE