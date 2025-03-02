import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVLoader:
    def __init__(self):
        """
        Initializes an empty dictionary to store multiple DataFrames.
        """
        self.dataframes = {}
    
    def load_csv(self, file_path: str, name: str):
        """
        Load a CSV file into a DataFrame and store it with the given name.
        Convert 'Date' column to datetime if it exists.
        :param file_path: Path to the CSV file.
        :param name: Name to store the DataFrame.
        """
        try:
            df = pd.read_csv(file_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                logger.info(f"Converted 'Date' column to datetime in DataFrame '{name}'.")
            self.dataframes[name] = df
            logger.info(f"CSV file '{file_path}' loaded successfully as '{name}'.")
            self.get_shape(name)
        except Exception as e:
            logger.error(f"Error loading CSV file '{file_path}': {e}")
    
    def get_shape(self, name: str):
        """
        Get the shape of the DataFrame.
        :param name: Name of the DataFrame.
        :return: Tuple (rows, columns)
        """
        if name in self.dataframes:
            shape = self.dataframes[name].shape
            logger.info(f"Shape of DataFrame '{name}': {shape}")
            return shape
        else:
            logger.warning(f"DataFrame '{name}' not found.")
            return None
    
    def check_missing_values(self, name: str):
        """
        Check for missing values in each column of the DataFrame.
        :param name: Name of the DataFrame.
        :return: DataFrame with columns, missing value count, and percentage.
        """
        if name in self.dataframes:
            df = self.dataframes[name]
            logger.info(f"Checking for missing values in DataFrame '{name}'.")
            missing_info = df.isnull().sum()
            missing_percentage = (missing_info / len(df)) * 100
            logger.info("Missing values check completed.")
            self.get_shape(name)
            return pd.DataFrame({
                'Column': df.columns,
                'Missing Values': missing_info,
                'Missing Percentage': missing_percentage,
                'Data Type': df.dtypes
            }).reset_index(drop=True)
        else:
            logger.warning(f"DataFrame '{name}' not found.")
            return None
    
    def get_dataframe(self, name: str):
        """
        Retrieve a DataFrame by its name.
        :param name: Name of the DataFrame.
        :return: The requested DataFrame or None if not found.
        """
        if name in self.dataframes:
            logger.info(f"Returning DataFrame '{name}'.")
            self.dataframes[name].shape
            return self.dataframes[name]
        else:
            logger.warning(f"DataFrame '{name}' not found.")
            return None
''' 
# Usage Example
if __name__ == "__main__":
    csv_loader = CSVLoader()
    
    # Load multiple CSV files
    csv_loader.load_csv("data1.csv", "df1")
    csv_loader.load_csv("data2.csv", "df2")
    
    # Check missing values
    print(csv_loader.check_missing_values("df1"))
    print(csv_loader.check_missing_values("df2"))
    
    # Retrieve a DataFrame
    df1 = csv_loader.get_dataframe("df1")
    if df1 is not None:
        print(df1.head())
'''