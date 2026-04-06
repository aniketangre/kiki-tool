# data/load_data.py
# Loads the raw Excel database and does basic cleanup.

import os
import pandas as pd


def load_data(filepath):
    """
    Load the dataset from an Excel file.

    Steps:
      1. Read the Excel file into a pandas DataFrame.
      2. Remove any rows that have missing (NaN) values.
      3. Reset the row index so it starts from 0.

    Parameters:
        filepath (str): Full path to the Excel file (e.g. "database_upd.xlsx").

    Returns:
        df (pd.DataFrame): Cleaned DataFrame ready for preprocessing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Database file not found: {filepath}")

    print(f"Loading data from: {filepath}")
    df = pd.read_excel(filepath, engine="openpyxl")

    # Remove rows with any missing values
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# -----------------------------------------------------------------------
# Run this file directly to test data loading:
#   python src/data/load_data.py
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Navigate two levels up from this file to reach the project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    filepath = os.path.join(project_root, "data", "raw", "database_upd.xlsx")

    df = load_data(filepath)
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nColumn names:")
    print(list(df.columns))
