"""
1. Data Loading

Load the weatherHistory.csv dataset into a pandas DataFrame for processing.
Load dataset using pandas
Use pandas to read the CSV file.
Ensure the file path is correct or accessible.

Display initial data preview
Print the first few rows and basic information (e.g., columns, data types).

Explanation: Loading the dataset allows inspection of its structure and ensures data is accessible for preprocessing.

"""


import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Preview:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    return df

if __name__ == "__main__":
    file_path = "weatherHistory.csv"
    df = load_data(file_path)