import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the DataFrame by handling missing values and duplicates."""
    df = df.drop_duplicates()
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def analyze_data(df):
    """Perform basic data analysis and return summary statistics."""
    summary = df.describe()
    return summary

def list_columns(df):
    """List all columns in the DataFrame."""
    return df.columns.tolist()

def drop_columns(df, columns):
    """Drop specified columns from the DataFrame."""
    return df.drop(columns=columns)

if __name__ == "__main__":
    # Example usage
    file_path = 'data/data.csv'  # Replace with your CSV file path
    df = load_data(file_path)
    
    # List all columns
    columns = list_columns(df)
    print("\nColumns in DataFrame:")
    print(columns)

    # Drop unnecessary columns
    columns_to_drop = ['StartDate', 'EndDate', 'Q1', 'Status', 'IPAddress', 'Progress', 'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId','ExternalReference', 'RecipientLastName', 'RecipientFirstName', 'RecipientEmail','LocationLatitude', 'LocationLongitude', 'DistributionChannel', 'UserLanguage',]  # Replace with actual column names
    df = drop_columns(df, columns_to_drop)
    print("\nDataFrame after dropping specified columns:")
    print(list_columns(df))

    # Remove duplicates and handle missing values
    df = clean_data(df)
    
    # Show summary statistics
    summary = analyze_data(df)
    print("\nSummary Statistics:")
    print(summary)

    # Save cleaned data to a new CSV file
    df.to_csv('data/cleaned_data.csv', index=False)
    summary.to_csv("data/summary.csv", index=True)
