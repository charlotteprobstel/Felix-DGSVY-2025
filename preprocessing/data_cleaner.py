import pandas as pd
import os
import json
import numpy as np

DATAFILE = 'data/data.csv'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class DataCleaner: 

    def __init__(self, file_path = DATAFILE, output_dir = "data/cleaned_data.csv"):

        """Initialize DataCleaner"""

        self.data = None
        self.file_path = file_path # Path to raw data file
        self.output_dir = output_dir # To save cleaned file
        self.headers = None # This is a dictionary e.g {"Q1": "What is your age?"}

        # Load data
        self.load_data()
        self.save_headers()

        # Drop unnecessary columns
        columns_to_drop = ['StartDate', 'EndDate', 'Q1', 'Status', 'IPAddress', 'Progress', 'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId','ExternalReference', 'RecipientLastName', 'RecipientFirstName', 'RecipientEmail','LocationLatitude', 'LocationLongitude', 'DistributionChannel', 'UserLanguage',]  # Replace with actual column names
        self.drop_columns(columns_to_drop)
        self.remove_metadata()

        # Clean data
        self.clean_data()
        self.save_cleaned_data()

    def load_data(self):
        """Load data from a CSV file."""
        if os.path.exists(self.file_path):
            self.data = pd.read_csv(self.file_path)
        else:
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

    def list_columns(self):
        """List all columns in the DataFrame."""
        for item in self.data.columns.tolist():
            print(f"{item.ljust(20)}: {self.data[item][0]}")

    def clean_data(self):
        """Clean the DataFrame by handling missing values and duplicates."""
        self.data = self.data.drop_duplicates()
        self.data = self.data.fillna(value="None")

    def drop_columns(self, columns):
        """Drop specified columns from the DataFrame."""
        self.data = self.data.drop(columns=columns)

    def remove_metadata(self):
        """Remove first two rows if they contain metadata."""
        self.data = self.data.iloc[3:].reset_index(drop=True)

    def save_cleaned_data(self):
        """Save the cleaned DataFrame to a CSV file."""
        self.data.to_csv(self.output_dir, index=False)

    def save_headers(self):
        """Save the headers of the DataFrame to a text file."""
        qs = self.data.columns.tolist()
        questions = self.data.iloc[0].tolist()
        self.headers = dict(zip(qs, questions))
        with open('data/headers.json', 'w') as f:
            json.dump(self.headers, f, cls=NumpyEncoder, indent=2)


if __name__ == "__main__":
   
    # Example usage
    loader = DataCleaner()

