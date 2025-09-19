import pandas as pd
import os
import json

DATAFILE = 'data/cleaned_data.csv'
HEADERSFILE = 'data/headers.json'

class DataLoader:

    def __init__(self, file_path = DATAFILE, headers_file = HEADERSFILE):
        self.file_path = file_path
        self.data = None
        self.headers = None
        self.load_data()
        self.load_headers()

    def load_data(self):
        """Load data from a CSV file."""
        if os.path.exists(self.file_path):
            self.data = pd.read_csv(self.file_path)
        else:
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
    def load_headers(self):
        """Load headers from a JSON file."""
        if os.path.exists(HEADERSFILE):
            with open(HEADERSFILE, 'r') as f:
                self.headers = json.load(f)
        else:
            raise FileNotFoundError(f"The file {HEADERSFILE} does not exist.")

if __name__ == "__main__":
   
    # Example usage
    loader = DataLoader()