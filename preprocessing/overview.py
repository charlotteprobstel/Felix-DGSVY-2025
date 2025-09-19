import pandas as pd
import numpy as np
from data_loader import DataLoader
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Overview:

    def __init__(self):
        self.dataLoader = DataLoader()
        self.data = self.dataLoader.data
        self.headers = self.dataLoader.headers

    def missing_values(self, column):
        """Check for missing values in the dataset."""
        return int(self.data[column].isnull().sum())
    
    def number_of_rows(self):
        """Get the number of rows in the dataset."""
        return int(self.data.shape[0])

    def value_counts(self, column):
        """Get value counts for a specific column."""
        if column in self.data.columns:
            return self.data[column].value_counts().to_dict()
        else:
            raise ValueError(f"Column {column} does not exist in the DataFrame.")
        
    def combine_values(self, column):
        """Combine similar values in a specific column."""
        main_dict = {}
        if column in self.data.columns:
            main_dict["question"]  = self.headers.get(column, "Unknown Question")
            main_dict['total_rows'] = self.number_of_rows()
            main_dict['missing_values'] = self.missing_values(column)
            main_dict['value_counts'] = self.value_counts(column)
            return main_dict
        else:
            raise ValueError(f"Column {column} does not exist in the DataFrame.")
        
    def combine_columns(self):
        main_dict = {}
        for column in self.data.columns:
            main_dict[column] = self.combine_values(column)

        with open('data/overview.json', 'w') as f:
            json.dump(main_dict, f, cls=NumpyEncoder, indent=2)
        
if __name__ == "__main__":
    overview = Overview()
    overview.combine_columns()