import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preprocessing.visualise import EDAVisualiser
from preprocessing.data_loader import DataLoader

class Location: 

    """Class to handle 'Location' data operations."""

    def __init__(self, output_dir='sections/location/plots'):
        """Initialize with the path to the data file."""
        # Load data
        self.DataLoader = DataLoader()
        self.df = self.DataLoader.data
        self.headers = self.DataLoader.headers

        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup visualiser 
        self.visualiser = EDAVisualiser(data = self.df, output_dir = self.output_dir)

        # Columns 
        self.relevant_columns = ["Q47", "Q49"]

    def which_on_campus(self):
        """Visualise substances taken on campus."""
        self.visualiser.horizontal_bar_chart(column="Q47", title=self.headers["Q47"], ylabel="Substances", parse_comma_separated=True)

    def which_drugs_on_campus(self):
        """Visualise substances taken on campus."""
        self.visualiser.horizontal_bar_chart(column="Q48", title=self.headers["Q48"], ylabel="Drugs", parse_comma_separated=True)

    def which_drugs_on_campus_text(self): 
        extras = self.df['Q48_16_TEXT'].dropna().unique()
        with open(os.path.join(self.output_dir, 'which_drugs_on_campus.txt'), 'w') as f:
            for extra in extras:
                f.write(f"{extra}\n")


    def plot_all(self):
        self.which_on_campus()  
        self.which_drugs_on_campus()
        self.which_drugs_on_campus_text()
  

if __name__ == "__main__":
    nicotine = Location()
    nicotine.plot_all()