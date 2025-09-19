import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preprocessing.visualise import EDAVisualiser
from preprocessing.data_loader import DataLoader

class AboutYou: 

    """Class to handle 'About You' data operations."""

    def __init__(self, output_dir='sections/about_you/plots'):
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

        # About you columns 
        self.relevant_columns = ["Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]

    def plot_age_distribution(self):
        """Plot the age distribution."""
        self.visualiser.bar_chart('Q2', xlabel = "Age", title = self.headers["Q2"], sort_ascending=True)

    def plot_type_distribution(self):
        """Plot the gender distribution."""
        self.visualiser.pie_chart('Q3', title = self.headers["Q3"])

    def plot_department_distribution(self):
        """Plot the department distribution."""
        self.visualiser.horizontal_bar_chart('Q4', ylabel = "Department", title = self.headers["Q4"])

    def plot_year_distribution(self):
        """Plot the year of study distribution."""
        self.visualiser.bar_chart('Q5', xlabel = "Year of Study", title = self.headers["Q5"], sort_ascending=True)

    def plot_gender_distribution(self):
        """Plot the gender distribution."""
        self.visualiser.pie_chart('Q6', title = self.headers["Q6"])

    def plot_gender_expression_distribution(self):
        """Plot the gender expression distribution."""
        self.visualiser.pie_chart('Q7', title = self.headers["Q7"])
    
    def plot_sexual_orientation_distribution(self):
        """Plot the sexual orientation distribution."""
        self.visualiser.pie_chart('Q8', title = self.headers["Q8"])

    def plot_all(self):
        """Plot all relevant distributions."""
        self.plot_age_distribution()
        self.plot_type_distribution()
        self.plot_department_distribution()
        self.plot_year_distribution()
        self.plot_gender_distribution()
        self.plot_gender_expression_distribution()
        self.plot_sexual_orientation_distribution()


if __name__ == "__main__":
    
    about_you = AboutYou()
    about_you.plot_all()
