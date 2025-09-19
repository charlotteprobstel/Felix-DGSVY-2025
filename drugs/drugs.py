import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualise import EDAVisualiser
from data_loader import DataLoader

class Drugs: 

    """Class to handle 'Drugs' data operations."""

    def __init__(self, output_dir='drugs/plots'):
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
        self.relevant_columns = ["Q35", "Q44"]

    def which_drugs(self):
        """Visualise drug usage data."""
        self.visualiser.horizontal_bar_chart('Q35', ylabel = "Drugs", title = self.headers["Q35"], parse_comma_separated=True)

    def which_drugs_text(self): 
        extras = self.df['Q35_16_TEXT'].dropna().unique()
        print("Extra drugs:")
        for extra in extras:
            print(extra)
       
    def which_drugs_combinations(self):
        """Visualise combinations of drugs used."""
        self.visualiser.horizontal_bar_chart('Q36', ylabel = "Drug Combinations", title = self.headers["Q36"], parse_comma_separated=True)

    def which_drugs_combinations_text(self):
        extras = self.df['Q36_16_TEXT'].dropna().unique()
        print("Extra drug combinations:")
        for extra in extras:
            print(extra)

    def which_drugs_never_taken(self):
        """Visualise drugs never taken."""
        self.visualiser.horizontal_bar_chart('Q37', ylabel = "Drugs Never Taken", title = self.headers["Q37"], parse_comma_separated=True)

    def which_drugs_never_taken_text(self):
        extras = self.df['Q37_16_TEXT'].dropna().unique()
        print("Extra drugs never taken:")
        for extra in extras:
            print(extra)

    def which_drugs_like_to_try(self):
        """Visualise drugs students would like to try."""
        self.visualiser.horizontal_bar_chart('Q38', ylabel = "Drugs Like to Try", title = self.headers["Q38"], parse_comma_separated=True)

    def how_frequently_take_drugs(self):
        """Visualise frequency of drug use."""
        order = ["Never", "Once or twice", "Monthly", "Weekly", "Daily or almost daily"]
        self.visualiser.bar_chart('Q39', xlabel = "Frequency", title = self.headers["Q39"], order=order)

    def does_family_know(self):
        """Visualise if family knows about drug use."""
        self.visualiser.pie_chart('Q40', title = self.headers["Q40"])

    def ever_needed_medical_attention(self):
        """Visualise if medical attention was needed due to drug use."""
        self.visualiser.pie_chart('Q41', title = self.headers["Q41"])

    def why_take_drugs(self):
        """Visualise reasons for taking drugs."""
        self.visualiser.horizontal_bar_chart('Q42', ylabel = "Reasons for Taking Drugs", title = self.headers["Q42"], parse_comma_separated=True)

    def why_take_drugs_text(self):
        extras = self.df['Q42_8_TEXT'].dropna().unique()
        print("Extra reasons for taking drugs:")
        for extra in extras:
            print(extra)

    def worst_time_getting_high(self):
        """Visualise worst experiences while high."""
        extras = self.df['Q43'].dropna().unique()
        print("Extra worst experiences while high:")
        for extra in extras:
            print(extra)

    def had_sex_on_drugs(self):
        """Visualise if students had"""
        self.visualiser.pie_chart('Q44', title = self.headers["Q44"])

    def age_first_had_alcohol(self):
        """Visualise age when first had alcohol."""
        self.visualiser.histogram('Q45_1', xlabel = "Age", title = self.headers["Q45_1"])

    def age_first_took_drugs(self):
        """Visualise age when first took drugs."""
        self.visualiser.histogram('Q46_1', xlabel = "Age", title = self.headers["Q46_1"])


    def plot_all(self):
        self.which_drugs()  
        self.which_drugs_text()
        self.which_drugs_combinations()
        self.which_drugs_combinations_text()
        self.which_drugs_never_taken()
        self.which_drugs_never_taken_text()
        self.which_drugs_like_to_try()
        self.how_frequently_take_drugs()
        self.does_family_know()
        self.ever_needed_medical_attention()
        self.why_take_drugs()
        self.why_take_drugs_text()
        self.worst_time_getting_high()
        self.had_sex_on_drugs()
        self.age_first_had_alcohol()
        self.age_first_took_drugs()

if __name__ == "__main__":
    nicotine = Drugs()
    nicotine.plot_all()