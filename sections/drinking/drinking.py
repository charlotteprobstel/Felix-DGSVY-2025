import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preprocessing.visualise import EDAVisualiser
from preprocessing.data_loader import DataLoader

class Drinking: 

    """Class to handle 'Drinking' data operations."""

    def __init__(self, output_dir='sections/drinking/plots'):
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
        self.relevant_columns = ["Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17"]

    def plot_drink(self):
        """Plot if students drink."""
        self.visualiser.pie_chart('Q9', title = self.headers["Q9"])

    def plot_drinking_frequency(self):
        """Plot the drinking frequency distribution."""
        order = ["Daily", "4-6 times a week", "2-3 times a week", "Weekly", "Fortnightly", "Monthly", "Every few months"]
        self.visualiser.bar_chart('Q10', xlabel = "Drinking Frequency", title = self.headers["Q10"], order=order)

    def plot_which_drinks(self):
        """Plot the which drinks distribution."""
        self.visualiser.horizontal_bar_chart('Q11', ylabel = "Drinks", title = self.headers["Q11"], parse_comma_separated=True)

    def plot_shots(self):
        """Plot if students drink shots."""
        self.visualiser.pie_chart('Q12', title = self.headers["Q12"])

    def plot_family_knows(self):
        """Plot if family knows about drinking habits."""
        self.visualiser.pie_chart('Q13', title = self.headers["Q13"])

    def plot_why_drink(self):
        """Plot the reasons for drinking distribution."""
        self.visualiser.horizontal_bar_chart('Q14', ylabel = "Reasons", title = self.headers["Q14"], parse_comma_separated=True)

    def plot_why_drink_extra(self):
        """Plot the extra reasons for drinking distribution."""
        extra_reasons = self.df['Q14_6_TEXT'].dropna().unique()
        with open(os.path.join(self.output_dir, 'extra_why_drink.txt'), 'w') as f:
            for extra in extra_reasons:
                f.write(f"{extra}\n")

    def plot_if_ill_from_drinking(self):
        """Plot if students feel ill from drinking."""
        self.visualiser.pie_chart('Q15', title = self.headers["Q15"])

    def plot_how_ill(self):
        """Plot how ill students have been from drinking."""
        self.visualiser.bar_chart('Q16', xlabel = "How Ill", title = self.headers["Q16"], sort_ascending=True)

    def average_units_per_week(self):
        """Plot the average units consumed per week."""
        order = ["0-1", "1-3", "4-6", "7-9", "10-12", "13-14", "15+"]
        self.visualiser.bar_chart('Q17', xlabel = "Units per Week", title = self.headers["Q17"], sort_ascending=True, order=order)

    def age_vs_drinking_frequency(self):
        """Plot age vs drinking frequency."""
        self.visualiser.combination_heatmap('Q10', 'Q2', xlabel="Age", ylabel="Drinking Frequency", title="Age vs Drinking Frequency")

    def age_vs_why_drinks(self):
        """Plot age vs why drinks."""
        self.visualiser.combination_heatmap('Q14', 'Q2', xlabel="Age", ylabel="Why Drinks", title="Age vs Why Drinks", parse_comma_separated=True)

    def age_vs_how_ill(self):
        """Plot age vs how ill."""
        self.visualiser.combination_heatmap('Q15', 'Q2', xlabel="Age", ylabel="How Ill", title="Age vs How Ill", parse_comma_separated=False)

    def plot_all(self):
        """Plot all relevant distributions."""
        self.plot_drink()
        self.plot_drinking_frequency()
        self.plot_which_drinks()
        self.plot_shots()
        self.plot_family_knows()
        self.plot_why_drink()
        self.plot_why_drink_extra()
        self.plot_if_ill_from_drinking()
        self.plot_how_ill()
        self.average_units_per_week()
        self.age_vs_drinking_frequency()
        self.age_vs_why_drinks()



if __name__ == "__main__":
    
    about_you = Drinking()
    about_you.plot_all()
