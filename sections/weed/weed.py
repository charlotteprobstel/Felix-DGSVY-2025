import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.visualise import EDAVisualiser
from data_loader import DataLoader

class Weed: 

    """Class to handle 'Weed' data operations."""

    def __init__(self, output_dir='weed/plots'):
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
        self.relevant_columns = ["Q50", "Q57"]

    def how_do_you_smoke_your_weed(self):
        """Visualise how students smoke their weed."""
        self.visualiser.horizontal_bar_chart('Q50', ylabel = "Methods of Consuming Weed", title = self.headers["Q50"], parse_comma_separated=True)

    def have_you_ever_greened_out(self):
        """Visualise if students have ever greened out."""
        self.visualiser.pie_chart('Q51', title = self.headers["Q51"])
    
    def have_you_ever_tried_edibles(self):
        """Visualise if students have ever tried edibles."""
        self.visualiser.pie_chart('Q52', title = self.headers["Q52"])

    def which_type_of_edibles(self):
        """Visualise the types of edibles students have tried."""
        self.visualiser.horizontal_bar_chart('Q53', ylabel = "Types of Edibles", title = self.headers["Q53"], parse_comma_separated=True)

    def which_type_of_edibles_text(self):
        extras = self.df['Q53_4_TEXT'].dropna().unique()
        print("Extra types of edibles:")
        for extra in extras:
            print(extra)

    def have_you_ever_ingested_spice(self):
        """Visualise if students have ever ingested spice."""
        self.visualiser.pie_chart('Q54', title = self.headers["Q54"])

    def where_do_you_get_your_drugs(self):
        """Visualise where students get their drugs."""
        self.visualiser.horizontal_bar_chart('Q55', ylabel = "Sources of Drugs", title = self.headers["Q55"], parse_comma_separated=True)

    def where_do_you_get_your_drugs_text(self):
        extras = self.df['Q55_6_TEXT'].dropna().unique()
        print("Extra sources of drugs:")
        for extra in extras:
            print(extra)

    def how_close_to_your_deeler(self):
        """Visualise how close students are to their dealer."""
        self.visualiser.horizontal_bar_chart('Q56', ylabel = "Closeness to Dealer", title = self.headers["Q56"], parse_comma_separated=True)

    def how_close_to_your_deeler_text(self):
        extras = self.df['Q56_4_TEXT'].dropna().unique()
        print("Extra closeness to dealer responses:")
        for extra in extras:
            print(extra)

    def how_to_contact_your_plug(self):
        """Visualise how students contact their plug."""
        self.visualiser.horizontal_bar_chart('Q57', ylabel = "Methods of Contacting Plug", title = self.headers["Q57"], parse_comma_separated=True)

    def how_to_contact_your_plug_text(self):
        extras = self.df['Q57_6_TEXT'].dropna().unique()
        print("Extra methods of contacting plug:")
        for extra in extras:
            print(extra)

    def plot_all(self):
        self.how_do_you_smoke_your_weed()
        self.have_you_ever_greened_out()
        self.have_you_ever_tried_edibles()
        self.which_type_of_edibles()
        self.which_type_of_edibles_text()
        self.have_you_ever_ingested_spice()
        self.where_do_you_get_your_drugs()
        self.where_do_you_get_your_drugs_text()
        self.how_close_to_your_deeler()
        self.how_close_to_your_deeler_text()
        self.how_to_contact_your_plug()
        self.how_to_contact_your_plug_text()

if __name__ == "__main__":
    weed = Weed()
    weed.plot_all()