import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.visualise import EDAVisualiser
from data_loader import DataLoader

class Nicotine: 

    """Class to handle 'Nicotine' data operations."""

    def __init__(self, output_dir='nicotine/plots'):
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
        self.relevant_columns = ["18", "Q34"]

    
    def do_you_consume_nicotine(self):
        """Plot if students consume nicotine."""
        self.visualiser.pie_chart('Q18', title = self.headers["Q18"])

    def what_type_of_nicotine(self):
        """Plot the types of nicotine products used."""
        self.visualiser.horizontal_bar_chart('Q19', ylabel = "Nicotine Products", title = self.headers["Q19"], parse_comma_separated=True)
    
    def which_type_of_cigarettes(self):
        """Plot the types of cigarettes used."""
        self.visualiser.horizontal_bar_chart('Q20', ylabel = "Cigarette Types", title = self.headers["Q20"], parse_comma_separated=True)

    def which_brand_of_cigarettes(self):
        """Plot the brands of cigarettes used."""
        self.visualiser.horizontal_bar_chart('Q21', ylabel = "Cigarette Brands", title = self.headers["Q21"], parse_comma_separated=True)
    
    def how_many_cigarettes_per_day(self):
        """Plot the number of cigarettes smoked per day."""
        order = ["<1","1-2", "3-5", "6-8", "9-11", "12-14","18-20", "20+"]
        self.visualiser.bar_chart('Q22', xlabel = "Cigarettes per Day", title = self.headers["Q22"], sort_ascending=True, order=order)

    def have_you_told_your_doctor_you_smoke(self):
        """Plot if students have told their doctor they smoke."""
        self.visualiser.pie_chart('Q23', title = self.headers["Q23"])

    def what_type_of_vape(self):
        """Plot the types of vape used."""
        self.visualiser.horizontal_bar_chart('Q24', ylabel = "Vape Types", title = self.headers["Q24"], parse_comma_separated=True)
    
    def which_brand_of_vape(self):
        """Plot the brands of vape used."""
        self.visualiser.horizontal_bar_chart('Q25', ylabel = "Vape Brands", title = self.headers["Q25"], parse_comma_separated=True)
    
    def favourite_vape_flavour(self):
        """Plot the favourite vape flavours."""
        self.visualiser.horizontal_bar_chart('Q26', ylabel = "Vape Flavours", title = self.headers["Q26"], parse_comma_separated=True)

    def how_long_to_finish_600_vape_puffs(self):
        """Plot the time taken to finish 600 vape puffs."""
        order = ["A few hours", "A day", "A few days", "A week", "The battery dies before I finish"]
        self.visualiser.bar_chart('Q27', xlabel = "Time to Finish 600 Puffs", title = self.headers["Q27"], sort_ascending=True, order=order)

    def how_much_snus_a_week(self):
        """Plot the amount of snus used per week."""
        order = ["More than three daily","Two-three daily", "At least one daily", "Two-three a week", "Once a week"]
        self.visualiser.bar_chart('Q29', xlabel = "Snus per Week", title = self.headers["Q29"], sort_ascending=True, order=order)

    def strength_of_snus(self):
        """Plot the strength of snus used."""
        order = ["3mg", "6mg", "8/9mg", "10mg", "10-15mg", "15-30mg", ">30mg"]
        self.visualiser.bar_chart('Q30', xlabel = "Strength of Snus", title = self.headers["Q30"], sort_ascending=True, order=order)
    
    def dot_strength_snus(self):
        """Plot the strength of snus used."""
        order = []
        for i in range(2, 7):
                order.append(f"{i} dot")
        self.visualiser.bar_chart('Q31', xlabel = "Strength of Snus", title = self.headers["Q31"], order = order)

    def favourite_snus_brand(self):
        """Plot the favourite snus brands."""
        self.visualiser.horizontal_bar_chart('Q32', ylabel = "Snus Brands", title = self.headers["Q32"], parse_comma_separated=True)

    def favourite_snus_brand_extra(self):
        """Plot the extra favourite snus brands."""
        extra_brands = self.df['Q32_10_TEXT'].dropna().unique()
        print("Extra favourite snus brands:")
        for brand in extra_brands:
            print(brand)
        ## The answers
        """""
        N/A
        """""

    def why_use_nicotine(self):
        """Plot the reasons for using nicotine products."""
        self.visualiser.horizontal_bar_chart('Q33', ylabel = "Reasons", title = self.headers["Q33"], parse_comma_separated=True)
    
    def why_use_nicotine_extra(self):
        """Plot the extra reasons for using nicotine products."""
        extra_reasons = self.df['Q33_6_TEXT'].dropna().unique()
        print("Extra reasons for using nicotine products:")
        for reason in extra_reasons:
            print(reason)
        
        ## The answers
        """
        N/A
        """
    
    def have_you_tried_quitting(self):
        """Plot if students have tried quitting nicotine."""
        self.visualiser.pie_chart('Q34', title = self.headers["Q34"])


    def plot_all(self):
        self.do_you_consume_nicotine()
        self.what_type_of_nicotine()
        self.which_type_of_cigarettes()
        self.which_brand_of_cigarettes()
        self.how_many_cigarettes_per_day()
        self.have_you_told_your_doctor_you_smoke()
        self.what_type_of_vape()
        self.which_brand_of_vape()
        self.favourite_vape_flavour()
        self.how_long_to_finish_600_vape_puffs()
        self.how_much_snus_a_week()
        self.strength_of_snus()
        self.dot_strength_snus()
        self.favourite_snus_brand()
        self.favourite_snus_brand_extra()
        self.why_use_nicotine()
        self.why_use_nicotine_extra()
        self.have_you_tried_quitting()
        
if __name__ == "__main__":
    
    nicotine = Nicotine()
    nicotine.plot_all()

