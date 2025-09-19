import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
class EDAVisualiser:
    def __init__(self, data, output_dir, color_palette='newspaper_elegant', font_family='DIN Alternate', font_size=11, style='white'):
        """
        Initialize the EDA Visualiser with custom theming options.
        
        Parameters:
        -----------
        color_palette : str or list
            Color palette for plots ('viridis', 'plasma', 'Set1', etc. or custom list of colors)
        font_family : str
            Font family for all text elements
        font_size : int
            Base font size for plots
        style : str
            Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        """
        self.color_palette = color_palette
        self.font_family = font_family
        self.font_size = font_size
        self.style = style

        self.data = data
        self.output_dir = output_dir

        # Set global styling
        sns.set_style(self.style)
        plt.rcParams.update({
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.titlesize': self.font_size + 2,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.font_size - 1,
            'figure.titlesize': self.font_size + 4,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.edgecolor': 'white',
            'axes.linewidth': 2,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        
        # Custom color schemes
        self.colors = self._get_color_scheme()
        
    def _get_color_scheme(self):
        """Get color scheme based on palette selection."""
        if isinstance(self.color_palette, list):
            return self.color_palette
        elif self.color_palette == 'newspaper_elegant':
            return ['#1B365D', '#800020', '#2F4F4F', '#000000', '#4F4F4F', '#8B0000', '#191970']
        elif self.color_palette == 'skyblue_elegant':
            return ['#87CEEB', '#4682B4', '#5F9EA0', '#B0E0E6', '#ADD8E6', '#6495ED', '#4169E1', '#1E90FF']
        elif self.color_palette == 'custom_blue':
            return ['#0066cc', '#4da6ff', '#b3d9ff', '#003d7a', '#1a8cff']
        elif self.color_palette == 'custom_warm':
            return ['#ff6b35', '#f7931e', '#ffd23f', '#06ffa5', '#118ab2']
        else:
            return sns.color_palette(self.color_palette, 10)
    
    def update_theme(self, color_palette=None, font_family=None, font_size=None, style=None):
        """Update theme settings dynamically."""
        if color_palette:
            self.color_palette = color_palette
            self.colors = self._get_color_scheme()
        if font_family:
            self.font_family = font_family
            plt.rcParams['font.family'] = font_family
        if font_size:
            self.font_size = font_size
            plt.rcParams.update({
                'font.size': font_size,
                'axes.titlesize': font_size + 2,
                'axes.labelsize': font_size,
                'xtick.labelsize': font_size - 1,
                'ytick.labelsize': font_size - 1,
                'legend.fontsize': font_size - 1,
                'figure.titlesize': font_size + 4
            })
        if style:
            self.style = style
            sns.set_style(style)

    def _add_info_box(self, ax, non_null_count, total_count=None):
        """Add info box in bottom right corner with non-null count information."""
        if total_count is None:
            info_text = f'Non-null: {non_null_count}'
        else:
            info_text = f'Non-null: {non_null_count}/{total_count}'

        # Create a text box in the bottom right corner
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='lightgrey', alpha=0.8, edgecolor='grey')
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=self.font_size-2,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=bbox_props, color='grey')

    # BASIC PLOTS
    def histogram(self, column, xlabel, bins=None, title=None, figsize=(10, 6), kde=True, alpha=0.7):
        """Create histogram with optional KDE overlay using seaborn."""
        _, ax = plt.subplots(figsize=figsize)

        # Clean data - convert to numeric and drop non-numeric values
        numeric_data = pd.to_numeric(self.data[column], errors='coerce').dropna()
        total_count = len(self.data[column])
        non_null_count = len(numeric_data)

        if len(numeric_data) == 0:
            ax.text(0.5, 0.5, f'No numeric data available for {column}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=self.font_size)
        else:
            # Use seaborn histogram
            sns.histplot(data=numeric_data, bins=bins or 'auto', kde=kde, alpha=alpha,
                        color=self.colors[0], edgecolor='white', linewidth=1.5, ax=ax)

        ax.set_title(title or f'Distribution of {column}', pad=20, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')

        # Add info box with non-null count
        self._add_info_box(ax, non_null_count, total_count)

        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{column}_histogram.svg', dpi=300, bbox_inches='tight')
        plt.close()

    def bar_chart(self, column, xlabel, title, figsize=(10, 6), sort_ascending=False, order=None):
        """Create bar chart of frequency for each unique value using seaborn.

        Parameters:
        -----------
        order : list, optional
            Custom order for x-axis values (e.g. ["0-1", "1-2", "2-3"])
        """
        _, ax = plt.subplots(figsize=figsize)

        # Filter out None/NaN values and get value counts
        data_clean = self.data[column].dropna()
        value_counts = data_clean.value_counts()
        total_count = len(self.data[column])
        non_null_count = len(data_clean)

        # Apply custom ordering if provided
        if order is not None:
            # Reindex to match the custom order, filling missing values with 0
            value_counts = value_counts.reindex(order, fill_value=0)
        elif sort_ascending:
            # Sort by index (values) if requested and no custom order
            value_counts = value_counts.sort_index(ascending=True)

        if len(value_counts) == 0:
            ax.text(0.5, 0.5, f'No data available for {column}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=self.font_size)
        else:
            # Use seaborn barplot with proper color handling
            colors_extended = (self.colors * (len(value_counts) // len(self.colors) + 1))[:len(value_counts)]
            sns.barplot(x=value_counts.index, y=value_counts.values, hue=value_counts.index,
                       palette=colors_extended, alpha=0.8, edgecolor='white', linewidth=1.5,
                       ax=ax, legend=False)

            # Rotate x labels
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for i, count in enumerate(value_counts.values):
                ax.text(i, count + max(value_counts.values) * 0.01, str(count),
                       ha='center', va='bottom')

        ax.set_title(title, pad=20, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')

        # Add info box with non-null count
        self._add_info_box(ax, non_null_count, total_count)

        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{column}_bar_chart.svg', dpi=300, bbox_inches='tight')
        plt.close()

    def pie_chart(self, column, title, figsize=(8, 8)):
        """Create seaborn-styled pie chart of each value (excluding None)."""
        _, ax = plt.subplots(figsize=figsize)

        # Filter out None/NaN values and get value counts
        data_clean = self.data[column].dropna()
        value_counts = data_clean.value_counts()
        total_count = len(self.data[column])
        non_null_count = len(data_clean)

        if len(value_counts) == 0:
            ax.text(0.5, 0.5, f'No data available for {column}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=self.font_size)
        else:
            # Use seaborn color palette or custom colors
            if isinstance(self.color_palette, str) and self.color_palette in ['viridis', 'plasma', 'husl', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20']:
                colors = sns.color_palette(self.color_palette, n_colors=len(value_counts))
            else:
                colors = self.colors[:len(value_counts)]

            _, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index,
                                            autopct='%1.1f%%', colors=colors,
                                            startangle=90, pctdistance=0.85, labeldistance=1.0,
                                            wedgeprops=dict(width=0.8, edgecolor='white', linewidth=2))

            # Apply seaborn styling to text elements
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
                autotext.set_fontsize(self.font_size - 1)

            for text in texts:
                text.set_fontsize(self.font_size - 1)
                text.set_weight('normal')

        ax.set_title(title, pad=20, fontweight='bold', fontsize=self.font_size + 2)
        ax.axis('equal')

        # Add info box with non-null count
        self._add_info_box(ax, non_null_count, total_count)

        # Apply seaborn despine for cleaner look
        sns.despine(left=True, bottom=True)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{column}_pie_chart.svg', dpi=300, bbox_inches='tight')
        plt.close()

    def horizontal_bar_chart(self, column, ylabel, title, figsize=(10, 6), parse_comma_separated=False):
        """Create horizontal bar chart with most frequent answer at the top using seaborn.

        Parameters:
        -----------
        parse_comma_separated : bool, optional
            If True, parse each value by comma and count individual values
            (e.g., "Beer, Wine" counts as two separate values)
        """
        _, ax = plt.subplots(figsize=figsize)

        # Filter out None/NaN values
        data_clean = self.data[column].dropna()
        total_count = len(self.data[column])

        if parse_comma_separated:
            # Parse comma-separated values and count individually
            all_values = []
            for value in data_clean:
                # Split by comma and strip whitespace
                individual_values = [v.strip() for v in str(value).split(',')]
                all_values.extend(individual_values)
            # Create a Series from the expanded values
            expanded_series = pd.Series(all_values)
            value_counts = expanded_series.value_counts()
            non_null_count = len(all_values)  # Count of individual parsed values
        else:
            # Standard value counts
            value_counts = data_clean.value_counts()
            non_null_count = len(data_clean)

        if len(value_counts) == 0:
            ax.text(0.5, 0.5, f'No data available for {column}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=self.font_size)
        else:
            # Use seaborn horizontal barplot with proper color handling
            colors_extended = (self.colors * (len(value_counts) // len(self.colors) + 1))[:len(value_counts)]
            sns.barplot(y=value_counts.index, x=value_counts.values, hue=value_counts.index,
                       palette=colors_extended, alpha=0.8, edgecolor='white', linewidth=1.5,
                       ax=ax, orient='h', legend=False)

            # Add value labels on bars
            for i, count in enumerate(value_counts.values):
                ax.text(count + max(value_counts.values) * 0.01, i, str(count),
                       ha='left', va='center')

        ax.set_title(title, pad=20, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Frequency')

        # Add info box with non-null count
        self._add_info_box(ax, non_null_count, total_count)

        sns.despine()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{column}_horizontal_bar.svg', dpi=300, bbox_inches='tight')
        plt.close()

    def combination_heatmap(self, column1, column2, xlabel, ylabel, title, figsize=(10, 8), parse_comma_separated=False):
        """Create heatmap counting frequency of combinations from two columns using seaborn.

        Parameters:
        -----------
        parse_comma_separated : bool, optional
            If True, parse each value by comma and count individual combinations
            (e.g., "Beer, Wine" in column1 and "Yes" in column2 creates combinations for both "Beer"-"Yes" and "Wine"-"Yes")
        """
        _, ax = plt.subplots(figsize=figsize)

        # Filter out None/NaN values from both columns
        data_clean = self.data[[column1, column2]].dropna()
        total_count = len(self.data[[column1, column2]].dropna(how='any'))

        if len(data_clean) == 0:
            ax.text(0.5, 0.5, f'No data available for {column1} and {column2}',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=self.font_size)
        else:
            if parse_comma_separated:
                # Parse comma-separated values and create expanded combinations
                expanded_data = []
                for _, row in data_clean.iterrows():
                    col1_values = [v.strip() for v in str(row[column1]).split(',')]
                    col2_values = [v.strip() for v in str(row[column2]).split(',')]

                    # Create all combinations between col1 and col2 values
                    for col1_val in col1_values:
                        for col2_val in col2_values:
                            expanded_data.append({column1: col1_val, column2: col2_val})

                # Create DataFrame from expanded combinations
                expanded_df = pd.DataFrame(expanded_data)
                crosstab = pd.crosstab(expanded_df[column1], expanded_df[column2])
                non_null_count = len(expanded_data)  # Count of expanded combinations
            else:
                # Standard crosstab
                crosstab = pd.crosstab(data_clean[column1], data_clean[column2])
                non_null_count = len(data_clean)

            # Create custom colormap for newspaper theme
            from matplotlib.colors import LinearSegmentedColormap
            colors_heatmap = ['#FFFFFF', '#1B365D', '#800020']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('newspaper', colors_heatmap, N=n_bins)

            # Create heatmap using seaborn
            sns.heatmap(crosstab, annot=True, fmt='d', cmap=cmap,
                       cbar_kws={'label': 'Frequency'}, square=False,
                       linewidths=0.5, linecolor='white', ax=ax)

        ax.set_title(title, pad=20, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Add info box with non-null count
        self._add_info_box(ax, non_null_count, total_count)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{column1}_{column2}_heatmap.svg', dpi=300, bbox_inches='tight')
        plt.close()
