import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class EDAVisualiser:
    def __init__(self, color_palette='viridis', font_family='Arial', font_size=12, style='whitegrid'):
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
            'figure.titlesize': self.font_size + 4
        })
        
        # Custom color schemes
        self.colors = self._get_color_scheme()
        
    def _get_color_scheme(self):
        """Get color scheme based on palette selection."""
        if isinstance(self.color_palette, list):
            return self.color_palette
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
    
    # BASIC PLOTS
    def histogram(self, data, column, bins=30, title=None, figsize=(10, 6), kde=True, alpha=0.7):
        """Create histogram with optional KDE overlay."""
        plt.figure(figsize=figsize)
        plt.hist(data[column], bins=bins, alpha=alpha, color=self.colors[0], edgecolor='black')
        
        if kde:
            sns.kdeplot(data[column], color=self.colors[1], linewidth=2)
        
        plt.title(title or f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def scatter_plot(self, data, x, y, hue=None, size=None, title=None, figsize=(10, 8)):
        """Create scatter plot with optional color and size mapping."""
        plt.figure(figsize=figsize)
        
        if hue and size:
            sns.scatterplot(data=data, x=x, y=y, hue=hue, size=size, palette=self.colors, alpha=0.7)
        elif hue:
            sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=self.colors, alpha=0.7)
        elif size:
            sns.scatterplot(data=data, x=x, y=y, size=size, color=self.colors[0], alpha=0.7)
        else:
            plt.scatter(data[x], data[y], color=self.colors[0], alpha=0.7)
        
        plt.title(title or f'{y} vs {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def line_plot(self, data, x, y, hue=None, title=None, figsize=(12, 6)):
        """Create line plot for time series or sequential data."""
        plt.figure(figsize=figsize)
        
        if hue:
            sns.lineplot(data=data, x=x, y=y, hue=hue, palette=self.colors)
        else:
            sns.lineplot(data=data, x=x, y=y, color=self.colors[0])
        
        plt.title(title or f'{y} over {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def bar_plot(self, data, x, y=None, hue=None, title=None, figsize=(10, 6), orient='v'):
        """Create bar plot for categorical data."""
        plt.figure(figsize=figsize)
        
        if y is None:
            if orient == 'v':
                sns.countplot(data=data, x=x, hue=hue, palette=self.colors)
            else:
                sns.countplot(data=data, y=x, hue=hue, palette=self.colors)
        else:
            if orient == 'v':
                sns.barplot(data=data, x=x, y=y, hue=hue, palette=self.colors)
            else:
                sns.barplot(data=data, x=y, y=x, hue=hue, palette=self.colors)
        
        plt.title(title or f'{y or "Count"} by {x}')
        if orient == 'v':
            plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    # DISTRIBUTION PLOTS
    def box_plot(self, data, x=None, y=None, hue=None, title=None, figsize=(10, 6), orient='v'):
        """Create box plot for distribution analysis."""
        plt.figure(figsize=figsize)
        
        if orient == 'v':
            sns.boxplot(data=data, x=x, y=y, hue=hue, palette=self.colors)
        else:
            sns.boxplot(data=data, x=y, y=x, hue=hue, palette=self.colors, orient='h')
        
        plt.title(title or f'Box Plot of {y or x}')
        if orient == 'v' and x:
            plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def violin_plot(self, data, x=None, y=None, hue=None, title=None, figsize=(10, 6), split=False):
        """Create violin plot for distribution analysis."""
        plt.figure(figsize=figsize)
        
        sns.violinplot(data=data, x=x, y=y, hue=hue, palette=self.colors, split=split)
        
        plt.title(title or f'Violin Plot of {y or x}')
        if x:
            plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def density_plot(self, data, columns, title=None, figsize=(12, 8)):
        """Create density plots for multiple variables."""
        plt.figure(figsize=figsize)
        
        for i, col in enumerate(columns):
            sns.kdeplot(data[col], label=col, color=self.colors[i % len(self.colors)])
        
        plt.title(title or 'Density Plots')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def distribution_comparison(self, data, column, by=None, title=None, figsize=(15, 10)):
        """Create comprehensive distribution comparison."""
        if by is None:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Histogram
            axes[0, 0].hist(data[column], bins=30, alpha=0.7, color=self.colors[0])
            axes[0, 0].set_title(f'Histogram of {column}')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot
            sns.boxplot(y=data[column], ax=axes[0, 1], color=self.colors[1])
            axes[0, 1].set_title(f'Box Plot of {column}')
            
            # Q-Q plot
            stats.probplot(data[column].dropna(), dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title(f'Q-Q Plot of {column}')
            
            # KDE plot
            sns.kdeplot(data[column], ax=axes[1, 1], color=self.colors[2])
            axes[1, 1].set_title(f'KDE Plot of {column}')
            
        else:
            unique_values = data[by].unique()
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Histogram by group
            for i, val in enumerate(unique_values):
                subset = data[data[by] == val]
                axes[0, 0].hist(subset[column], alpha=0.6, label=str(val), 
                               color=self.colors[i % len(self.colors)])
            axes[0, 0].set_title(f'Histogram of {column} by {by}')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot by group
            sns.boxplot(data=data, x=by, y=column, ax=axes[0, 1], palette=self.colors)
            axes[0, 1].set_title(f'Box Plot of {column} by {by}')
            
            # Violin plot by group
            sns.violinplot(data=data, x=by, y=column, ax=axes[1, 0], palette=self.colors)
            axes[1, 0].set_title(f'Violin Plot of {column} by {by}')
            
            # KDE plot by group
            for i, val in enumerate(unique_values):
                subset = data[data[by] == val]
                sns.kdeplot(subset[column], ax=axes[1, 1], label=str(val),
                           color=self.colors[i % len(self.colors)])
            axes[1, 1].set_title(f'KDE Plot of {column} by {by}')
            axes[1, 1].legend()
        
        plt.suptitle(title or f'Distribution Analysis of {column}', fontsize=self.font_size + 4)
        plt.tight_layout()
        return fig
    
    # CORRELATION AND RELATIONSHIP PLOTS
    def correlation_heatmap(self, data, title=None, figsize=(12, 10), annot=True, method='pearson'):
        """Create correlation heatmap."""
        plt.figure(figsize=figsize)
        
        corr_matrix = data.select_dtypes(include=[np.number]).corr(method=method)
        
        sns.heatmap(corr_matrix, annot=annot, cmap=self.color_palette, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title(title or f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        return plt.gcf()
    
    def pair_plot(self, data, hue=None, title=None, figsize=None):
        """Create pair plot for multiple variables."""
        if figsize:
            g = sns.PairGrid(data, hue=hue, palette=self.colors, height=figsize[0]//len(data.columns))
        else:
            g = sns.PairGrid(data, hue=hue, palette=self.colors)
        
        g.map_upper(sns.scatterplot, alpha=0.7)
        g.map_lower(sns.scatterplot, alpha=0.7)
        g.map_diag(sns.histplot, alpha=0.7)
        
        if hue:
            g.add_legend()
        
        if title:
            g.fig.suptitle(title, y=1.02, fontsize=self.font_size + 4)
        
        return g.fig
    
    def scatter_matrix(self, data, columns=None, title=None, figsize=(15, 15)):
        """Create scatter matrix with correlation coefficients."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6 for readability
        
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, n_cols, figsize=figsize)
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                ax = axes[i, j] if n_cols > 1 else axes
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(data[col1], bins=20, alpha=0.7, color=self.colors[0])
                    ax.set_title(col1)
                elif i < j:
                    # Upper triangle: scatter plot
                    ax.scatter(data[col1], data[col2], alpha=0.6, color=self.colors[1])
                    corr = data[col1].corr(data[col2])
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    # Lower triangle: scatter plot with regression line
                    sns.regplot(data=data, x=col1, y=col2, ax=ax, color=self.colors[2], scatter_kws={'alpha': 0.6})
                
                if i == n_cols - 1:
                    ax.set_xlabel(col2)
                if j == 0:
                    ax.set_ylabel(col1)
        
        plt.suptitle(title or 'Scatter Matrix', fontsize=self.font_size + 4)
        plt.tight_layout()
        return fig
    
    # ADVANCED STATISTICAL PLOTS
    def regression_plot(self, data, x, y, hue=None, title=None, figsize=(10, 8), 
                       order=1, robust=False, ci=95):
        """Create regression plot with confidence intervals."""
        plt.figure(figsize=figsize)
        
        if hue:
            unique_vals = data[hue].unique()
            for i, val in enumerate(unique_vals):
                subset = data[data[hue] == val]
                sns.regplot(data=subset, x=x, y=y, label=str(val), 
                           color=self.colors[i % len(self.colors)],
                           order=order, robust=robust, ci=ci, scatter_kws={'alpha': 0.6})
            plt.legend()
        else:
            sns.regplot(data=data, x=x, y=y, color=self.colors[0], 
                       order=order, robust=robust, ci=ci, scatter_kws={'alpha': 0.6})
        
        plt.title(title or f'Regression: {y} vs {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def residual_plot(self, data, x, y, title=None, figsize=(12, 5)):
        """Create residual plots for regression diagnostics."""
        from sklearn.linear_model import LinearRegression
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Fit linear regression
        X = data[x].values.reshape(-1, 1)
        y_vals = data[y].values
        
        model = LinearRegression()
        model.fit(X, y_vals)
        y_pred = model.predict(X)
        residuals = y_vals - y_pred
        
        # Residual vs Fitted
        axes[0].scatter(y_pred, residuals, alpha=0.6, color=self.colors[0])
        axes[0].axhline(y=0, color=self.colors[1], linestyle='--')
        axes[0].set_xlabel('Fitted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Fitted')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot of residuals
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot of Residuals')
        
        plt.suptitle(title or f'Residual Analysis: {y} vs {x}', fontsize=self.font_size + 2)
        plt.tight_layout()
        return fig
    
    def qq_plot(self, data, column, distribution='norm', title=None, figsize=(8, 6)):
        """Create Q-Q plot for normality testing."""
        plt.figure(figsize=figsize)
        
        stats.probplot(data[column].dropna(), dist=distribution, plot=plt)
        plt.title(title or f'Q-Q Plot of {column}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def outlier_detection_plot(self, data, columns=None, title=None, figsize=(15, 10), method='iqr'):
        """Create comprehensive outlier detection visualization."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns[:4]
        
        n_cols = len(columns)
        fig, axes = plt.subplots(2, n_cols, figsize=figsize)
        
        for i, col in enumerate(columns):
            # Box plot
            if n_cols > 1:
                ax1, ax2 = axes[0, i], axes[1, i]
            else:
                ax1, ax2 = axes[0], axes[1]
            
            sns.boxplot(y=data[col], ax=ax1, color=self.colors[0])
            ax1.set_title(f'Box Plot: {col}')
            
            # Histogram with outliers highlighted
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outliers = data[col][z_scores > 3]
            
            ax2.hist(data[col], bins=30, alpha=0.7, color=self.colors[1], label='Normal')
            ax2.hist(outliers, bins=20, alpha=0.8, color=self.colors[3], label='Outliers')
            ax2.set_title(f'Distribution: {col}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title or 'Outlier Detection Analysis', fontsize=self.font_size + 4)
        plt.tight_layout()
        return fig
    
    def statistical_summary_plot(self, data, column, title=None, figsize=(15, 10)):
        """Create comprehensive statistical summary visualization."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Histogram
        axes[0, 0].hist(data[column], bins=30, alpha=0.7, color=self.colors[0])
        axes[0, 0].set_title('Histogram')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        sns.boxplot(y=data[column], ax=axes[0, 1], color=self.colors[1])
        axes[0, 1].set_title('Box Plot')
        
        # Q-Q plot
        stats.probplot(data[column].dropna(), dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot')
        
        # KDE plot
        sns.kdeplot(data[column], ax=axes[1, 0], color=self.colors[2])
        axes[1, 0].set_title('Density Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Violin plot
        sns.violinplot(y=data[column], ax=axes[1, 1], color=self.colors[3])
        axes[1, 1].set_title('Violin Plot')
        
        # Statistics text
        axes[1, 2].axis('off')
        stats_text = f"""
        Statistical Summary:
        
        Mean: {data[column].mean():.3f}
        Median: {data[column].median():.3f}
        Std Dev: {data[column].std():.3f}
        Min: {data[column].min():.3f}
        Max: {data[column].max():.3f}
        
        Skewness: {data[column].skew():.3f}
        Kurtosis: {data[column].kurtosis():.3f}
        
        Missing Values: {data[column].isnull().sum()}
        """
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=self.font_size-1, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors[0], alpha=0.1))
        
        plt.suptitle(title or f'Statistical Analysis of {column}', fontsize=self.font_size + 4)
        plt.tight_layout()
        return fig
    
    # TIME SERIES PLOTS
    def time_series_plot(self, data, date_col, value_cols, title=None, figsize=(15, 8), 
                        resample_freq=None, rolling_window=None):
        """Create comprehensive time series visualization."""
        plt.figure(figsize=figsize)
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Sort by date
        data_sorted = data.sort_values(date_col)
        
        # Handle multiple value columns
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        
        for i, col in enumerate(value_cols):
            series = data_sorted.set_index(date_col)[col]
            
            # Resample if specified
            if resample_freq:
                series = series.resample(resample_freq).mean()
            
            # Plot original series
            plt.plot(series.index, series.values, label=col, 
                    color=self.colors[i % len(self.colors)], alpha=0.7)
            
            # Add rolling average if specified
            if rolling_window:
                rolling_avg = series.rolling(window=rolling_window).mean()
                plt.plot(rolling_avg.index, rolling_avg.values, 
                        label=f'{col} (Rolling {rolling_window})', 
                        color=self.colors[i % len(self.colors)], linewidth=2)
        
        plt.title(title or 'Time Series Analysis')
        plt.xlabel(date_col)
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def seasonal_decomposition_plot(self, data, date_col, value_col, period=365, 
                                  title=None, figsize=(15, 12)):
        """Create seasonal decomposition plot."""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Prepare data
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
        
        series = data.set_index(date_col)[value_col].sort_index()
        
        # Perform decomposition
        decomposition = seasonal_decompose(series, period=period, model='additive')
        
        # Create plots
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        decomposition.observed.plot(ax=axes[0], color=self.colors[0])
        axes[0].set_title('Original')
        axes[0].grid(True, alpha=0.3)
        
        decomposition.trend.plot(ax=axes[1], color=self.colors[1])
        axes[1].set_title('Trend')
        axes[1].grid(True, alpha=0.3)
        
        decomposition.seasonal.plot(ax=axes[2], color=self.colors[2])
        axes[2].set_title('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        decomposition.resid.plot(ax=axes[3], color=self.colors[3])
        axes[3].set_title('Residual')
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(title or f'Seasonal Decomposition of {value_col}', fontsize=self.font_size + 4)
        plt.tight_layout()
        return fig
    
    # CATEGORICAL PLOTS
    def categorical_analysis(self, data, cat_col, num_col=None, title=None, figsize=(15, 10)):
        """Create comprehensive categorical analysis."""
        if num_col is None:
            # Single categorical variable analysis
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Count plot
            sns.countplot(data=data, x=cat_col, ax=axes[0, 0], palette=self.colors)
            axes[0, 0].set_title(f'Count of {cat_col}')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Pie chart
            value_counts = data[cat_col].value_counts()
            axes[0, 1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                          colors=self.colors[:len(value_counts)])
            axes[0, 1].set_title(f'Distribution of {cat_col}')
            
            # Horizontal bar plot
            sns.countplot(data=data, y=cat_col, ax=axes[1, 0], palette=self.colors)
            axes[1, 0].set_title(f'Count of {cat_col} (Horizontal)')
            
            # Statistics text
            axes[1, 1].axis('off')
            stats_text = f"""
            Category Statistics:
            
            Total Count: {len(data[cat_col])}
            Unique Values: {data[cat_col].nunique()}
            Most Common: {data[cat_col].mode().iloc[0]}
            Missing Values: {data[cat_col].isnull().sum()}
            
            Value Counts:
            {value_counts.to_string()}
            """
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                           fontsize=self.font_size-1, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors[0], alpha=0.1))
            
        else:
            # Categorical vs numerical analysis
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Box plot
            sns.boxplot(data=data, x=cat_col, y=num_col, ax=axes[0, 0], palette=self.colors)
            axes[0, 0].set_title(f'{num_col} by {cat_col}')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Violin plot
            sns.violinplot(data=data, x=cat_col, y=num_col, ax=axes[0, 1], palette=self.colors)
            axes[0, 1].set_title(f'{num_col} Distribution by {cat_col}')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Bar plot (mean)
            sns.barplot(data=data, x=cat_col, y=num_col, ax=axes[1, 0], palette=self.colors)
            axes[1, 0].set_title(f'Mean {num_col} by {cat_col}')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Strip plot
            sns.stripplot(data=data, x=cat_col, y=num_col, ax=axes[1, 1], palette=self.colors, alpha=0.6)
            axes[1, 1].set_title(f'{num_col} Points by {cat_col}')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title or f'Categorical Analysis: {cat_col}', fontsize=self.font_size + 4)
        plt.tight_layout()
        return fig
    
    # INTERACTIVE PLOTS (PLOTLY)
    def interactive_scatter(self, data, x, y, color=None, size=None, hover_data=None, title=None):
        """Create interactive scatter plot using Plotly."""
        fig = px.scatter(data, x=x, y=y, color=color, size=size, hover_data=hover_data,
                        title=title or f'{y} vs {x}',
                        color_discrete_sequence=self.colors)
        
        fig.update_layout(
            font_family=self.font_family,
            font_size=self.font_size,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    def interactive_line_plot(self, data, x, y, color=None, title=None):
        """Create interactive line plot using Plotly."""
        fig = px.line(data, x=x, y=y, color=color,
                     title=title or f'{y} over {x}',
                     color_discrete_sequence=self.colors)
        
        fig.update_layout(
            font_family=self.font_family,
            font_size=self.font_size,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )
        return fig
    
    def interactive_histogram(self, data, column, nbins=30, title=None):
        """Create interactive histogram using Plotly."""
        fig = px.histogram(data, x=column, nbins=nbins,
                          title=title or f'Distribution of {column}')
        
        fig.update_traces(marker_color=self.colors[0])
        fig.update_layout(
            font_family=self.font_family,
            font_size=self.font_size,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    def interactive_box_plot(self, data, x=None, y=None, color=None, title=None):
        """Create interactive box plot using Plotly."""
        fig = px.box(data, x=x, y=y, color=color,
                    title=title or f'Box Plot of {y or x}',
                    color_discrete_sequence=self.colors)
        
        fig.update_layout(
            font_family=self.font_family,
            font_size=self.font_size,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    def interactive_3d_scatter(self, data, x, y, z, color=None, size=None, title=None):
        """Create interactive 3D scatter plot using Plotly."""
        fig = px.scatter_3d(data, x=x, y=y, z=z, color=color, size=size,
                           title=title or f'3D Scatter: {x}, {y}, {z}',
                           color_discrete_sequence=self.colors)
        
        fig.update_layout(
            font_family=self.font_family,
            font_size=self.font_size,
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z,
                bgcolor='white'
            )
        )
        return fig
    
    # COMPLEX MULTI-PLOT LAYOUTS
    def comprehensive_eda_dashboard(self, data, target_col=None, max_cols=4, figsize=(20, 25)):
        """Create comprehensive EDA dashboard with multiple visualizations."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # Calculate layout
        total_plots = len(numeric_cols) + len(categorical_cols) + 2  # +2 for summary and correlation
        n_rows = (total_plots + max_cols - 1) // max_cols
        
        fig, axes = plt.subplots(n_rows, max_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if max_cols == 1 else axes
        
        plot_idx = 0
        
        # Dataset summary
        axes[plot_idx].axis('off')
        summary_text = f"""
        Dataset Summary:
        
        Shape: {data.shape[0]} rows × {data.shape[1]} columns
        
        Numeric Columns: {len(numeric_cols)}
        Categorical Columns: {len(categorical_cols)}
        
        Missing Values: {data.isnull().sum().sum()}
        Duplicates: {data.duplicated().sum()}
        
        Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        """
        axes[plot_idx].text(0.1, 0.9, summary_text, transform=axes[plot_idx].transAxes,
                           fontsize=self.font_size, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors[0], alpha=0.1))
        axes[plot_idx].set_title('Dataset Overview', fontsize=self.font_size + 2)
        plot_idx += 1
        
        # Correlation heatmap (if enough numeric columns)
        if len(numeric_cols) > 1:
            corr_data = data[numeric_cols + ([target_col] if target_col and target_col in data.select_dtypes(include=[np.number]).columns else [])]
            corr_matrix = corr_data.corr()
            
            sns.heatmap(corr_matrix, ax=axes[plot_idx], annot=True, cmap=self.color_palette,
                       center=0, square=True, linewidths=0.5, cbar=False, fmt='.2f')
            axes[plot_idx].set_title('Correlation Matrix', fontsize=self.font_size + 2)
            plot_idx += 1
        
        # Numeric columns distribution
        for col in numeric_cols[:max_cols-2]:  # Leave space for other plots
            if plot_idx < len(axes):
                axes[plot_idx].hist(data[col], bins=30, alpha=0.7, color=self.colors[plot_idx % len(self.colors)])
                axes[plot_idx].set_title(f'Distribution: {col}', fontsize=self.font_size)
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Categorical columns
        for col in categorical_cols[:max_cols-2]:  # Leave space for other plots
            if plot_idx < len(axes):
                value_counts = data[col].value_counts().head(10)  # Top 10 categories
                axes[plot_idx].bar(range(len(value_counts)), value_counts.values,
                                  color=self.colors[plot_idx % len(self.colors)])
                axes[plot_idx].set_title(f'Top Categories: {col}', fontsize=self.font_size)
                axes[plot_idx].set_xticks(range(len(value_counts)))
                axes[plot_idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Comprehensive EDA Dashboard', fontsize=self.font_size + 6, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    
    def comparison_dashboard(self, data1, data2, label1='Dataset 1', label2='Dataset 2', 
                            columns=None, figsize=(20, 15)):
        """Create dashboard comparing two datasets."""
        if columns is None:
            # Get common numeric columns
            cols1 = set(data1.select_dtypes(include=[np.number]).columns)
            cols2 = set(data2.select_dtypes(include=[np.number]).columns)
            columns = list(cols1.intersection(cols2))[:6]  # Limit to 6 for readability
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(columns):
            if i < len(axes):
                # Overlapping histograms
                axes[i].hist(data1[col], bins=30, alpha=0.6, label=label1, color=self.colors[0])
                axes[i].hist(data2[col], bins=30, alpha=0.6, label=label2, color=self.colors[1])
                axes[i].set_title(f'Comparison: {col}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Dataset Comparison: {label1} vs {label2}', fontsize=self.font_size + 4)
        plt.tight_layout()
        return fig
    
    def feature_importance_dashboard(self, data, target_col, top_n=10, figsize=(20, 12)):
        """Create feature importance analysis dashboard."""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        feature_data = data.select_dtypes(include=[np.number]).drop(columns=[target_col])
        target_data = data[target_col]
        
        # Handle categorical target
        is_classification = target_data.dtype == 'object' or target_data.nunique() < 10
        
        if is_classification:
            le = LabelEncoder()
            target_encoded = le.fit_transform(target_data.astype(str))
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            target_encoded = target_data
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit model and get feature importance
        model.fit(feature_data, target_encoded)
        importance_df = pd.DataFrame({
            'feature': feature_data.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Create dashboard
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Feature importance bar plot
        axes[0, 0].barh(importance_df['feature'][::-1], importance_df['importance'][::-1],
                       color=self.colors[0])
        axes[0, 0].set_title('Feature Importance')
        axes[0, 0].set_xlabel('Importance')
        
        # Top features vs target
        top_features = importance_df['feature'].head(5).tolist()
        
        for i, feature in enumerate(top_features[:5]):
            row, col = divmod(i + 1, 3)
            if row < 2 and col < 3:
                if is_classification:
                    sns.boxplot(data=data, x=target_col, y=feature, ax=axes[row, col], 
                               palette=self.colors)
                else:
                    axes[row, col].scatter(data[feature], data[target_col], 
                                         alpha=0.6, color=self.colors[i % len(self.colors)])
                axes[row, col].set_title(f'{feature} vs {target_col}')
                
        # Hide unused subplot
        if len(top_features) < 5:
            axes[1, 2].axis('off')
        
        plt.suptitle(f'Feature Importance Analysis for {target_col}', fontsize=self.font_size + 4)
        plt.tight_layout()
        return fig
    
    def missing_data_analysis(self, data, figsize=(15, 10)):
        """Create comprehensive missing data analysis."""
        missing_data = data.isnull().sum()
        missing_percent = (missing_data / len(data)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percent': missing_percent.values
        }).sort_values('Missing_Count', ascending=False)
        
        # Filter only columns with missing data
        missing_df = missing_df[missing_df['Missing_Count'] > 0]
        
        if missing_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No missing data found!', transform=ax.transAxes,
                   fontsize=self.font_size + 4, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors[0], alpha=0.3))
            ax.axis('off')
            ax.set_title('Missing Data Analysis', fontsize=self.font_size + 6)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Missing data bar plot
        axes[0, 0].barh(missing_df['Column'][::-1], missing_df['Missing_Count'][::-1],
                       color=self.colors[0])
        axes[0, 0].set_title('Missing Data Count')
        axes[0, 0].set_xlabel('Count')
        
        # Missing data percentage
        axes[0, 1].barh(missing_df['Column'][::-1], missing_df['Missing_Percent'][::-1],
                       color=self.colors[1])
        axes[0, 1].set_title('Missing Data Percentage')
        axes[0, 1].set_xlabel('Percentage (%)')
        
        # Missing data heatmap
        sns.heatmap(data.isnull(), ax=axes[1, 0], cbar=True, yticklabels=False,
                   cmap=['white', self.colors[2]])
        axes[1, 0].set_title('Missing Data Pattern')
        axes[1, 0].set_xlabel('Columns')
        
        # Missing data summary
        axes[1, 1].axis('off')
        total_missing = missing_data.sum()
        total_cells = data.shape[0] * data.shape[1]
        missing_summary = f"""
        Missing Data Summary:
        
        Total Missing Values: {total_missing:,}
        Total Data Points: {total_cells:,}
        Overall Missing %: {(total_missing/total_cells)*100:.2f}%
        
        Columns with Missing Data: {len(missing_df)}
        Complete Columns: {data.shape[1] - len(missing_df)}
        
        Worst Column: {missing_df.iloc[0]['Column']}
        ({missing_df.iloc[0]['Missing_Percent']:.1f}% missing)
        """
        axes[1, 1].text(0.1, 0.9, missing_summary, transform=axes[1, 1].transAxes,
                        fontsize=self.font_size-1, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors[3], alpha=0.1))
        
        plt.suptitle('Missing Data Analysis', fontsize=self.font_size + 4)
        plt.tight_layout()
        return fig
    
    def save_all_plots(self, plots_dict, output_dir='plots', dpi=300, format='png'):
        """Save multiple plots to files."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        for plot_name, plot_fig in plots_dict.items():
            filename = f"{plot_name}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            if hasattr(plot_fig, 'write_html'):  # Plotly figure
                plot_fig.write_html(filepath.replace(f'.{format}', '.html'))
                saved_files.append(filepath.replace(f'.{format}', '.html'))
            else:  # Matplotlib figure
                plot_fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
                saved_files.append(filepath)
        
        return saved_files
    
    def create_custom_subplot_layout(self, plot_functions, layout=(2, 2), figsize=(15, 12), 
                                   titles=None, **kwargs):
        """Create custom subplot layout with different plot types."""
        rows, cols = layout
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        for i, (plot_func, plot_kwargs) in enumerate(plot_functions[:len(axes)]):
            ax = axes[i]
            plot_kwargs.update(kwargs)
            
            # Call the plotting function with the specific axis
            if hasattr(self, plot_func):
                getattr(self, plot_func)(ax=ax, **plot_kwargs)
            
            if titles and i < len(titles):
                ax.set_title(titles[i])
        
        # Hide unused subplots
        for idx in range(len(plot_functions), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig