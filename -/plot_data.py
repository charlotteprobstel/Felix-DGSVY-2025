import pandas as pd
import numpy as np
import os
import warnings
from itertools import combinations, permutations
from typing import List, Dict, Tuple, Any, Optional
import json
from datetime import datetime
from visualise import EDAVisualiser

warnings.filterwarnings('ignore')


class AutoEDAAnalyzer:
    """
    Automated EDA Analysis class that iterates through column combinations,
    detects data types, and generates appropriate visualizations.
    """
    
    def __init__(self, data_path: str = 'data/cleaned_data.csv', 
                 output_dir: str = 'auto_eda_results',
                 color_theme: str = 'viridis',
                 font_family: str = 'Arial',
                 font_size: int = 10,
                 max_combinations: int = 100):
        """
        Initialize the AutoEDAAnalyzer.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV data file
        output_dir : str
            Directory to save generated plots and results
        color_theme : str
            Color theme for visualizations
        font_family : str
            Font family for plots
        font_size : int
            Font size for plots
        max_combinations : int
            Maximum number of column combinations to analyze
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_combinations = max_combinations
        
        # Initialize visualizer with custom theme
        self.viz = EDAVisualiser(
            color_palette=color_theme,
            font_family=font_family,
            font_size=font_size
        )
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load and prepare data
        self.data = self._load_and_clean_data()
        self.column_types = self._detect_column_types()
        self.plot_results = {}
        self.execution_log = []
        
        print(f"Loaded dataset with shape: {self.data.shape}")
        print(f"Column types detected: {len(self.column_types)} columns categorized")
    
    def _load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the dataset."""
        try:
            # Load data, handling the complex header structure
            data = pd.read_csv(self.data_path)
            
            # Remove metadata rows (first few rows that don't contain actual data)
            # Find the first row that contains actual survey responses
            actual_data_start = 0
            for i, row in data.iterrows():
                # Look for a row that doesn't contain JSON-like strings or metadata
                if not any(str(val).startswith('{"ImportId"') for val in row.values if pd.notna(val)):
                    if not any(str(val).startswith('How old') for val in row.values if pd.notna(val)):
                        if not str(row.iloc[0]).startswith('Q'):
                            actual_data_start = i
                            break
            
            if actual_data_start > 0:
                data = data.iloc[actual_data_start:].reset_index(drop=True)
            
            # Convert numeric-like columns
            for col in data.columns:
                # Try to convert to numeric if possible
                try:
                    # Replace common non-numeric placeholders
                    temp_col = data[col].replace({
                        'SKIP': np.nan,
                        '': np.nan,
                        'nan': np.nan,
                        'N/A': np.nan,
                        'n/a': np.nan
                    })
                    
                    # Check if column looks numeric
                    numeric_col = pd.to_numeric(temp_col, errors='ignore')
                    if not numeric_col.equals(temp_col):
                        data[col] = numeric_col
                except:
                    continue
            
            # Remove columns that are mostly empty or have JSON-like strings
            cols_to_drop = []
            for col in data.columns:
                # Drop columns with high percentage of missing values
                missing_pct = data[col].isnull().sum() / len(data)
                if missing_pct > 0.8:
                    cols_to_drop.append(col)
                    continue
                
                # Drop columns that contain mostly JSON-like strings
                json_count = data[col].astype(str).str.contains('ImportId|QID', na=False).sum()
                if json_count / len(data) > 0.5:
                    cols_to_drop.append(col)
            
            data = data.drop(columns=cols_to_drop)
            
            print(f"Cleaned data: removed {len(cols_to_drop)} columns with high missing values or metadata")
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create sample data for demonstration
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration if real data can't be loaded."""
        np.random.seed(42)
        n_samples = 500
        
        sample_data = {
            'age': np.random.randint(18, 65, n_samples),
            'income': np.random.normal(50000, 15000, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
            'satisfaction_score': np.random.uniform(1, 10, n_samples),
            'department': np.random.choice(['Engineering', 'Science', 'Arts', 'Business'], n_samples),
            'year_of_study': np.random.choice(['First', 'Second', 'Third', 'Fourth'], n_samples),
            'gpa': np.random.uniform(2.0, 4.0, n_samples),
            'hours_studied': np.random.poisson(20, n_samples),
            'extracurricular': np.random.choice(['Yes', 'No'], n_samples)
        }
        
        return pd.DataFrame(sample_data)
    
    def _detect_column_types(self) -> Dict[str, Dict]:
        """
        Detect and categorize column data types.
        
        Returns:
        --------
        Dict with column names as keys and type info as values
        """
        column_info = {}
        
        for col in self.data.columns:
            series = self.data[col]
            
            # Basic pandas dtype
            pandas_dtype = str(series.dtype)
            
            # Inferred pandas type
            inferred_type = pd.api.types.infer_dtype(series, skipna=True)
            
            # Custom categorization
            category = self._categorize_column_type(series, pandas_dtype, inferred_type)
            
            # Additional statistics
            unique_count = series.nunique()
            missing_count = series.isnull().sum()
            missing_pct = missing_count / len(series)
            
            column_info[col] = {
                'pandas_dtype': pandas_dtype,
                'inferred_type': inferred_type,
                'category': category,
                'unique_count': unique_count,
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'sample_values': series.dropna().head(5).tolist() if not series.empty else []
            }
        
        return column_info
    
    def _categorize_column_type(self, series: pd.Series, pandas_dtype: str, inferred_type: str) -> str:
        """Categorize column into visualization-relevant types."""
        # Numeric types
        if pandas_dtype in ['int64', 'int32', 'float64', 'float32'] or inferred_type in ['integer', 'floating']:
            if series.nunique() <= 10 and series.min() >= 0:
                return 'categorical_numeric'  # Like rating scales
            return 'continuous_numeric'
        
        # Categorical types
        if inferred_type in ['string', 'object'] or pandas_dtype == 'category':
            if series.nunique() <= 20:
                return 'categorical'
            return 'high_cardinality_categorical'
        
        # Boolean
        if inferred_type == 'boolean' or pandas_dtype == 'bool':
            return 'boolean'
        
        # Datetime
        if inferred_type in ['datetime', 'datetime64'] or pandas_dtype.startswith('datetime'):
            return 'datetime'
        
        # Default
        return 'mixed'
    
    def _get_appropriate_plots(self, col1_info: Dict, col2_info: Dict = None) -> List[str]:
        """
        Determine appropriate plot types based on column type combinations.
        
        Parameters:
        -----------
        col1_info : Dict
            Information about first column
        col2_info : Dict
            Information about second column (None for single column plots)
        
        Returns:
        --------
        List of appropriate plot method names
        """
        plots = []
        
        if col2_info is None:
            # Single column plots
            if col1_info['category'] == 'continuous_numeric':
                plots.extend(['histogram', 'statistical_summary_plot'])
                if col1_info['missing_pct'] > 0:
                    plots.append('qq_plot')
            
            elif col1_info['category'] in ['categorical', 'boolean']:
                plots.append('categorical_analysis')
            
            elif col1_info['category'] == 'categorical_numeric':
                plots.extend(['histogram', 'categorical_analysis'])
        
        else:
            # Two column plots
            cat1, cat2 = col1_info['category'], col2_info['category']
            
            # Numeric vs Numeric
            if cat1 == 'continuous_numeric' and cat2 == 'continuous_numeric':
                plots.extend(['scatter_plot', 'regression_plot'])
                if col1_info['unique_count'] < 100 and col2_info['unique_count'] < 100:
                    plots.append('residual_plot')
            
            # Categorical vs Numeric
            elif (cat1 in ['categorical', 'boolean'] and cat2 == 'continuous_numeric') or \
                 (cat2 in ['categorical', 'boolean'] and cat1 == 'continuous_numeric'):
                plots.extend(['box_plot', 'violin_plot', 'bar_plot'])
            
            # Categorical vs Categorical
            elif cat1 in ['categorical', 'boolean'] and cat2 in ['categorical', 'boolean']:
                # Create contingency table analysis (using bar plot)
                plots.append('bar_plot')
            
            # Mixed types
            elif cat1 == 'categorical_numeric' or cat2 == 'categorical_numeric':
                plots.extend(['scatter_plot', 'box_plot'])
        
        return plots
    
    def _generate_single_column_plots(self, column: str) -> Dict[str, Any]:
        """Generate plots for a single column."""
        results = {}
        col_info = self.column_types[column]
        
        try:
            plots_to_create = self._get_appropriate_plots(col_info)
            
            for plot_type in plots_to_create:
                try:
                    if plot_type == 'histogram':
                        fig = self.viz.histogram(self.data, column, 
                                               title=f'Distribution of {column}')
                        results[f'{column}_{plot_type}'] = fig
                    
                    elif plot_type == 'statistical_summary_plot':
                        fig = self.viz.statistical_summary_plot(self.data, column,
                                                               title=f'Statistical Summary: {column}')
                        results[f'{column}_{plot_type}'] = fig
                    
                    elif plot_type == 'qq_plot':
                        fig = self.viz.qq_plot(self.data, column,
                                             title=f'Q-Q Plot: {column}')
                        results[f'{column}_{plot_type}'] = fig
                    
                    elif plot_type == 'categorical_analysis':
                        fig = self.viz.categorical_analysis(self.data, column,
                                                          title=f'Categorical Analysis: {column}')
                        results[f'{column}_{plot_type}'] = fig
                    
                except Exception as e:
                    self.execution_log.append(f"Error creating {plot_type} for {column}: {e}")
            
        except Exception as e:
            self.execution_log.append(f"Error processing column {column}: {e}")
        
        return results
    
    def _generate_two_column_plots(self, col1: str, col2: str) -> Dict[str, Any]:
        """Generate plots for two columns."""
        results = {}
        col1_info = self.column_types[col1]
        col2_info = self.column_types[col2]
        
        try:
            plots_to_create = self._get_appropriate_plots(col1_info, col2_info)
            
            for plot_type in plots_to_create:
                try:
                    if plot_type == 'scatter_plot':
                        # Determine which should be x and y
                        if col1_info['category'] == 'continuous_numeric':
                            x, y = col1, col2
                        else:
                            x, y = col2, col1
                        
                        fig = self.viz.scatter_plot(self.data, x, y,
                                                  title=f'{y} vs {x}')
                        results[f'{col1}_{col2}_{plot_type}'] = fig
                    
                    elif plot_type == 'regression_plot':
                        if col1_info['category'] == 'continuous_numeric':
                            x, y = col1, col2
                        else:
                            x, y = col2, col1
                        
                        fig = self.viz.regression_plot(self.data, x, y,
                                                     title=f'Regression: {y} vs {x}')
                        results[f'{col1}_{col2}_{plot_type}'] = fig
                    
                    elif plot_type == 'residual_plot':
                        if col1_info['category'] == 'continuous_numeric':
                            x, y = col1, col2
                        else:
                            x, y = col2, col1
                        
                        fig = self.viz.residual_plot(self.data, x, y,
                                                   title=f'Residual Analysis: {y} vs {x}')
                        results[f'{col1}_{col2}_{plot_type}'] = fig
                    
                    elif plot_type == 'box_plot':
                        # Determine categorical and numeric columns
                        if col1_info['category'] in ['categorical', 'boolean']:
                            x, y = col1, col2
                        else:
                            x, y = col2, col1
                        
                        fig = self.viz.box_plot(self.data, x, y,
                                              title=f'{y} by {x}')
                        results[f'{col1}_{col2}_{plot_type}'] = fig
                    
                    elif plot_type == 'violin_plot':
                        if col1_info['category'] in ['categorical', 'boolean']:
                            x, y = col1, col2
                        else:
                            x, y = col2, col1
                        
                        fig = self.viz.violin_plot(self.data, x, y,
                                                 title=f'{y} Distribution by {x}')
                        results[f'{col1}_{col2}_{plot_type}'] = fig
                    
                    elif plot_type == 'bar_plot':
                        if col1_info['category'] in ['categorical', 'boolean']:
                            if col2_info['category'] == 'continuous_numeric':
                                x, y = col1, col2
                            else:
                                x, y = col1, None  # Count plot
                        else:
                            x, y = col2, col1
                        
                        fig = self.viz.bar_plot(self.data, x, y,
                                              title=f'{y or "Count"} by {x}')
                        results[f'{col1}_{col2}_{plot_type}'] = fig
                    
                except Exception as e:
                    self.execution_log.append(f"Error creating {plot_type} for {col1} vs {col2}: {e}")
        
        except Exception as e:
            self.execution_log.append(f"Error processing columns {col1} vs {col2}: {e}")
        
        return results
    
    def run_complete_analysis(self, include_single_plots: bool = True,
                            include_pair_plots: bool = True,
                            include_dashboard: bool = True,
                            max_single_plots: int = 20,
                            max_pair_plots: int = 50) -> Dict[str, Any]:
        """
        Run complete automated EDA analysis.
        
        Parameters:
        -----------
        include_single_plots : bool
            Whether to create single-column plots
        include_pair_plots : bool
            Whether to create two-column plots
        include_dashboard : bool
            Whether to create comprehensive dashboard
        max_single_plots : int
            Maximum number of single-column plots
        max_pair_plots : int
            Maximum number of two-column plots
        
        Returns:
        --------
        Dictionary containing all generated plots and analysis results
        """
        print("Starting automated EDA analysis...")
        
        # Single column analysis
        if include_single_plots:
            print("Generating single-column visualizations...")
            eligible_columns = [col for col in self.data.columns 
                              if self.column_types[col]['missing_pct'] < 0.9]
            
            single_cols_to_analyze = eligible_columns[:max_single_plots]
            
            for col in single_cols_to_analyze:
                print(f"  Processing column: {col}")
                single_results = self._generate_single_column_plots(col)
                self.plot_results.update(single_results)
        
        # Two column analysis
        if include_pair_plots:
            print("Generating two-column visualizations...")
            
            # Get columns suitable for pairing
            eligible_columns = [col for col in self.data.columns
                              if self.column_types[col]['missing_pct'] < 0.8 and
                              self.column_types[col]['unique_count'] > 1]
            
            # Generate column pairs
            column_pairs = list(combinations(eligible_columns, 2))
            
            # Limit pairs and prioritize interesting combinations
            pairs_to_analyze = self._prioritize_column_pairs(column_pairs)[:max_pair_plots]
            
            for col1, col2 in pairs_to_analyze:
                print(f"  Processing pair: {col1} vs {col2}")
                pair_results = self._generate_two_column_plots(col1, col2)
                self.plot_results.update(pair_results)
        
        # Dashboard plots
        if include_dashboard:
            print("Generating comprehensive dashboards...")
            try:
                # Overall EDA dashboard
                dashboard = self.viz.comprehensive_eda_dashboard(
                    self.data,
                    title="Complete Dataset Overview"
                )
                self.plot_results['comprehensive_dashboard'] = dashboard
                
                # Missing data analysis
                missing_analysis = self.viz.missing_data_analysis(
                    self.data,
                    title="Missing Data Analysis"
                )
                self.plot_results['missing_data_dashboard'] = missing_analysis
                
                # Correlation analysis for numeric columns
                numeric_cols = [col for col in self.data.columns
                               if self.column_types[col]['category'] == 'continuous_numeric']
                
                if len(numeric_cols) > 1:
                    correlation_heatmap = self.viz.correlation_heatmap(
                        self.data[numeric_cols],
                        title="Correlation Matrix - Numeric Variables"
                    )
                    self.plot_results['correlation_heatmap'] = correlation_heatmap
                
            except Exception as e:
                self.execution_log.append(f"Error creating dashboards: {e}")
        
        print(f"Analysis complete! Generated {len(self.plot_results)} visualizations.")
        return self.plot_results
    
    def _prioritize_column_pairs(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Prioritize column pairs based on potential interest."""
        scored_pairs = []
        
        for col1, col2 in pairs:
            score = 0
            col1_info = self.column_types[col1]
            col2_info = self.column_types[col2]
            
            # Prefer numeric vs categorical combinations
            if (col1_info['category'] == 'continuous_numeric' and 
                col2_info['category'] in ['categorical', 'boolean']) or \
               (col2_info['category'] == 'continuous_numeric' and 
                col1_info['category'] in ['categorical', 'boolean']):
                score += 3
            
            # Prefer numeric vs numeric
            if (col1_info['category'] == 'continuous_numeric' and 
                col2_info['category'] == 'continuous_numeric'):
                score += 2
            
            # Penalize high missing data
            avg_missing = (col1_info['missing_pct'] + col2_info['missing_pct']) / 2
            score -= avg_missing * 2
            
            # Prefer reasonable cardinality
            if col1_info['unique_count'] < 50 and col2_info['unique_count'] < 50:
                score += 1
            
            scored_pairs.append((score, col1, col2))
        
        # Sort by score descending
        scored_pairs.sort(reverse=True, key=lambda x: x[0])
        
        return [(col1, col2) for _, col1, col2 in scored_pairs]
    
    def save_all_results(self, save_plots: bool = True, save_metadata: bool = True,
                        plot_format: str = 'png', dpi: int = 300) -> Dict[str, str]:
        """Save all analysis results to files."""
        saved_files = {}
        
        if save_plots and self.plot_results:
            print(f"Saving {len(self.plot_results)} plots...")
            plots_dir = os.path.join(self.output_dir, 'plots')
            plot_files = self.viz.save_all_plots(
                self.plot_results, 
                output_dir=plots_dir,
                format=plot_format,
                dpi=dpi
            )
            saved_files['plots'] = plot_files
        
        if save_metadata:
            # Save column type analysis
            metadata_file = os.path.join(self.output_dir, 'column_analysis.json')
            with open(metadata_file, 'w') as f:
                json.dump(self.column_types, f, indent=2, default=str)
            saved_files['column_analysis'] = metadata_file
            
            # Save execution log
            log_file = os.path.join(self.output_dir, 'execution_log.txt')
            with open(log_file, 'w') as f:
                f.write(f"Auto EDA Analysis - {datetime.now()}\n")
                f.write("="*50 + "\n\n")
                f.write(f"Dataset shape: {self.data.shape}\n")
                f.write(f"Total plots generated: {len(self.plot_results)}\n\n")
                f.write("Execution Log:\n")
                for entry in self.execution_log:
                    f.write(f"- {entry}\n")
            saved_files['execution_log'] = log_file
            
            # Save data summary
            summary_file = os.path.join(self.output_dir, 'data_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("Dataset Summary\n")
                f.write("="*50 + "\n")
                f.write(f"Shape: {self.data.shape}\n")
                f.write(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
                f.write("Column Types:\n")
                for col, info in self.column_types.items():
                    f.write(f"{col}: {info['category']} ({info['unique_count']} unique, "
                           f"{info['missing_pct']:.1%} missing)\n")
            saved_files['data_summary'] = summary_file
        
        print(f"Results saved to: {self.output_dir}")
        return saved_files
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        numeric_cols = [col for col, info in self.column_types.items() 
                       if info['category'] == 'continuous_numeric']
        categorical_cols = [col for col, info in self.column_types.items() 
                           if info['category'] in ['categorical', 'boolean']]
        
        report = {
            'dataset_info': {
                'shape': self.data.shape,
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
                'total_missing_values': self.data.isnull().sum().sum(),
                'duplicate_rows': self.data.duplicated().sum()
            },
            'column_summary': {
                'total_columns': len(self.data.columns),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols),
                'high_missing_columns': len([col for col, info in self.column_types.items() 
                                           if info['missing_pct'] > 0.5])
            },
            'visualization_summary': {
                'total_plots_generated': len(self.plot_results),
                'single_column_plots': len([k for k in self.plot_results.keys() 
                                          if '_vs_' not in k and 'dashboard' not in k]),
                'two_column_plots': len([k for k in self.plot_results.keys() 
                                       if '_vs_' in k or any(plot_type in k for plot_type in 
                                                           ['scatter_plot', 'regression_plot', 'box_plot'])]),
                'dashboard_plots': len([k for k in self.plot_results.keys() 
                                      if 'dashboard' in k])
            },
            'column_types': self.column_types,
            'execution_errors': self.execution_log
        }
        
        return report


def main():
    """Main execution function demonstrating the AutoEDAAnalyzer."""
    print("Starting Automated EDA Analysis...")
    
    # Initialize analyzer
    analyzer = AutoEDAAnalyzer(
        data_path='data/cleaned_data.csv',
        output_dir='auto_eda_results',
        color_theme='viridis',
        max_combinations=50
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        include_single_plots=True,
        include_pair_plots=True,
        include_dashboard=True,
        max_single_plots=15,
        max_pair_plots=30
    )
    
    # Save all results
    saved_files = analyzer.save_all_results(
        save_plots=True,
        save_metadata=True,
        plot_format='png',
        dpi=300
    )
    
    # Print analysis report
    report = analyzer.get_analysis_report()
    print("\nAnalysis Complete!")
    print(f"Generated {report['visualization_summary']['total_plots_generated']} plots")
    print(f"Processed {report['column_summary']['total_columns']} columns")
    print(f"Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()