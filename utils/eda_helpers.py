import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SolarDataAnalyzer:
    def __init__(self, country_name, file_path):
        self.country = country_name
        self.file_path = file_path
        self.df = None
        self.clean_df = None
        
    def load_data(self):
        """Load and prepare the solar dataset"""
        print(f"Loading data for {self.country}...")
        self.df = pd.read_csv(self.file_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print("First 5 rows:")
        print(self.df.head())
        
        # Convert timestamp
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df = self.df.sort_values('Timestamp')
        
        return self.df

    def summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\n" + "="*60)
        print(f"SUMMARY STATISTICS - {self.country.upper()}")
        print("="*60)
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nLast 5 rows:")
        print(self.df.tail())
        
        print("\nRandom sample of 5 rows:")
        print(self.df.sample(5))
        
        print("\nDataset info:")
        print(self.df.info())
        
        print("\nMissing values:")
        missing_report = self.df.isna().sum()
        print(missing_report)
        
        print("\nColumns with >5% nulls:")
        missing_pct = (missing_report / len(self.df)) * 100
        high_missing = missing_pct[missing_pct > 5]
        print(high_missing)
        
        print("\nDescription of numeric columns:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe())
        
        # Check for categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            print("\nDescription of categorical columns:")
            print(self.df[cat_cols].describe())
        
        # Exact duplicate rows
        dup_count = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {dup_count}")
        
        # Cardinality for categoricals
        if len(cat_cols) > 0:
            cardinality = {c: self.df[c].nunique() for c in cat_cols}
            print(f"Cardinality (categoricals): {cardinality}")

    def univariate_analysis(self):
        """Perform univariate analysis on all numeric columns"""
        print("\n" + "="*60)
        print(f"UNIVARIATE ANALYSIS - {self.country.upper()}")
        print("="*60)
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\nAnalyzing {len(num_cols)} numeric columns...")
        
        # Histograms for all numeric columns
        for c in num_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[c], kde=True)
            plt.title(f"Distribution: {c} - {self.country}")
            plt.xlabel(c)
            plt.ylabel("Count")
            plt.show()
        
        # Box plots for all numeric columns
        for c in num_cols:
            plt.figure(figsize=(8, 6))
            plt.boxplot(self.df[c].dropna(), vert=True)
            plt.title(f"Box plot: {c} - {self.country}")
            plt.ylabel(c)
            plt.show()

    def outlier_detection(self):
        print("\n" + "="*60)
        print(f"OUTLIER DETECTION - {self.country.upper()}")
        print("="*60)
        
        # Start with all data
        self.clean_df = self.df.copy()
        original_rows = len(self.clean_df)
        
        print("Using conservative cleaning - only removing physically impossible values")
        
        if 'GHI' in self.clean_df.columns:
            invalid_ghi = self.clean_df['GHI'] < 0
            if invalid_ghi.any():
                print(f"Removing {invalid_ghi.sum()} rows with negative GHI")
                self.clean_df = self.clean_df[~invalid_ghi]
        
        # Impute missing values (but don't remove any for being outliers)
        numeric_cols = self.clean_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.clean_df[col].isna().sum() > 0:
                na_count = self.clean_df[col].isna().sum()
                self.clean_df[col] = self.clean_df[col].fillna(self.clean_df[col].median())
                print(f"Imputed {na_count} missing values in {col}")
        
        print(f"Original: {original_rows} rows, Cleaned: {len(self.clean_df)} rows")
        
        # Save
        self.clean_df.to_csv(f"../data/{self.country}_clean.csv", index=False)
        print(f"Cleaned dataset saved to: data/{self.country}_clean.csv")
    
        return self.clean_df
        
    def time_series_analysis(self):
        """Analyze time series patterns in all solar parameters"""
        print("\n" + "="*60)
        print(f"TIME SERIES ANALYSIS - {self.country.upper()}")
        print("="*60)
        
        if self.clean_df is None:
            self.clean_df = self.df.copy()
        
        # Analyze all numeric columns in time series
        numeric_cols = self.clean_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Group columns for better visualization
        irradiance_cols = [col for col in numeric_cols if any(x in col for x in ['GHI', 'DNI', 'DHI'])]
        temp_cols = [col for col in numeric_cols if any(x in col for x in ['Tamb', 'TModA', 'TModB'])]
        wind_cols = [col for col in numeric_cols if any(x in col for x in ['WS', 'WD'])]
        other_cols = [col for col in numeric_cols if col not in irradiance_cols + temp_cols + wind_cols]
        
        # Plot irradiance parameters
        if irradiance_cols:
            fig, axes = plt.subplots(len(irradiance_cols), 1, figsize=(15, 4*len(irradiance_cols)))
            if len(irradiance_cols) == 1:
                axes = [axes]
            for i, col in enumerate(irradiance_cols):
                axes[i].plot(self.clean_df['Timestamp'], self.clean_df[col], alpha=0.7, linewidth=0.5)
                axes[i].set_title(f'{col} Over Time')
                axes[i].set_ylabel(col)
                axes[i].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()
        
        # Plot temperature parameters
        if temp_cols:
            fig, axes = plt.subplots(len(temp_cols), 1, figsize=(15, 4*len(temp_cols)))
            if len(temp_cols) == 1:
                axes = [axes]
            for i, col in enumerate(temp_cols):
                axes[i].plot(self.clean_df['Timestamp'], self.clean_df[col], alpha=0.7, linewidth=0.5, color='red')
                axes[i].set_title(f'{col} Over Time')
                axes[i].set_ylabel(col)
                axes[i].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()
        
        # Plot wind parameters
        if wind_cols:
            fig, axes = plt.subplots(len(wind_cols), 1, figsize=(15, 4*len(wind_cols)))
            if len(wind_cols) == 1:
                axes = [axes]
            for i, col in enumerate(wind_cols):
                axes[i].plot(self.clean_df['Timestamp'], self.clean_df[col], alpha=0.7, linewidth=0.5, color='green')
                axes[i].set_title(f'{col} Over Time')
                axes[i].set_ylabel(col)
                axes[i].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

    def bivariate_analysis(self):
        """Perform bivariate analysis on solar parameters"""
        print("\n" + "="*60)
        print(f"BIVARIATE ANALYSIS - {self.country.upper()}")
        print("="*60)
        
        # Key relationships to analyze
        relationships = [
            ('Tamb', 'GHI', 'Temperature vs GHI'),
            ('RH', 'GHI', 'Relative Humidity vs GHI'),
            ('WS', 'GHI', 'Wind Speed vs GHI'),
            ('BP', 'GHI', 'Pressure vs GHI'),
            ('Tamb', 'DNI', 'Temperature vs DNI'),
            ('Tamb', 'DHI', 'Temperature vs DHI'),
            ('WS', 'WSgust', 'Wind Speed vs Wind Gust'),
            ('ModA', 'ModB', 'Module A vs Module B'),
            ('TModA', 'TModB', 'Module Temp A vs Module Temp B')
        ]
        
        for x_col, y_col, title in relationships:
            if x_col in self.clean_df.columns and y_col in self.clean_df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.clean_df[x_col], self.clean_df[y_col], alpha=0.5)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'{title} - {self.country}')
                plt.show()

    def correlation_analysis(self):
        """Perform correlation analysis on all numeric parameters"""
        print("\n" + "="*60)
        print(f"CORRELATION ANALYSIS - {self.country.upper()}")
        print("="*60)
        
        # Correlation matrix for all numeric parameters
        numeric_cols = self.clean_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr = self.clean_df[numeric_cols].corr()
            print("Correlation Matrix:")
            print(corr)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, center=0)
            plt.title(f"Correlation Matrix - {self.country}")
            plt.tight_layout()
            plt.show()
            
            # Print strong correlations (|r| > 0.7)
            print("\nStrong correlations (|r| > 0.7):")
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        print(f"{corr.columns[i]} vs {corr.columns[j]}: {corr.iloc[i, j]:.3f}")

    def cleaning_impact_analysis(self):
        """Analyze impact of cleaning on module performance"""
        print("\n" + "="*60)
        print(f"CLEANING IMPACT ANALYSIS - {self.country.upper()}")
        print("="*60)
        
        if 'Cleaning' in self.clean_df.columns:
            # Analyze all module-related columns
            module_cols = [col for col in self.clean_df.columns if 'Mod' in col]
            
            if module_cols:
                cleaning_effect = self.clean_df.groupby('Cleaning')[module_cols].mean()
                print("Average module readings by cleaning status:")
                print(cleaning_effect)
                
                # Plotting
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Bar plot
                cleaning_effect.plot(kind='bar', ax=axes[0])
                axes[0].set_title('Module Readings by Cleaning Status')
                axes[0].set_ylabel('Reading Value')
                axes[0].tick_params(axis='x', rotation=0)
                
                # Line plot for trend
                if len(cleaning_effect) > 1:
                    cleaning_effect.plot(kind='line', marker='o', ax=axes[1])
                    axes[1].set_title('Module Performance: Pre vs Post Cleaning')
                    axes[1].set_ylabel('Reading Value')
                
                plt.tight_layout()
                plt.show()

    def wind_analysis(self):
        """Comprehensive wind analysis"""
        print("\n" + "="*60)
        print(f"WIND ANALYSIS - {self.country.upper()}")
        print("="*60)
        
        wind_cols = [col for col in self.clean_df.columns if any(x in col for x in ['WS', 'WD'])]
        
        if wind_cols:
            print(f"Analyzing wind parameters: {wind_cols}")
            
            # Wind speed analysis
            if 'WS' in self.clean_df.columns:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.hist(self.clean_df['WS'].dropna(), bins=50, alpha=0.7, edgecolor='black')
                plt.title('Wind Speed Distribution')
                plt.xlabel('Wind Speed (m/s)')
                plt.ylabel('Frequency')
                
                if 'WSgust' in self.clean_df.columns:
                    plt.subplot(2, 2, 2)
                    plt.scatter(self.clean_df['WS'], self.clean_df['WSgust'], alpha=0.5)
                    plt.xlabel('Wind Speed (m/s)')
                    plt.ylabel('Wind Gust (m/s)')
                    plt.title('Wind Speed vs Wind Gust')
                
                if 'WSstdev' in self.clean_df.columns:
                    plt.subplot(2, 2, 3)
                    plt.scatter(self.clean_df['WS'], self.clean_df['WSstdev'], alpha=0.5)
                    plt.xlabel('Wind Speed (m/s)')
                    plt.ylabel('Wind Speed Std Dev')
                    plt.title('Wind Speed vs Variability')
                
                plt.tight_layout()
                plt.show()
            
            # Wind direction analysis
            if 'WD' in self.clean_df.columns:
                plt.figure(figsize=(10, 6))
                plt.hist(self.clean_df['WD'].dropna(), bins=36, alpha=0.7, edgecolor='black')
                plt.title('Wind Direction Distribution')
                plt.xlabel('Wind Direction (degrees)')
                plt.ylabel('Frequency')
                plt.show()
                
                if 'WDstdev' in self.clean_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(self.clean_df['WD'], self.clean_df['WDstdev'], alpha=0.5)
                    plt.xlabel('Wind Direction (degrees)')
                    plt.ylabel('Wind Direction Std Dev')
                    plt.title('Wind Direction vs Variability')
                    plt.show()

    def temperature_analysis(self):
        """Comprehensive temperature analysis"""
        print("\n" + "="*60)
        print(f"TEMPERATURE ANALYSIS - {self.country.upper()}")
        print("="*60)
        
        temp_cols = [col for col in self.clean_df.columns if any(x in col for x in ['Tamb', 'TMod'])]
        
        if temp_cols:
            print(f"Analyzing temperature parameters: {temp_cols}")
            
            # Temperature relationships
            if 'Tamb' in self.clean_df.columns and 'RH' in self.clean_df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.clean_df['Tamb'], self.clean_df['RH'], alpha=0.5, color='purple')
                plt.xlabel('Ambient Temperature (°C)')
                plt.ylabel('Relative Humidity (%)')
                plt.title('Temperature vs Relative Humidity')
                plt.show()
            
            # Module temperature comparison
            if 'TModA' in self.clean_df.columns and 'TModB' in self.clean_df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.clean_df['TModA'], self.clean_df['TModB'], alpha=0.5, color='orange')
                plt.xlabel('Module Temperature A (°C)')
                plt.ylabel('Module Temperature B (°C)')
                plt.title('Module Temperature A vs B')
                plt.show()

    def bubble_chart_analysis(self):
        """Create multiple bubble charts for multivariate relationships"""
        print("\n" + "="*60)
        print(f"BUBBLE CHART ANALYSIS - {self.country.upper()}")
        print("="*60)
        
        # Multiple bubble chart configurations
        bubble_configs = [
            ('Tamb', 'GHI', 'RH', 'Temperature vs GHI (RH as bubble size)'),
            ('WS', 'GHI', 'BP', 'Wind Speed vs GHI (Pressure as bubble size)'),
            ('Tamb', 'DNI', 'RH', 'Temperature vs DNI (RH as bubble size)'),
        ]
        
        for x_col, y_col, size_col, title in bubble_configs:
            if all(col in self.clean_df.columns for col in [x_col, y_col, size_col]):
                plt.figure(figsize=(12, 8))
                
                # Normalize bubble size
                size_normalized = (self.clean_df[size_col] - self.clean_df[size_col].min()) / \
                                (self.clean_df[size_col].max() - self.clean_df[size_col].min())
                bubble_sizes = 50 + size_normalized * 200
                
                scatter = plt.scatter(self.clean_df[x_col], self.clean_df[y_col], 
                                    s=bubble_sizes, alpha=0.6, c=self.clean_df[size_col], 
                                    cmap='viridis')
                plt.colorbar(scatter, label=size_col)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'{self.country} - {title}')
                plt.tight_layout()
                plt.show()

    def comprehensive_analysis(self):
        """Run complete EDA pipeline"""
        print(f"🚀 STARTING COMPREHENSIVE EDA FOR {self.country.upper()}")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.summary_statistics()
        self.univariate_analysis()
        self.outlier_detection()
        self.time_series_analysis()
        self.bivariate_analysis()
        self.correlation_analysis()
        self.cleaning_impact_analysis()
        self.wind_analysis()
        self.temperature_analysis()
        self.bubble_chart_analysis()
        
        print(f"EDA COMPLETED FOR {self.country.upper()}")
        print(f"Cleaned data saved to: data/{self.country}_clean.csv")
        print("="*80)
