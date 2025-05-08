import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def load_and_inspect_data(file_path):
    """Load the data and perform initial inspection"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    print("\nInitial Data Overview:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    
    print("\nFirst few rows of the data:")
    print(df.head())
    
    print("\nData types of columns:")
    print(df.dtypes)
    
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    return df

def clean_data(df):
    """Clean the dataset"""
    print("\nCleaning data...")
    
    # Make a copy of the dataframe
    df_clean = df.copy()
    
    # Convert date columns to datetime if they exist
    date_columns = df_clean.select_dtypes(include=['object']).columns
    for col in date_columns:
        try:
            df_clean[col] = pd.to_datetime(df_clean[col])
            print(f"Converted {col} to datetime")
        except:
            pass
    
    # Handle missing values
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df_clean[col].isnull().sum() > 0:
            # Fill missing values with median for numeric columns
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            print(f"Filled missing values in {col} with median")
    
    # Remove duplicates if any
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    if len(df_clean) < initial_rows:
        print(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    return df_clean

def analyze_data(df):
    """Perform basic analysis on the cleaned data"""
    print("\nPerforming basic analysis...")
    
    # Summary statistics for numeric columns
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Correlation analysis for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        print("\nCorrelation Matrix:")
        correlation_matrix = numeric_df.corr()
        print(correlation_matrix)
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
    
    return df

def main():
    file_path = "MY Stock Market.csv"
    
    # Load and inspect data
    df = load_and_inspect_data(file_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Analyze data
    df_analyzed = analyze_data(df_clean)
    
    # Save cleaned data
    df_clean.to_csv('cleaned_stock_market.csv', index=False)
    print("\nCleaned data saved to 'cleaned_stock_market.csv'")

if __name__ == "__main__":
    main() 