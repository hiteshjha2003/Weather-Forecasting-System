"""
4. Exploratory Data Analysis (EDA)

Perform EDA to understand data patterns and relationships.
Visualize distributions

Use histograms for numerical features (e.g., temperature, humidity).


Analyze correlations
Create a correlation heatmap using seaborn.



Summarize insights
Identify key patterns (e.g., temperature trends by month).



Explanation: EDA helps uncover trends and informs feature selection for modeling.

"""

import matplotlib.pyplot as plt
import seaborn as sns
from data_loading import load_data
from data_preprocesssing import preprocess_data

def perform_eda(df, output_dir="eda_plots"):
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Histogram of temperature
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Temperature (C)'], bins=30)
    plt.title('Temperature Distribution')
    plt.savefig(f"{output_dir}/temperature_histogram.png")
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    
    # Summary statistics
    summary = df.describe()
    print("Summary Statistics:")
    print(summary)

if __name__ == "__main__":
    file_path = "weatherHistory.csv"
    df = load_data(file_path)
    df_processed, _ = preprocess_data(df)
    perform_eda(df_processed)

