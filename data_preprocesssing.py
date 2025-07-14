
"""

2. Preprocessing

Prepare the dataset for analysis and modeling by cleaning and transforming the data.
Handle missing values
Check for missing values and impute (e.g., mean for numerical, mode for categorical).
Convert date columns
Convert Formatted Date to datetime and extract features like hour, day, month.
Feature engineering
Create new features (e.g., is_rainy based on Precip Type).
Drop redundant columns (e.g., Daily Summary, Loud Cover).

Normalize/scale features
Scale numerical features (e.g., temperature, humidity) using StandardScaler
Explanation: Preprocessing ensures data quality and compatibility with machine learning models.

"""
from sklearn.preprocessing import StandardScaler
from data_loading import load_data
import pandas as pd
import datetime as dt

# def shift_date_range(df):
#     # Convert 'Formatted Date' to datetime with timezone
#     df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    
#     # Calculate offset to shift from 2006 to 2015
#     original_start = pd.Timestamp('2006-01-01', tz='UTC')
#     new_start = pd.Timestamp('2015-01-01', tz='UTC')
#     offset = new_start - original_start

#     # Shift the dates
#     df['Formatted Date'] = df['Formatted Date'] + offset
#     return df

def preprocess_data(df):
    # Ensure 'Formatted Date' is in datetime format
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)



    # Handle missing values
    df.fillna({'Temperature (C)': df['Temperature (C)'].mean(),
               'Humidity': df['Humidity'].mean(),
               'Precip Type': df['Precip Type'].mode()[0]}, inplace=True)
    
    # Extract features from the shifted date
    df['Hour'] = df['Formatted Date'].dt.hour
    df['Day'] = df['Formatted Date'].dt.day
    df['Month'] = df['Formatted Date'].dt.month
    
    # Feature engineering
    df['Is_Rainy'] = df['Precip Type'].apply(lambda x: 1 if x == 'rain' else 0)
    
    # Drop unnecessary columns
    # df = df.drop(['Formatted Date', 'Daily Summary', 'Loud Cover', 'Summary'], axis=1)
    df = df.drop(['Formatted Date', 'Daily Summary', 'Loud Cover', 'Summary'], axis=1)

    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 
                      'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 
                      'Pressure (millibars)']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler


if __name__ == "__main__":
    file_path = "weatherHistory.csv"
    
    # Load and shift dates
    df = load_data(file_path)
    # df = shift_date_range(df)
    
    # Save the DataFrame with shifted dates back to the same file
    df.to_csv(file_path, index=False)

    # Proceed with preprocessing
    df_processed, scaler = preprocess_data(df)
 