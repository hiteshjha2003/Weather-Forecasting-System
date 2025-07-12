
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

def preprocess_data(df):
    # Handle missing values
    df.fillna({'Temperature (C)': df['Temperature (C)'].mean(),
               'Humidity': df['Humidity'].mean(),
               'Precip Type': df['Precip Type'].mode()[0]}, inplace=True)
    
    # Convert date to datetime and extract features
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df['Hour'] = df['Formatted Date'].dt.hour
    df['Day'] = df['Formatted Date'].dt.day
    df['Month'] = df['Formatted Date'].dt.month
    
    # Feature engineering
    df['Is_Rainy'] = df['Precip Type'].apply(lambda x: 1 if x == 'rain' else 0)
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
    df = load_data(file_path)
    df_processed, scaler = preprocess_data(df)


