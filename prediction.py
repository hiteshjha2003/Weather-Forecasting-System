"""
5. Model Development

Develop and train a machine learning model to predict temperature, rainfall, and humidity.

Split data
Split data into training (80%) and testing (20%) sets.



Define and train model
Use RandomForestRegressor for temperature and humidity, RandomForestClassifier for rainfall (Is_Rainy).



Evaluate model
Use metrics like MSE, RMSE for regression, and accuracy, F1-score for classification.



Explanation: RandomForest is robust for weather data due to its ability to handle non-linear relationships.


"""


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import joblib

def train_models(df):
    X = df.drop(['Temperature (C)', 'Humidity', 'Is_Rainy'], axis=1)
    y_temp = df['Temperature (C)']
    y_hum = df['Humidity']
    y_rain = df['Is_Rainy']
    
    # Split data
    X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    _, _, y_hum_train, y_hum_test = train_test_split(X, y_hum, test_size=0.2, random_state=42)
    _, _, y_rain_train, y_rain_test = train_test_split(X, y_rain, test_size=0.2, random_state=42)
    
    # Train models
    temp_model = RandomForestRegressor(random_state=42)
    hum_model = RandomForestRegressor(random_state=42)
    rain_model = RandomForestClassifier(random_state=42)
    
    temp_model.fit(X_train, y_temp_train)
    hum_model.fit(X_train, y_hum_train)
    rain_model.fit(X_train, y_rain_train)
    
    # Evaluate models
    temp_pred = temp_model.predict(X_test)
    hum_pred = hum_model.predict(X_test)
    rain_pred = rain_model.predict(X_test)
    
    print("Temperature MSE:", mean_squared_error(y_temp_test, temp_pred))
    print("Humidity MSE:", mean_squared_error(y_hum_test, hum_pred))
    print("Rain Accuracy:", accuracy_score(y_rain_test, rain_pred))
    print("Rain F1-Score:", f1_score(y_rain_test, rain_pred))
    
    # Save models
    joblib.dump(temp_model, 'temp_model.pkl')
    joblib.dump(hum_model, 'hum_model.pkl')
    joblib.dump(rain_model, 'rain_model.pkl')
    
    return temp_model, hum_model, rain_model

def predict_weather(models, input_data):
    temp_model, hum_model, rain_model = models
    temp_pred = temp_model.predict(input_data)
    hum_pred = hum_model.predict(input_data)
    rain_pred = rain_model.predict(input_data)
    return temp_pred, hum_pred, rain_pred

