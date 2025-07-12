# Weather Forecasting System
1. Introduction
Weather forecasting is critical for sectors like agriculture, transportation, and disaster preparedness, enabling informed decision-making. By integrating machine learning with historical weather data, we can enhance prediction accuracy, while a database ensures efficient data management. This guide details the creation of a Streamlit application that uses a Kaggle dataset to forecast weather metrics (temperature, rainfall, humidity) and interacts with an MS SQL Server database. UML diagrams will document the database and application architecture, making the system scalable and maintainable. The steps are designed for users with basic programming skills, ensuring clarity and actionability.

This project implements a weather forecasting application using a Kaggle dataset, machine learning, MS SQL Server, and Streamlit.

2. SETUP 
## Setup
1. Create a Virtual Environment : 
    `python -m venv env`
    `env\Scripts\activate`

2. Install dependencies: 
    `pip install -r requirements.txt`

3. Configure SQL Server connection in 
    `sql_connection.py`.

4. Run the application: 
    `streamlit run app.py`

## Files
- `app.py`: Main Streamlit application.
- `dashboard.py`: Displays EDA visualizations.
- `data_loading_preprocessing.py`: Handles data loading and preprocessing.
- `eda.py`: Performs exploratory data analysis.
- `prediction.py`: Trains and predicts using machine learning models.
- `sql_connection.py`: Manages SQL Server connection.
- `db.sql`: SQL script for database setup.
- `requirements.txt`: Project dependencies.
- `weatherHistory.csv`: Sample dataset.

## Requirements
- Python 3.8+
- MS SQL Server
- Kaggle dataset (`weatherHistory.csv`)




3.UML Diagrams

Database UML:

Entity: WeatherData
Attributes: 
ID (PK), 
Temperature, 
ApparentTemperature, 
Humidity, 
WindSpeed, 
WindBearing, 
Visibility, 
Pressure, 
IsRainy, 
Hour, 
Day, 
Month



Application UML:
Classes: DataLoader, Preprocessor, SQLConnector, EDAAnalyzer, Predictor, StreamlitApp
Relationships: StreamlitApp uses all other classes; Predictor depends on Preprocessor.



4. Conclusion
This guide provides a comprehensive approach to building a weather forecasting system using Streamlit, a Kaggle dataset, and an MS SQL Server database. Key steps include downloading and preprocessing the dataset, setting up the Streamlit app, connecting to the database, designing the schema, training machine learning models, creating visualizations, and documenting with UML diagrams. The system can be extended with features like real-time data integration, user authentication, or advanced models like ARIMA for time series forecasting.

References:
Streamlit Documentation
pyodbc Documentation
SQLAlchemy Documentation
Scikit-learn Documentation
Plotly Documentation
draw.io
Kaggle Weather Dataset
