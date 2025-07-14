# Weather Forecasting System

The Weather Forecasting System is a comprehensive application designed to predict weather parameters like temperature, humidity, and wind speed. It combines machine learning, data visualization, and database integration to provide an end-to-end solution for weather analysis and forecasting. Key features include:

Advanced Time Series Forecasting:
    Uses a custom Transformer model named AdaptiveDropPatch for multi-step weather forecasting.
    Predicts multiple weather variables (e.g., temperature, humidity, wind speed) over a user-defined time horizon.
Interactive EDA Dashboard:
    Built with Plotly, it provides dynamic visualizations such as histograms, correlation heatmaps, and trend lines.
    Visualizations are saved as PNG files for reference.
SQL Server Integration:
    Connects to a Microsoft SQL Server database to store and retrieve weather data.
    Scalable design for future extensions (e.g., storing predictions).
Modular Codebase:
    Organized with a clear separation of concerns, making it maintainable and extensible.



# Project Structure
Weather Forecasting System/
├── app.py                         # Main Streamlit application entrypoint
├── data_loading.py                # Data loading utilities
├── data_preprocessing.py          # Data preprocessing pipeline
├── db.sql                        # SQL Server schema & setup script
├── eda.py                        # EDA functions and plot generation
├── eda_plots/                    # Directory for saved EDA plot images
├── models/                       # Saved models & metadata files
│   ├── adaptive_drop_patch_model.pth
│   ├── adaptive_drop_patch_model.json
├── models_trainings/             # Core ML training, dataset, transformer implementation
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── forecast_cli.py
│   ├── plot_app.py
│   ├── test_import.py
│   ├── train_model.py
│   ├── transformer.py
├── requirements.txt              # Python dependencies
├── sql_connection.py             # SQL Server connection helpers
├── test.py                      # Misc test scripts
├── weatherHistory.csv            # Raw weather dataset CSV


The project is organized into a well-structured directory to separate functionalities like data handling, model training, visualization, and database operations. Below is a breakdown of the key files and directories:

app.py: The main entry point for the Streamlit application, orchestrating the user interface and interaction with other modules.
data_loading.py: Contains utilities for loading data, primarily from the weatherHistory.csv dataset.
data_preprocessing.py: Handles data cleaning and preprocessing, preparing raw data for analysis and modeling.
db.sql: SQL script to set up the database schema and tables in MS SQL Server.
eda.py: Implements functions for generating EDA visualizations using Plotly.
eda_plots/: Directory where generated EDA visualizations (e.g., PNG images) are saved.
models/:
Stores trained model files and metadata.
adaptive_drop_patch_model.pth: The trained Transformer model file.
adaptive_drop_patch_model.json: Metadata for the model (e.g., configuration details).
models_trainings/:
Contains core machine learning components, including:
__init__.py: Marks the directory as a Python package.
config.py: Configuration settings for model training (e.g., hyperparameters).
dataset.py: Defines the dataset class for handling weather data in training.
forecast_cli.py: Command-line interface for running forecasts.
plot_app.py: Functions for generating forecast visualizations.
test_import.py: Utility for testing imports (likely for debugging).
train_model.py: Script to train the Transformer model.
transformer.py: Implementation of the custom AdaptiveDropPatch Transformer model.
requirements.txt: Lists Python dependencies (e.g., Streamlit, Plotly, PyTorch, pyodbc).
sql_connection.py: Helper functions for connecting to and querying the SQL Server database.
test.py: Miscellaneous test scripts for development and debugging.
weatherHistory.csv: The raw dataset containing historical weather data (e.g., temperature, humidity, wind speed).



# Features
The application offers four main functionalities, each designed to address specific aspects of weather forecasting and analysis:

Transformer-Based Weather Forecasting:
    Utilizes the AdaptiveDropPatch Transformer model, a custom implementation tailored for time series forecasting.
    Supports multi-step predictions for variables like temperature, humidity, and wind speed.
    Allows loading and saving model checkpoints with associated metadata for reuse.

Interactive EDA Dashboard:
Automatically generates visualizations such as:
Histograms: To show the distribution of weather parameters.
Correlation Heatmaps: To identify relationships between variables.
Trend Lines: To visualize temporal patterns.
Visualizations are interactive (via Plotly) and embedded in the Streamlit UI.
Saves plots as PNG files in the eda_plots/ directory for offline reference.
SQL Server Database Integration:
    Connects to an MS SQL Server instance to retrieve and manage weather data stored in a WeatherData table.
    Designed for scalability, supporting potential future features like storing predictions or user inputs.
    Users can view database contents directly in the Streamlit app.
Modular Design:
    The codebase separates concerns (e.g., data loading, preprocessing, modeling, visualization, and database operations) to ensure maintainability and ease of extension.


# Setup Instructions
The README provides detailed steps to set up and run the application. 
Here’s a detailed explanation of each step:

# Prerequisites
Python 3.8+: Required to run the application and its dependencies.
MS SQL Server + SQL Server Management Studio (SSMS): Needed only if using database features. 
SSMS is used to manage the database.
Dataset: 
    The weatherHistory.csv file, included in the repository, contains historical weather data.

# Installation
Clone the Repository:
bash
`git clone https://github.com/your-username/Weather-Forecasting-System.git`
`cd Weather-Forecasting-System`

Clones the project repository and navigates to the project directory.
Install Python Dependencies:
bash
`pip install -r requirements.txt`

Installs required Python libraries listed in requirements.txt (e.g., Streamlit, Plotly, PyTorch, pyodbc).
Optional (Linux Users):
bash
`sudo apt install -y libgl1-mesa-glx`


Installs system libraries required for Plotly to export visualizations as images (e.g., PNG files).
Run the App:
bash
`streamlit run app.py`
Launches the Streamlit web application, accessible via a browser (typically at http://localhost:8501).


# Home:
    Displays an overview of the application and instructions for use.
Train Model:
    Allows users to configure training parameters (e.g., target variables, number of epochs, batch size, learning rate).
    Trains a new Transformer model on the weatherHistory.csv dataset.
    Saves the trained model and metadata to the models/ directory.
Forecasting:
    Loads a pre-trained Transformer model from the models/ directory.
    Users can select the forecast horizon (e.g., number of days) and target features (e.g., temperature, humidity).
    Displays forecasts as interactive Plotly charts with features like hover, zoom, and series toggling.
Database Access:
    Users enter SQL Server credentials to connect to the database.
    Displays contents of the WeatherData table within the Streamlit UI.
Exploratory Data Analysis (EDA):
    Generates interactive visualizations (e.g., histograms, heatmaps, trend lines) for the dataset.
    Users can manage cached plots (e.g., delete or regenerate) stored in the eda_plots/ directory.

EDA Visualizations:
    Example: A plot named temperature_by_month.png in the eda_plots/ directory, showing temperature trends by month.
    Visualizations are saved as PNG files for easy reference.
Forecasting Interactive Plot:
    Plotly charts display forecasts with interactive features (e.g., hover to view data points, zoom, toggle series).
    Embedded directly in the Streamlit UI for a seamless user experience.


Future Enhancements
    Ensemble Models: Integrate additional models like LSTM or XGBoost to improve forecast accuracy.
    Geo-Spatial Data: Add location-based predictions with maps.
    Docker Containerization: Package the app for easy deployment.
    User Authentication: Allow users to save preferences or access personalized features.
    Expanded Database Support: Add compatibility with PostgreSQL or SQLite.


# UML Diagrams
The README includes UML diagrams to illustrate the database and application structure. Below is a detailed explanation of each:
Database UML (Entity-Relationship Diagram)
Entity: WeatherData
Attributes:
ID (Primary Key, integer): Unique identifier for each record.
Temperature (float): Measured temperature.
ApparentTemperature (float): Perceived temperature.
Humidity (float): Relative humidity.
WindSpeed (float): Wind speed.
WindBearing (float): Wind direction.
Visibility (float): Visibility distance.
Pressure (float): Atmospheric pressure.
IsRainy (boolean): Indicates if it’s raining.
Hour, Day, Month (integers): Time components for temporal analysis.
Purpose: Represents the structure of the WeatherData table in the SQL Server database.


#Application UML (Class Diagram)
Classes:
DataLoader: Loads raw data from weatherHistory.csv using methods like load_data(file) and read_csv().
Preprocessor: Cleans and preprocesses data with methods like preprocess_data(df) and clean_missing_values().
EDAAnalyzer: Generates EDA visualizations using methods like perform_eda(df), generate_plots(), and get_data_info(df).
Predictor: Manages the Transformer model with attributes (model, meta) and methods like load_model(path) and forecast(data).
StreamlitApp: Coordinates the UI with attributes (current_page, session_state) and methods like run_app(), show_forecasting(), show_eda(), and show_database_access().
SQLConnector: Handles database operations with methods like connect(server, db, user, pw), retrieve_data(), and insert_data().
Relationships:
StreamlitApp uses all other classes to manage data flow and UI rendering.
Predictor depends on Preprocessor for prepared input data.
EDAAnalyzer uses preprocessed data for visualizations.
SQLConnector interacts with the SQL Server database.
DataLoader provides raw data to other components.


# References
Streamlit Documentation: For building the web application.
pyodbc Documentation: For SQL Server connectivity.
SQLAlchemy Documentation: For database interactions (possibly used alongside pyodbc).
Scikit-learn Documentation: For data preprocessing or baseline models.
Plotly Documentation: For creating interactive visualizations.
Transformer Models Research Papers and Tutorials: For the custom AdaptiveDropPatch model.


# UML Diagram Generation
The response to your request for UML diagram generation provides text-based representations of the database and application UML diagrams, which can be visualized using tools like Draw.io or PlantUML.

Database UML
Describes the WeatherData entity with its attributes (e.g., ID, Temperature, Humidity).
The primary key (ID) ensures unique records.
Can be visualized as a single entity with attributes listed.
Application UML (Class Diagram)
Lists classes (DataLoader, Preprocessor, EDAAnalyzer, Predictor, StreamlitApp, SQLConnector) with their methods and attributes.
Defines relationships (e.g., StreamlitApp uses all other classes, Predictor depends on Preprocessor).
Includes a PlantUML code snippet for generating the class diagram programmatically:
plantuml


@startuml
class DataLoader {
  +load_data(file)
  +read_csv()
}
class Preprocessor {
  +preprocess_data(df)
  +clean_missing_values()
}
class EDAAnalyzer {
  +perform_eda(df)
  +generate_plots()
  +get_data_info(df)
}
class Predictor {
  -model
  -meta
  +load_model(path)
  +forecast(data)
}
class StreamlitApp {
  -current_page
  -session_state
  +run_app()
  +show_forecasting()
  +show_eda()
  +show_database_access()
}
class SQLConnector {
  +connect(server, db, user, pw)
  +retrieve_data()
  +insert_data()
}

StreamlitApp --> DataLoader
StreamlitApp --> Preprocessor
StreamlitApp --> EDAAnalyzer
StreamlitApp --> Predictor
StreamlitApp --> SQLConnector
Predictor --> Preprocessor


Visualization Instructions:
Draw.io: Use the “Entity Relationship” template for the database UML and the “Class” template for the application UML. Draw classes/entities and connect them with arrows based on relationships.
PlantUML:  the provided PlantUML code into a PlantUML editor (e.g., PlantUML’s online editor or VS Code with the PlantUML extension) to generate a graphical class diagram.

Additional Notes
The AdaptiveDropPatch model is a custom Transformer implementation, likely designed to handle time series data with a focus on weather forecasting. The name suggests it incorporates techniques like adaptive dropout or patch-based processing, but further details would be in transformer.py.
The application is designed for scalability, with potential to add features like ensemble models or geospatial data.
The weatherHistory.csv dataset is critical, as it serves as the primary data source for training and EDA.
The SQL Server integration is optional, allowing users to run the app without a database if needed.


# Conclusion

    The Weather Forecasting System is a robust, modular, and extensible application that combines the power of:
    Transformer-based deep learning models for time series forecasting
    Interactive Streamlit dashboards for user-friendly exploration and prediction
    Advanced EDA with Plotly for dynamic visual insights
    SQL Server integration for enterprise-grade data storage and accessThis project showcases how modern machine learning and full-stack development can be brought together to solve real-world challenges like weather prediction. With features such as model training, forecasting, data analysis, and database access all in one interface, this system serves as an end-to-end solution for both data scientists and domain experts.

# What You Gained from This Project
    A real-world use case of transformer models in forecasting

    Hands-on application of data preprocessing and EDA techniques

    Seamless SQL database integration for storing and retrieving structured weather data

    A fully interactive dashboard built using Streamlit and Plotly

    Clean, scalable project structure for further development and deployment


