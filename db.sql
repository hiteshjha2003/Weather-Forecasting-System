-- Creating database
CREATE DATABASE WeatherDB;
GO

USE WeatherDB;
GO

-- Creating WeatherData table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name='WeatherData')
CREATE TABLE WeatherData (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    Temperature FLOAT,
    ApparentTemperature FLOAT,
    Humidity FLOAT,
    WindSpeed FLOAT,
    WindBearing FLOAT,
    Visibility FLOAT,
    Pressure FLOAT,
    IsRainy BIT,
    Hour INT,
    Day INT,
    Month INT
);

-- Creating UserInputs table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name='UserInputs')
CREATE TABLE UserInputs (
    id INT PRIMARY KEY IDENTITY(1,1),
    date DATETIME,
    query_city VARCHAR(50),
    query_date DATETIME
);

-- Creating Predictions table
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name='Predictions')
CREATE TABLE Predictions (
    id INT PRIMARY KEY IDENTITY(1,1),
    user_input_id INT,
    predicted_temperature FLOAT,
    predicted_rainfall FLOAT,
    predicted_humidity FLOAT,
    prediction_time DATETIME,
    FOREIGN KEY (user_input_id) REFERENCES UserInputs(id)
);