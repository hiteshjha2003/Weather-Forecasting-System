import pyodbc
import pandas as pd

def connect_to_db(server, database, user, password):
    conn_str = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={user};PWD={password}"
    conn = pyodbc.connect(conn_str)
    return conn

def store_data(conn, df):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO WeatherData (Temperature, ApparentTemperature, Humidity, WindSpeed, 
                                    WindBearing, Visibility, Pressure, IsRainy, Hour, Day, Month)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(row))
    conn.commit()

def retrieve_data(conn):
    query = "SELECT * FROM WeatherData"
    return pd.read_sql(query, conn)

if __name__ == "__main__":
    server = "your_server"
    database = "your_database"
    user = "your_user"
    password = "your_password"
    conn = connect_to_db(server, database, user, password)