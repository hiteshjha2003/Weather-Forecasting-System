from sqlalchemy import create_engine
import pandas as pd

def connect_to_db(server, database):
    connection_string = (
        f"mssql+pyodbc://{server}/{database}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
    )
    engine = create_engine(connection_string)
    return engine.connect()

def store_data(conn, df):
    df.to_sql("WeatherData", con=conn, if_exists='append', index=False)

def retrieve_data(conn):
    query = "SELECT * FROM WeatherData"
    return pd.read_sql(query, conn)

# Example usage (if needed for testing)
if __name__ == "__main__":
    server = "your_server"
    database = "your_database"
    conn = connect_to_db(server, database)
    df = retrieve_data(conn)
    print(df.head())
