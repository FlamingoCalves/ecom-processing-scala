import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('olist.sqlite')

# Get a list of all tables
tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(tables_query, conn)

for table_name in tables['name'].to_list():
    # Get the column names
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    df.to_csv(f"{table_name}.csv", index=False)

conn.close()