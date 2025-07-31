import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Date

# 1. Load CSV into a DataFrame
df = pd.read_csv("../backtest_log.csv")
df = df.reset_index()  # If 'date' is the index, this will make it a column
df['date'] = pd.to_datetime(df['date']).dt.date
df = df.drop(columns=['index'])
# 2. Create DB engine (change details accordingly)
engine = create_engine("postgresql+psycopg2://phantronbeo:Truong15397298@gulugulu-db.c9i0iiackcds.ap-southeast-2.rds.amazonaws.com/postgres")

# 3. Upload to 'stock_trades' table (append mode)
print(df.head())

# Upload to PostgreSQL
df.to_sql(
    'stock_trades',
    con=engine,
    if_exists='append',
    index=False,  # Don't upload DataFrame index
    dtype={'date': Date()}
)

