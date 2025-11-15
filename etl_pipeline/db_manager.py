from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)

def get_db_engine():
    return engine

def create_tables():
    with engine.begin() as connection:
        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS symbols (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) UNIQUE NOT NULL,
            company_name VARCHAR(255),
            sector VARCHAR(100),
            industry VARCHAR(255)
        );
        """))

        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS stock_data_daily (
        time TIMESTAMPTZ NOT NULL,
        symbol_id INTEGER NOT NULL,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume BIGINT,
        PRIMARY KEY (time, symbol_id),
        CONSTRAINT fk_symbol FOREIGN KEY (symbol_id) REFERENCES symbols(id)
        );
        """))

        connection.execute(text(
            "SELECT create_hypertable('stock_data_daily', 'time', if_not_exists => TRUE);"
        ))

    print("Tables created successfully!")

def create_macro_table():
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS macro_data_daily (
                time TIMESTAMPTZ NOT NULL,
                ticker VARCHAR(20) NOT NULL,
                close DOUBLE PRECISION,
                PRIMARY KEY (time, ticker)
            );
        """))

        connection.execute(text(
            "SELECT create_hypertable('macro_data_daily', 'time', if_not_exists => TRUE);"
        ))

    print("Table 'macro_data_daily' created successfully!")

def create_news_table():
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS market_news
            (
                id           SERIAL PRIMARY KEY,
                published_at TIMESTAMPTZ NOT NULL,
                source_name  VARCHAR(100),
                headline     TEXT        NOT NULL,
                CONSTRAINT unique_headline_time UNIQUE (published_at, headline)
            );
        """))
    print("Table 'market_news' created successfully!")