"""
Database management module for the Stock Prediction System.

This module handles database connection setup and table creation/migration for storing:
- Stock symbols (S&P 500 companies)
- Daily stock price data (OHLCV)
- Macroeconomic indicators (VIX, Treasury yields, commodities, etc.)
- Market news articles from RSS feeds
- Engineered feature data used by downstream models

Uses PostgreSQL with TimescaleDB extension for time-series data optimization.
"""

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database connection parameters loaded from environment variables
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

# Construct PostgreSQL connection URL
DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)

def get_db_engine():
    """
    Returns the database engine instance.
    
    Returns:
        sqlalchemy.engine.Engine: The database engine for executing queries
    """
    return engine

def create_tables():
    """
    Creates the main database tables for stock symbols and daily stock data.
    
    Creates two tables:
    1. symbols: Stores S&P 500 company information (ticker, name, sector, industry)
    2. stock_data_daily: Stores daily OHLCV (Open, High, Low, Close, Volume) price data
    
    Also converts stock_data_daily to a TimescaleDB hypertable for optimized
    time-series queries and automatic data partitioning.
    """
    with engine.begin() as connection:
        # Create symbols table to store company information
        # This acts as a reference table for stock tickers
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
        connection.execute(text("""
        CREATE TABLE IF NOT EXISTS symbols (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) UNIQUE NOT NULL,
            company_name VARCHAR(255),
            sector VARCHAR(100),
            industry VARCHAR(255)
        );
        """))

        # Create stock_data_daily table for storing daily price data
        # Uses composite primary key (time, symbol_id) to allow one record per symbol per day
        # Foreign key ensures data integrity by referencing the symbols table
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

        # Convert to TimescaleDB hypertable for time-series optimization
        # This enables automatic partitioning and improved query performance
        connection.execute(text(
            "SELECT create_hypertable('stock_data_daily', 'time', if_not_exists => TRUE);"
        ))

    print("Tables created successfully!")

def create_macro_table():
    """
    Creates the macro_data_daily table for storing macroeconomic indicators.
    
    Stores daily closing prices for indicators like:
    - VIX (Volatility Index)
    - Treasury yields
    - Commodities (Oil, Gold)
    - Currency exchange rates
    
    Also converts to a TimescaleDB hypertable for time-series optimization.
    """
    with engine.begin() as connection:
        # Create table for macroeconomic data
        # Uses ticker to identify different macro indicators (e.g., '^VIX', 'CL=F')
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS macro_data_daily (
                time TIMESTAMPTZ NOT NULL,
                ticker VARCHAR(20) NOT NULL,
                close DOUBLE PRECISION,
                PRIMARY KEY (time, ticker)
            );
        """))

        # Convert to TimescaleDB hypertable for time-series optimization
        connection.execute(text(
            "SELECT create_hypertable('macro_data_daily', 'time', if_not_exists => TRUE);"
        ))

    print("Table 'macro_data_daily' created successfully!")

def create_news_table():
    """
    Creates the market_news table for storing financial news articles.
    
    Stores news headlines from various RSS feeds with publication timestamps.
    Uses a unique constraint to prevent duplicate articles based on headline and time.
    """
    with engine.begin() as connection:
        # Create table for market news articles
        # Unique constraint prevents storing the same headline at the same time
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
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

def create_or_migrate_feature_table(template_df):
    """
    Creates or incrementally migrates the `stock_data_features` table.

    The schema is inferred from a template DataFrame:
    - Requires a `time` column and a `symbol_id` column
    - All other columns are treated as numeric feature columns and added as
      DOUBLE PRECISION columns if they do not already exist
    - Ensures the table is registered as a TimescaleDB hypertable

    This allows you to safely evolve your feature set over time (e.g., when you
    add new engineered features) without dropping or recreating the table.

    Args:
        template_df (pd.DataFrame): Example feature DataFrame whose columns
            define the desired schema of `stock_data_features`.
    """
    # All columns except the time key and symbol foreign key are treated as features
    feature_columns = [c for c in template_df.columns if c not in ["time", "symbol_id"]]

    with engine.begin() as conn:
        # Create base table if missing
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stock_data_features (
                time TIMESTAMPTZ NOT NULL,
                symbol_id INTEGER NOT NULL,
                PRIMARY KEY (time, symbol_id),
                CONSTRAINT fk_symbol FOREIGN KEY (symbol_id) REFERENCES symbols(id)
            );
        """))

        # Fetch existing columns from information_schema
        existing_cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name='stock_data_features';
        """)).fetchall()

        existing_cols = {row[0] for row in existing_cols}

        # Add missing columns
        for col in feature_columns:
            if col not in existing_cols:
                # Quote the column name to support feature names with special characters
                conn.execute(text(
                    f'ALTER TABLE stock_data_features ADD COLUMN "{col}" DOUBLE PRECISION;'
                ))
                print(f"[MIGRATION] Added missing column: {col}")

        # Convert to hypertable (no-op if already converted)
        conn.execute(text(
            "SELECT create_hypertable('stock_data_features', 'time', if_not_exists => TRUE);"
        ))
    print("Feature table created / migrated successfully!")


def create_sentiment_table():
    """
    Creates the `sentiment_daily` table for storing aggregated news sentiment.

    Each row stores the average sentiment score for a ticker on a given day,
    sourced from a particular news provider, and is indexed for time-series queries.
    """
    with engine.begin() as connection:
        # Ensure TimescaleDB is available for hypertable support
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
        # Store daily sentiment aggregates per ticker/source pair
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS sentiment_daily
            (
                date                TIMESTAMPTZ NOT NULL,
                ticker              VARCHAR(20) NOT NULL,
                source_name          VARCHAR(100) NOT NULL,
                avg_sentiment_score DOUBLE PRECISION,
                article_count       INTEGER     NOT NULL,
                PRIMARY KEY (date, ticker)
            );
        """))
        connection.execute(text(
            "SELECT create_hypertable('sentiment_daily', 'date', if_not_exists => TRUE);"
        ))
    print("Table 'sentiment_daily' created successfully!")