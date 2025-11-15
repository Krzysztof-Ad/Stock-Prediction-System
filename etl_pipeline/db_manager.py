"""
Database management module for the Stock Prediction System.

This module handles database connection setup and table creation for storing:
- Stock symbols (S&P 500 companies)
- Daily stock price data (OHLCV)
- Macroeconomic indicators (VIX, Treasury yields, commodities, etc.)
- Market news articles from RSS feeds

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