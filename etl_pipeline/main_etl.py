"""
Main ETL (Extract, Transform, Load) pipeline for the Stock Prediction System.

This module orchestrates the complete data pipeline:
1. Extract: Fetches data from external sources (Wikipedia, Yahoo Finance, RSS feeds)
2. Transform: Cleans and structures data for database insertion
3. Load: Saves data to PostgreSQL database with incremental updates

The pipeline is designed to be idempotent - it can be run multiple times
and will only fetch new data since the last run.
"""

import pandas as pd
from .db_manager import get_db_engine, create_tables, create_macro_table, create_news_table
from .extractors import get_sp500_tickers, fetch_macro_data, fetch_stock_data, get_macro_tickers, fetch_market_news_rss
from tqdm import tqdm
from sqlalchemy import text
from datetime import time

def get_next_start_date(table_name, engine):
    """
    Determines the start date for incremental data downloads.
    
    Checks the database for the latest date and calculates the next date to fetch.
    Implements smart logic to avoid unnecessary downloads:
    - If data is up to date (has today's data, or yesterday's data and market is still open), returns 'UP_TO_DATE'
    - If data exists, returns the next date after the latest date
    - If no data exists, returns None to trigger a full historical download
    
    Args:
        table_name (str): Name of the database table to check (e.g., 'stock_data_daily')
        engine: SQLAlchemy database engine
    
    Returns:
        str or None: Start date in 'YYYY-MM-DD' format, 'UP_TO_DATE', or None for full download
    """
    try:
        with engine.connect() as connection:
            # Find the most recent date in the table
            result = connection.execute(text(f"SELECT MAX(time) FROM {table_name};")).scalar()
        if result:
            last_date_in_db = result.date()
            today_utc = pd.Timestamp.utcnow().date()
            # Check if market is closed (after 22:00 UTC, which is typically after market close)
            market_closed = pd.Timestamp.utcnow().time() > time(22, 0)

            # If we have today's data, or yesterday's data and market is still open, we're up to date
            if last_date_in_db == today_utc or (
                    last_date_in_db == (today_utc - pd.Timedelta(days=1)) and not market_closed):
                print(f"Data in '{table_name}' are up to date (from date {last_date_in_db}).")
                return 'UP_TO_DATE'

            # Calculate next date to fetch (day after the latest date in DB)
            start_date = (last_date_in_db + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Found latest data in {table_name}. Starting download at {start_date}.")
            return start_date
        else:
            # No data in table, trigger full historical download
            print(f"No data found in {table_name}. Starting full download (period='max').")
            return None
    except Exception as e:
        print(f"ERROR checking MAX(time) for {table_name}: {e}. Starting full download (period='max').")

def run_full_etl():
    """
    Executes the complete ETL pipeline for stock prediction data.
    
    This is the main orchestration function that:
    1. Sets up database tables
    2. Fetches S&P 500 company list and saves to database
    3. Downloads stock price data (with incremental updates)
    4. Downloads macroeconomic indicator data (with incremental updates)
    5. Fetches market news articles
    6. Transforms and loads all data into the database
    
    The function is designed to be run periodically (e.g., daily) and will
    only fetch new data since the last run, making it efficient for updates.
    """
    engine = get_db_engine()

    # Step 1: Create database tables if they don't exist
    print("Creating tables...")
    create_tables()  # Creates symbols and stock_data_daily tables
    create_macro_table()  # Creates macro_data_daily table
    create_news_table()  # Creates market_news table

    # Step 2: Fetch and save S&P 500 company information
    print("Fetching tickers...")
    tickers_data = get_sp500_tickers()
    symbols_df = pd.DataFrame(tickers_data)
    symbols_df = symbols_df[['ticker', 'company_name', 'sector', 'industry']]

    # Get list of macroeconomic indicator tickers
    macro_ticker_data = get_macro_tickers()
    macro_ticker_list = list(macro_ticker_data.keys())

    # Fetch news articles from RSS feeds
    news_df = fetch_market_news_rss()


    # Step 3: Save symbols to database (skip if already exists)
    print(f"Saving {len(symbols_df)} tickers to database...")
    try:
        symbols_df.to_sql('symbols', engine, if_exists='append', index=False, method='multi', chunksize=100)
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e):
            print("Tickers already exist in the database. Skipping symbol load.")
        else:
            print(f"An error occurred while saving symbols: {e}")

    # Prepare ticker list for yfinance (replace dots with hyphens for compatibility)
    ticker_list = [t.replace('.', '-') for t in symbols_df['ticker'].tolist()]

    # Step 4: Fetch stock price data (incremental or full download)
    stock_start_date = get_next_start_date('stock_data_daily', engine)
    if stock_start_date == 'UP_TO_DATE':
        print("Stock data is up to date. Skipping.")
        all_data = pd.DataFrame()
    elif stock_start_date is None:
        # No data exists, download full history
        all_data = fetch_stock_data(ticker_list, start_date=None)
    else:
        # Incremental update: download from the next date after latest in DB
        all_data = fetch_stock_data(ticker_list, stock_start_date)

    # Step 5: Fetch macroeconomic data (incremental or full download)
    macro_start_date = get_next_start_date('macro_data_daily', engine)
    if macro_start_date == 'UP_TO_DATE':
        print("Macro data is up to date. Skipping.")
        all_macro_data = pd.DataFrame()
    elif macro_start_date is None:
        # No data exists, download full history
        all_macro_data = fetch_macro_data(macro_ticker_list, start_date=None)
    else:
        # Incremental update: download from the next date after latest in DB
        all_macro_data = fetch_macro_data(macro_ticker_list, macro_start_date)

    # Step 6: Transform and load stock data
    if not all_data.empty:
        print("Transforming stock data...")

        # Convert multi-index DataFrame (Date, Ticker) to long format
        all_data_stacked = all_data.stack(level=1, future_stack=True).reset_index()
        all_data_stacked = all_data_stacked.rename(columns={
            all_data_stacked.columns[0]: 'time',
            all_data_stacked.columns[1]: 'ticker'
        })

        # Handle adjusted close if regular close is not available
        if 'Adj Close' in all_data_stacked.columns and 'Close' not in all_data_stacked.columns:
            all_data_stacked['Close'] = all_data_stacked['Adj Close']

        # Select only the columns we need
        all_data_stacked = all_data_stacked[['time', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Map ticker symbols to symbol IDs from the database
        symbols_map = pd.read_sql("SELECT id, ticker FROM symbols", engine)
        final_data = pd.merge(all_data_stacked, symbols_map, on='ticker', how='inner')

        # Rename columns to match database schema (lowercase, snake_case)
        final_data = final_data.rename(columns={
            'id': 'symbol_id',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        final_data = final_data[['time', 'symbol_id', 'open', 'high', 'low', 'close', 'volume']]

        # Ensure timestamps are in UTC timezone
        print("Converting timestamps to UTC...")
        final_data['time'] = pd.to_datetime(final_data['time'], utc=True)

        # Remove rows with missing essential price data
        final_data = final_data.dropna(subset=['open', 'close', 'high', 'low'])

        # Drop exact duplicates (same symbol and timestamp) that may come from retries
        final_data = final_data.drop_duplicates(subset=['time', 'symbol_id'])

        # Pull the latest timestamp per symbol already stored in the DB so we only insert new rows
        last_dates = pd.read_sql(
            "SELECT symbol_id, MAX(time) as last_time FROM stock_data_daily GROUP BY symbol_id",
            engine
        )

        last_dates_dict = dict(zip(last_dates['symbol_id'], last_dates['last_time']))

        # Keep only the rows that are strictly newer than what we already have for each symbol
        final_data = final_data[final_data.apply(
            lambda row: row['time'] > last_dates_dict.get(row['symbol_id'], pd.Timestamp('1970-01-01')),
            axis=1
        )]


        # Save to database in chunks to avoid memory issues
        print(f"Saving {len(final_data)} rows of data to database...")
        try:
            chunksize = 10000  # Process 10k rows at a time
            for start in tqdm(range(0, len(final_data), chunksize), desc="Uploading to DB"):
                chunk = final_data.iloc[start:start + chunksize]

                # Commit each chunk independently so partial progress persists
                with engine.begin() as connection:
                    chunk.to_sql(
                        'stock_data_daily',
                        connection,
                        if_exists='append',
                        index=False,
                        method='multi'
                    )
            print(f"Saving {len(final_data)} rows of data was successful!")
        except Exception as e:
            print("Could not finish saving data due to error:", e)
    else:
        print("No stock data available.")

    # Step 7: Transform and load macroeconomic data
    if not all_macro_data.empty:
        print("Transforming macro data...")
        # Convert from wide format (tickers as columns) to long format
        all_macro_data_stacked = all_macro_data.stack(future_stack=True).reset_index()
        all_macro_data_stacked.columns = ['time', 'ticker', 'close']
        # Remove rows with missing values
        all_macro_data_stacked = all_macro_data_stacked.dropna()
        # Ensure timestamps are in UTC
        all_macro_data_stacked['time'] = pd.to_datetime(all_macro_data_stacked['time'], utc=True)

        # Drop duplicate ticker/timestamp combinations that can arise when re-fetching data
        all_macro_data_stacked = all_macro_data_stacked.drop_duplicates(subset=['time', 'ticker'])

        # Fetch the last stored timestamp per macro ticker to ensure we only append new rows
        last_macro = pd.read_sql(
            "SELECT ticker, MAX(time) as last_time FROM macro_data_daily GROUP BY ticker",
            engine
        )
        last_macro_dict = dict(zip(last_macro['ticker'], last_macro['last_time']))

        # Keep only rows newer than the last stored timestamp for each ticker
        all_macro_data_stacked = all_macro_data_stacked[
            all_macro_data_stacked.apply(
                lambda row: row['time'] > last_macro_dict.get(row['ticker'], pd.Timestamp('1970-01-01')),
                axis=1
            )
        ]

        print(f"Saving {len(all_macro_data_stacked)} macro data to database...")
        try:
            chunksize = 5000
            for start in tqdm(range(0, len(all_macro_data_stacked), chunksize), desc="Uploading macro data"):
                chunk = all_macro_data_stacked.iloc[start:start + chunksize]

                with engine.begin() as connection:
                    chunk.to_sql(
                        'macro_data_daily',
                        con=connection,
                        if_exists='append',
                        index=False,
                        method='multi'
                    )
            print(f"Saving {len(all_macro_data_stacked)} macro data to database was successful.")
        except Exception as e:
            print(f"Could not finish saving macro data due to error: {e}")
    else:
        print("No macro data available.")

    # Step 8: Load news data
    if not news_df.empty:
        print(f"Saving {len(news_df)} news data to database...")
        try:
            news_df.to_sql(
                'market_news',
                engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000  # Process 1k rows at a time
            )
            print(f"Successfully saved news data to database.")
        except Exception as e:
            # Handle duplicate news articles gracefully
            if "duplicate key value violates unique constraint" in str(e):
                print("News data already exists in the database. Skipping.")
            else:
                print(f"Could not save news data due to error: {e}")
    else:
        print("No news data available.")

    print("ETL process complete.")

if __name__ == '__main__':
    run_full_etl()