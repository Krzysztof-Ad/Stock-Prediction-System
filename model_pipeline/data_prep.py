"""
Data Preparation Module for Stock Prediction System

This module handles loading and combining different types of data (macroeconomic indicators,
sentiment scores, and technical features) to create a complete training dataset for
machine learning models. It also generates the target variable (whether stock price goes up).
"""

import pandas as pd
import numpy as np
from etl_pipeline.db_manager import get_db_engine


def load_macro_features(engine):
    print("Loading macro data...")
    # Get all macro data: time, ticker (which macro indicator), and its close price
    df = pd.read_sql("SELECT time, ticker, close FROM macro_data_daily", engine)

    # Convert from long format (many rows) to wide format (one column per macro indicator)
    # Example: Instead of rows for "OIL", "GOLD", etc., we get columns for each
    df_pivot = df.pivot(index="time", columns="ticker", values="close")
    
    # Fill missing values by carrying forward the last known value
    # (if we don't have data for a day, use yesterday's value)
    df_pivot = df_pivot.ffill()
    
    # Make sure the time column is properly formatted as datetime
    df_pivot.index = pd.to_datetime(df_pivot.index, utc=True)
    
    # Add "Macro_" prefix to column names so we know these are macro features
    # Example: "OIL" becomes "Macro_OIL"
    df_pivot.columns = [f"Macro_{col}" for col in df_pivot.columns]
    return df_pivot


def load_sentiment_features(engine):
    print("Loading sentiment data...")
    # Get sentiment data for S&P 500 (market-wide sentiment, not individual stocks)
    query = "SELECT date, avg_sentiment_score, article_count FROM sentiment_daily WHERE ticker = 'SP500'"
    df = pd.read_sql(query, engine)
    
    # Rename columns to match our naming convention
    df = df.rename(columns={
        'date': 'time',  # Use 'time' consistently across all data
        'avg_sentiment_score': 'Sentiment_Score',  # How positive/negative the news is
        'article_count': 'Sentiment_Count'  # How many articles were analyzed
    })
    
    # Make sure time is in datetime format
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    # Set time as the index so we can merge with other data easily
    return df.set_index('time')


def prepare_training_dataset(limit=None, target_threshold=0.005):
    engine = get_db_engine()

    # Load macro and sentiment data (these are market-wide, same for all stocks)
    macro_df = load_macro_features(engine)
    sentiment_df = load_sentiment_features(engine)

    print("Loading technical indicators (this may take a while)...")
    # Build SQL query with optional limit for testing
    limit_sql = f"LIMIT {limit}" if limit else ""

    # Get technical features (RSI, moving averages, etc.) and the actual stock price
    # We join with stock_data_daily to get the closing price we'll use to create targets
    query = f"""
        SELECT f.*, d.close as price_close
        FROM stock_data_features f
        JOIN stock_data_daily d ON f.time = d.time AND f.symbol_id = d.symbol_id
        ORDER BY f.time ASC
        {limit_sql}
    """

    df = pd.read_sql(query, engine)
    # Make sure time is in datetime format
    df['time'] = pd.to_datetime(df['time'], utc=True)

    print("Generating cycle features...")
    # Day of week encoding (sine/cosine for cyclical representation)
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 5))
    df['day_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 5))

    # Month of year encoding
    df['month'] = df['time'].dt.month
    df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))

    df = df.drop(columns=['day_of_week', 'month'])

    # Add macro indicators to our dataset
    # Left join means: keep all stock data, add macro data where dates match
    print("Merging with macro data...")
    df = df.merge(macro_df, on='time', how='left')

    # Add sentiment data the same way
    print("Merging with sentiment data...")
    df = df.merge(sentiment_df, on='time', how='left')

    # Fill missing sentiment values with 0 (no sentiment data = neutral)
    # This is safer than dropping rows, as sentiment data might be sparse
    df['Sentiment_Score'] = df['Sentiment_Score'].fillna(0)
    df['Sentiment_Count'] = df['Sentiment_Count'].fillna(0)

    # Create binary target: will price go up tomorrow?
    print(f"Generating target data (Threshold={target_threshold*100}%)...")
    df = df.sort_values(['symbol_id', 'time'])

    # For each stock, get tomorrow's closing price
    # shift(-1) means "get the next row's value" (tomorrow's price)
    df['Next_Close'] = df.groupby('symbol_id')['price_close'].shift(-1)

    df['Next_Return'] = (df['Next_Close'] - df['price_close']) / df['price_close']

    df['Target'] = (df['Next_Return'] > target_threshold).astype(int)

    # Remove rows where we don't have tomorrow's price (can't create target for last day)
    df = df.dropna(subset=['Next_Close'])

    # Clean up: remove helper columns we don't need for training
    cols_to_drop = ['Next_Close', 'price_close']
    df = df.drop(columns=cols_to_drop)

    print(f"Ready dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


if __name__ == "__main__":
    data = prepare_training_dataset(limit=10000)
    print(data.head())
    print(data.columns.tolist())