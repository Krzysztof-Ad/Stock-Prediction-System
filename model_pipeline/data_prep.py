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
    """
    Loads macroeconomic indicators (like oil prices, gold, etc.) from the database.
    Converts them into a format where each indicator is a column, indexed by date.
    """
    print("Loading macro data...")
    
    # Grab all macro data from the database
    df = pd.read_sql("SELECT time, ticker, close FROM macro_data_daily", engine)

    # Transform from long format (one row per indicator per day) to wide format (one column per indicator)
    # So instead of rows like "2020-01-01, OIL, 50" and "2020-01-01, GOLD, 1500",
    # we get columns like "OIL" and "GOLD" with values for each date
    df_pivot = df.pivot(index="time", columns="ticker", values="close")
    
    # If we're missing data for a day, just use the last known value (forward fill)
    # This prevents gaps in our data
    df_pivot = df_pivot.ffill()
    
    # Make sure dates are properly formatted
    df_pivot.index = pd.to_datetime(df_pivot.index, utc=True)
    
    # Add "Macro_" prefix to all column names so we can easily identify them later
    # Example: "OIL" becomes "Macro_OIL"
    df_pivot.columns = [f"Macro_{col}" for col in df_pivot.columns]
    return df_pivot


def load_sentiment_features(engine):
    """
    Loads market-wide sentiment scores (for the whole S&P 500, not individual stocks).
    These scores tell us how positive or negative the news was on each day.
    """
    print("Loading sentiment data...")
    
    # Get sentiment data for the overall market (S&P 500)
    query = "SELECT date, avg_sentiment_score, article_count FROM sentiment_daily WHERE ticker = 'SP500'"
    df = pd.read_sql(query, engine)
    
    # Rename columns to match our naming style
    df = df.rename(columns={
        'date': 'time',  # Keep 'time' consistent across all datasets
        'avg_sentiment_score': 'Sentiment_Score',  # Average sentiment (-1 to 1, roughly)
        'article_count': 'Sentiment_Count'  # How many news articles we analyzed
    })
    
    # Convert time to proper datetime format
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    # Use time as the index so we can easily merge with stock data later
    return df.set_index('time')


def prepare_training_dataset(limit=None, target_threshold=0.005):
    """
    Builds the complete training dataset by combining:
    - Technical indicators (RSI, moving averages, etc.) for each stock
    - Macroeconomic data (oil, gold, etc.)
    - Market sentiment scores
    - Creates the target variable: will the stock price go up tomorrow?
    
    Args:
        limit: Optional limit on number of rows (useful for testing)
        target_threshold: Minimum return to count as "up" (default 0.5%)
    """
    engine = get_db_engine()

    # Load market-wide data that applies to all stocks
    macro_df = load_macro_features(engine)
    sentiment_df = load_sentiment_features(engine)

    print("Loading technical indicators (this may take a while)...")
    
    # Optional limit for quick testing
    limit_sql = f"LIMIT {limit}" if limit else ""

    # Get all technical features plus the actual closing price
    # We need the closing price to calculate tomorrow's return (our target)
    query = f"""
        SELECT f.*, d.close as price_close
        FROM stock_data_features f
        JOIN stock_data_daily d ON f.time = d.time AND f.symbol_id = d.symbol_id
        ORDER BY f.time ASC
        {limit_sql}
    """

    df = pd.read_sql(query, engine)
    df['time'] = pd.to_datetime(df['time'], utc=True)

    print("Generating cycle features...")
    
    # Encode day of week as sine/cosine so the model understands it's cyclical
    # (Monday and Friday are close to each other in the cycle)
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 5))
    df['day_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 5))

    # Same thing for months (January and December are close)
    df['month'] = df['time'].dt.month
    df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
    df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))

    # Drop the raw columns, we only need the encoded versions
    df = df.drop(columns=['day_of_week', 'month'])

    # Add macro indicators and sentiment data to each stock's data
    # Left join keeps all stock rows and adds macro data where dates match, same with sentiment
    print("Merging with macro/sentiment data...")
    df = df.merge(macro_df, on='time', how='left')
    df = df.merge(sentiment_df, on='time', how='left')

    # If we don't have sentiment data for a day, assume neutral (0)
    # Better than dropping rows since sentiment data can be sparse
    df['Sentiment_Score'] = df['Sentiment_Score'].fillna(0)
    df['Sentiment_Count'] = df['Sentiment_Count'].fillna(0)

    # Create our target variable: will the stock price go up tomorrow?
    print(f"Generating target data (Threshold={target_threshold*100}%)...")
    
    # Important: Sort by stock first, then time
    # This groups each stock's data together so we can correctly get tomorrow's price
    # If we sorted by time first, rows would be interleaved (Stock A day 1, Stock B day 1, etc.)
    # and the shift operation would get confused
    df = df.sort_values(['symbol_id', 'time'])

    # For each stock, grab tomorrow's closing price
    # shift(-1) looks at the next row in the group (which is tomorrow for that stock)
    df['Next_Close'] = df.groupby('symbol_id')['price_close'].shift(-1)

    # Calculate the return: (tomorrow's price - today's price) / today's price
    df['Next_Return'] = (df['Next_Close'] - df['price_close']) / df['price_close']

    # Create binary target: 1 if return > threshold, 0 otherwise
    df['Target'] = (df['Next_Return'] > target_threshold).astype(int)

    # Drop rows where we can't calculate tomorrow's price (the last day for each stock)
    df = df.dropna(subset=['Next_Close'])

    # Now sort by time first for proper train/test splitting
    # This ensures when we split by date, no future data leaks into training
    df = df.sort_values(['time', 'symbol_id'])

    # Remove helper columns we used to create the target
    cols_to_drop = ['Next_Close', 'price_close', 'Next_Return']
    df = df.drop(columns=cols_to_drop)

    #StandardScaler will crash on NaN data, so we fill them
    df = df.fillna(0)

    print(f"Ready dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


if __name__ == "__main__":
    data = prepare_training_dataset(limit=None)
    print(data.head())
    print(data.columns.tolist())