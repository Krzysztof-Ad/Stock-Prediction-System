"""
Feature generation pipeline for engineered signals.

This module:
- Pulls raw OHLCV data from `stock_data_daily`
- Computes a wide range of technical indicators (via `ta` library)
- Adds custom lagged return features
- Writes all engineered features into `stock_data_features`, expanding the schema
  on-the-fly as new features are introduced

The pipeline supports incremental updates by only generating features for dates
that are newer than the last processed timestamp per symbol.
"""

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm
import warnings
from ta import add_all_ta_features
from ta.utils import dropna
from etl_pipeline.db_manager import create_or_migrate_feature_table, get_db_engine

# Silence noisy warnings from TA library and pandas when processing large frames
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TEST_MODE = False
TEST_TICKER_COUNT = 6

def get_last_feature_dates_map(engine):
    """
    Returns a mapping of symbol_id -> latest feature timestamp already stored.

    This allows the generator to run incrementally, only computing features for
    periods that have not yet been processed.
    """
    sql = text("SELECT symbol_id, MAX(time) AS last_time FROM stock_data_features GROUP BY symbol_id;")
    with engine.begin() as conn:
        rows = conn.execute(sql).fetchall()
    return {int(row[0]): row[1] for row in rows}

def get_feature_template(engine, symbol_id_for_template):
    """
    Builds a reference feature DataFrame that defines the schema for the feature table.

    A single symbol (first symbol in DB) is used to:
    - Fetch historical OHLCV data
    - Compute all TA indicators and lag features
    - Determine the set of feature columns (used to migrate/create the table)
    """
    print(f"Generating feature template using symbol_id: {symbol_id_for_template}...")
    sql_template = text("""
                        SELECT time, open, high, low, close, volume
                            FROM stock_data_daily
                            WHERE symbol_id = :id
                            --ORDER BY time ASC
                        """)
    df_template = pd.read_sql(sql_template, engine, params={"id": int(symbol_id_for_template)}, index_col="time")
    if df_template.empty or len(df_template) < 50:
        raise ValueError(f"Not enough data for template symbol {symbol_id_for_template} to generate features.")
    df_template = dropna(df_template)
    add_all_ta_features(
        df_template, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )
    df_template['Daily_Return'] = df_template['close'].pct_change()
    lag_features_list = []
    for lag in range(1, 11):
        lag_name = f"Lag_Return_{lag}"
        lag_series = df_template['Daily_Return'].shift(lag).rename(lag_name)
        lag_features_list.append(lag_series)
    df_template = pd.concat([df_template] + lag_features_list, axis=1)
    df_template = df_template.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Daily_Return'])
    df_template = df_template.dropna()

    if df_template.empty:
        raise ValueError(f"Template generation resulted in empty DataFrame for symbol {symbol_id_for_template}.")

    df_template = df_template.reset_index()
    df_template['symbol_id'] = symbol_id_for_template

    print(f"Template generated successfully ({len(df_template.columns) - 2} features).")
    return df_template


def generate_features_for_df(df, symbol_id):
    """
    Generates all TA indicators + lagged returns for a single symbol.

    Args:
        df (pd.DataFrame): OHLCV data indexed by time
        symbol_id (int): Symbol identifier (needed for DB writes)
    """
    df = df.copy()
    add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )
    returns = df["close"].pct_change()
    for lag in range(1, 11):
        df[f"Lag_Return_{lag}"] = returns.shift(lag)
    df = df.dropna()
    df["symbol_id"] = symbol_id
    df = df.reset_index()
    return df

def run_feature_generator():
    """
    Main orchestration entrypoint that:
    1. Loads tickers available in the DB
    2. Generates a feature template and ensures the destination table matches it
    3. Iterates over each ticker to compute missing features (incremental)
    4. Persists all newly generated features in batches
    """
    engine = get_db_engine()
    print("Fetching tickers...")
    try:
        symbols_df = pd.read_sql("SELECT id, ticker FROM symbols", engine)
    except Exception as e:
        print(f"Critical error: Could not fetch tickers from database: {e}")
        return
    if symbols_df.empty:
        print("No ticker symbols found. Run etl_pipeline first.")
        return

    try:
        template_symbol_id = symbols_df.iloc[0]['id']
        template_df = get_feature_template(engine, template_symbol_id)
        create_or_migrate_feature_table(template_df)
    except Exception as e:
        print(f"Critical error during feature table creation: {e}")
        return

    # ======================TEST MODE================================
    if TEST_MODE:
        print("=" * 50)
        print(f"--- RUNNING IN TEST MODE ---")
        print(f"Processing only {TEST_TICKER_COUNT} tickers.")
        print("=" * 50)
        symbols_df = symbols_df.head(TEST_TICKER_COUNT)
    # ===============================================================

    print(f"Starting feature generation for {len(symbols_df)} tickers...")
    all_features_dfs = []

    last_times_map = get_last_feature_dates_map(engine)

    for index, row in tqdm(symbols_df.iterrows(), total=symbols_df.shape[0], desc="Generating features"):
        symbol_id = row['id']
        ticker = row['ticker']

        try:
            if symbol_id == template_symbol_id:
                if last_times_map.get(symbol_id) is None:
                    all_features_dfs.append(template_df)
                continue
            last_time = last_times_map.get(symbol_id)
            print(f"Last feature timestamp for {ticker}: {last_time}")

            if last_time is not None:
                # Only grab rows newer than the last generated feature timestamp for incremental updates
                sql_loop = text("""
                                SELECT time, open, high, low, close, volume
                                FROM stock_data_daily
                                WHERE symbol_id = :id
                                  AND time > :last_time
                                --ORDER BY time ASC
                                """)
                params = {'id': int(symbol_id), 'last_time': last_time}
            else:
                # First run for this ticker: pull the full history
                sql_loop = text("""
                                SELECT time, open, high, low, close, volume
                                FROM stock_data_daily
                                WHERE symbol_id = :id
                                --ORDER BY time ASC
                                """)
                params = {'id': int(symbol_id)}

            df = pd.read_sql(sql_loop, engine, params=params, index_col='time')
            if df.empty or len(df) < 50:
                continue
            features_df = generate_features_for_df(df, symbol_id)

            if not features_df.empty:
                all_features_dfs.append(features_df[template_df.columns])
        except Exception as e:
            print(f"\nERROR during feature generation for {ticker} (ID: {symbol_id}): {e}")
            continue

    if not all_features_dfs:
        print("Did not generate any features.")
        return

    print("\nMerging all features together...")
    full_features_df = pd.concat(all_features_dfs)

    print(f"Saving {len(full_features_df)} rows of features (this might take a while)...")
    try:
        chunksize = 10000
        total_rows = len(full_features_df)

        for start in tqdm(range(0, total_rows, chunksize), total=(total_rows // chunksize) + 1,
                          desc="Saving features"):
            chunk = full_features_df.iloc[start:start + chunksize]

            # Open a new transaction per chunk so partial progress is durable
            with engine.begin() as connection:
                chunk.to_sql(
                    'stock_data_features',
                    connection,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
        print("Feature engineering (Time-Series) completed successfully!")
    except Exception as e:
        print(f"ERROR saving to database: {e}")

if __name__ == "__main__":
    run_feature_generator()