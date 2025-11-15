import yfinance as yf
import pandas as pd
from .db_manager import get_db_engine, create_tables
from .extractors import get_sp500_tickers
from tqdm import tqdm

def run_full_etl():
    engine = get_db_engine()

    print("Creating tables...")
    create_tables()

    print("Fetching tickers...")
    tickers_data = get_sp500_tickers()
    symbols_df = pd.DataFrame(tickers_data)
    symbols_df = symbols_df[['ticker', 'company_name', 'sector']]

    print(f"Saving {len(symbols_df)} tickers to database...")
    try:
        symbols_df.to_sql('symbols', engine, if_exists='append', index=False, method='multi', chunksize=100)
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e):
            print("Tickers already exist in the database. Skipping symbol load.")
        else:
            print(f"An error occurred while saving symbols: {e}")

    print("Downloading historical data (may take a while)...")
    ticker_list = [t.replace('.', '-') for t in symbols_df['ticker'].tolist()]


    batch_size = 50
    all_batches = []

    for i in tqdm(range(0, len(ticker_list), batch_size), desc="Downloading batches"):
        batch = ticker_list[i:i + batch_size]
        for attempt in range(3):
            try:
                batch_data = yf.download(batch, period='max', interval='1d', auto_adjust=True)
                all_batches.append(batch_data)
                break
            except Exception as e:
                print(f"Error downloading batch {i // batch_size + 1}, attempt {attempt + 1}: {e}")
                if attempt == 2:
                    print("Skipping this batch after 3 failed attempts.")

    print("Transforming data...")
    all_data = pd.concat(all_batches, axis=1)

    all_data_stacked = all_data.stack(level=1, future_stack=True).reset_index()
    print(all_data_stacked.columns)
    all_data_stacked = all_data_stacked.rename(columns={
        all_data_stacked.columns[0]: 'time',
        all_data_stacked.columns[1]: 'ticker'
    })

    if 'Adj Close' in all_data_stacked.columns and 'Close' not in all_data_stacked.columns:
        all_data_stacked['Close'] = all_data_stacked['Adj Close']

    all_data_stacked = all_data_stacked[['time', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

    symbols_map = pd.read_sql("SELECT id, ticker FROM symbols", engine)
    final_data = pd.merge(all_data_stacked, symbols_map, on='ticker', how='inner')

    final_data = final_data.rename(columns={
        'id': 'symbol_id',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    final_data = final_data[['time', 'symbol_id', 'open', 'high', 'low', 'close', 'volume']]

    print("Converting timestamps to UTC...")
    final_data['time'] = pd.to_datetime(final_data['time'], utc=True)

    final_data = final_data.dropna(subset=['open', 'close', 'high', 'low'])

    print(f"Loading {len(final_data)} rows of data to database...")
    try:
        chunksize = 10000
        for start in tqdm(range(0, len(final_data), chunksize), desc="Uploading to DB"):
            chunk = final_data.iloc[start:start + chunksize]
            chunk.to_sql('stock_data_daily', engine, if_exists='append', index=False, method='multi')
        print("ETL successful!")
    except Exception as e:
        print("Could not finish ETL due to error:", e)


if __name__ == '__main__':
    run_full_etl()