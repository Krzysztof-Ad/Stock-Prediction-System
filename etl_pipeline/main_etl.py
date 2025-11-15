import pandas as pd
from .db_manager import get_db_engine, create_tables, create_macro_table
from .extractors import get_sp500_tickers, fetch_macro_data, fetch_stock_data, get_macro_tickers
from tqdm import tqdm

def run_full_etl():
    engine = get_db_engine()

    print("Creating tables...")
    create_tables()
    create_macro_table()

    print("Fetching tickers...")
    tickers_data = get_sp500_tickers()
    symbols_df = pd.DataFrame(tickers_data)
    symbols_df = symbols_df[['ticker', 'company_name', 'sector', 'industry']]

    macro_ticker_data = get_macro_tickers()
    macro_ticker_list = list(macro_ticker_data.keys())


    print(f"Saving {len(symbols_df)} tickers to database...")
    try:
        symbols_df.to_sql('symbols', engine, if_exists='append', index=False, method='multi', chunksize=100)
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e):
            print("Tickers already exist in the database. Skipping symbol load.")
        else:
            print(f"An error occurred while saving symbols: {e}")

    ticker_list = [t.replace('.', '-') for t in symbols_df['ticker'].tolist()]

    all_data = fetch_stock_data(ticker_list)

    all_macro_data = fetch_macro_data(macro_ticker_list)
    if not all_macro_data.empty:
        print("Transforming macro data...")
        all_macro_data_stacked = all_macro_data.stack(future_stack=True).reset_index()
        all_macro_data_stacked.columns = ['time', 'ticker', 'close']
        all_macro_data_stacked = all_macro_data_stacked.dropna()
        all_macro_data_stacked['time'] = pd.to_datetime(all_macro_data_stacked['time'], utc=True)

        print(f"Saving {len(all_macro_data_stacked)} macro data to database...")
        try:
            all_macro_data_stacked.to_sql(
                'macro_data_daily',
                engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=5000
            )
            print(f"Saving {len(all_macro_data_stacked)} macro data to database was successful.")
        except Exception as e:
            print(f"Could not finish saving macro data due to error: {e}")
    else:
        print("No macro data available.")

    if not all_data.empty:
        print("Transforming stock data...")

        all_data_stacked = all_data.stack(level=1, future_stack=True).reset_index()
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

        print(f"Saving {len(final_data)} rows of data to database...")
        try:
            chunksize = 10000
            for start in tqdm(range(0, len(final_data), chunksize), desc="Uploading to DB"):
                chunk = final_data.iloc[start:start + chunksize]
                chunk.to_sql('stock_data_daily', engine, if_exists='append', index=False, method='multi')
            print(f"Saving {len(final_data)} rows of data was successful!")
        except Exception as e:
            print("Could not finish saving data due to error:", e)
    else:
        print("No stock data available.")

    print("ETL process complete.")

if __name__ == '__main__':
    run_full_etl()