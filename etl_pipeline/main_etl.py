import pandas as pd
from .db_manager import get_db_engine, create_tables, create_macro_table, create_news_table
from .extractors import get_sp500_tickers, fetch_macro_data, fetch_stock_data, get_macro_tickers, fetch_market_news_rss
from tqdm import tqdm
from sqlalchemy import text
from datetime import datetime, time

def get_next_start_date(table_name, engine):
    try:
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT MAX(time) FROM {table_name};")).scalar()
        if result:
            last_date_in_db = result.date()
            today_utc = pd.Timestamp.utcnow().date()
            market_closed = pd.Timestamp.utcnow().time() > time(22, 0)

            if last_date_in_db == today_utc or (
                    last_date_in_db == (today_utc - pd.Timedelta(days=1)) and not market_closed):
                print(f"Data in '{table_name}' are up to date (from date {last_date_in_db}).")
                return 'UP_TO_DATE'

            start_date = (result + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Found latest data in {table_name}. Starting download at {start_date}.")
            return start_date
        else:
            print(f"No data found in {table_name}. Starting full dowload (period='max').")
            return None
    except Exception as e:
        print(f"ERROR checking MAX(time) for {table_name}: {e}. Starting full download (period='max').")

def run_full_etl():
    engine = get_db_engine()

    print("Creating tables...")
    create_tables() #stock market data
    create_macro_table() #macro data
    create_news_table() # news data

    print("Fetching tickers...")
    tickers_data = get_sp500_tickers()
    symbols_df = pd.DataFrame(tickers_data)
    symbols_df = symbols_df[['ticker', 'company_name', 'sector', 'industry']]

    macro_ticker_data = get_macro_tickers()
    macro_ticker_list = list(macro_ticker_data.keys())

    news_df = fetch_market_news_rss()


    print(f"Saving {len(symbols_df)} tickers to database...")
    try:
        symbols_df.to_sql('symbols', engine, if_exists='append', index=False, method='multi', chunksize=100)
    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e):
            print("Tickers already exist in the database. Skipping symbol load.")
        else:
            print(f"An error occurred while saving symbols: {e}")

    ticker_list = [t.replace('.', '-') for t in symbols_df['ticker'].tolist()]

    stock_start_date = get_next_start_date('stock_data_daily', engine)
    if stock_start_date == 'UP_TO_DATE':
        print("Stock data is up to date. Skipping.")
        all_data = pd.DataFrame()
    elif stock_start_date is None:
        all_data = fetch_stock_data(ticker_list, start_date=None)
    else:
        all_data = fetch_stock_data(ticker_list, stock_start_date)

    macro_start_date = get_next_start_date('macro_data_daily', engine)
    if macro_start_date == 'UP_TO_DATE':
        print("Macro data is up to date. Skipping.")
        all_macro_data = pd.DataFrame()
    elif macro_start_date is None:
        all_macro_data = fetch_macro_data(ticker_list, start_date=None)
    else:
        all_macro_data = fetch_macro_data(macro_ticker_list, macro_start_date)

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

    if not news_df.empty:
        print(f"Saving {len(news_df)} news data to database...")
        try:
            news_df.to_sql(
                'market_news',
                engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            print(f"Successfully saved news data to database.")
        except Exception as e:
            if "duplicate key value violates unique constraint" in str(e):
                print("News data already exists in the database. Skipping.")
            else:
                print(f"Could not save news data due to error: {e}")
    else:
        print("No news data available.")

    print("ETL process complete.")

if __name__ == '__main__':
    run_full_etl()