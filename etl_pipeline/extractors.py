"""
Data extraction module for the Stock Prediction System.

This module handles fetching data from various external sources:
- S&P 500 company list from Wikipedia
- Stock price data from Yahoo Finance (via yfinance)
- Macroeconomic indicators (VIX, Treasury yields, commodities, currencies)
- Financial news articles from RSS feeds

All functions return data in formats suitable for database insertion.
"""

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm
from io import StringIO
import feedparser
from time import mktime


def clean_sp500_data(records):
    """
    Filters S&P 500 records to only include allowed columns.
    
    Removes any extra columns that might be present in the Wikipedia table
    to ensure data consistency.
    
    Args:
        records (list): List of dictionaries containing S&P 500 company data
        
    Returns:
        list: Filtered list of dictionaries with only allowed columns
    """
    allowed_columns = {'ticker', 'company_name', 'sector', 'industry'}
    cleaned = [{k: v for k, v in record.items() if k in allowed_columns} for record in records]
    return cleaned


def get_sp500_tickers():
    """
    Fetches the current list of S&P 500 companies from Wikipedia.
    
    Scrapes the Wikipedia page containing the S&P 500 list and extracts
    company information including ticker symbols, company names, sectors, and industries.
    
    Returns:
        list: List of dictionaries, each containing ticker, company_name, sector, and industry.
              Returns empty list on error.
    """
    print('Getting sp500 tickers from Wikipedia...')

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # User-Agent header to avoid being blocked by Wikipedia
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/117.0 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse all HTML tables from the Wikipedia page
        tables = pd.read_html(StringIO(response.text))

        # Find the table containing S&P 500 data (identified by 'Symbol' column)
        sp500_df = None
        for table in tables:
            if 'Symbol' in table.columns:
                sp500_df = table
                break
        if sp500_df is None:
            print("Cannot find S&P 500 table on Wikipedia.")
            return []

        # Rename columns to match our database schema
        sp500_df.rename(columns={
            'Symbol': 'ticker',
            'Security': 'company_name',
            'GICS Sector': 'sector',
            'GICS Sub-Industry': 'industry'
        }, inplace=True)

        # Convert to list of dictionaries and clean
        records = sp500_df.to_dict('records')
        records = clean_sp500_data(records)

        return records

    except Exception as e:
        print(f"ERROR while getting sp500 tickers from Wikipedia: {e}")
        return []

def fetch_stock_data(ticker_list, start_date=None):
    """
    Downloads historical stock price data for a list of tickers from Yahoo Finance.
    
    Downloads data in batches to avoid API rate limits and includes retry logic
    for failed downloads. Uses yfinance library to fetch OHLCV data.
    
    Args:
        ticker_list (list): List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. If None, downloads
                                    all available historical data (period='max')
    
    Returns:
        pd.DataFrame: Multi-index DataFrame with columns (Open, High, Low, Close, Volume)
                      and index (Date, Ticker). Returns empty DataFrame on error.
    """
    print("Downloading historical data (may take a while)...")
    batch_size = 50  # Download 50 tickers at a time to avoid rate limits
    all_batches = []

    # Configure download parameters
    download_params = {
        'interval': '1d',  # Daily data
        'auto_adjust': True  # Automatically adjust for stock splits and dividends
    }
    if start_date:
        download_params['start'] = start_date
    else:
        download_params['period'] = 'max'  # Download all available history

    # Download in batches with progress bar
    for i in tqdm(range(0, len(ticker_list), batch_size), desc="Downloading batches"):
        batch = ticker_list[i:i+batch_size]
        # Retry up to 3 times for each batch
        for attempt in range(3):
            try:
                batch_data = yf.download(batch, **download_params)
                all_batches.append(batch_data)
                break
            except Exception as e:
                print(f"Error downloading batch {i // batch_size + 1}, attempt {attempt + 1}: {e}")
                if attempt == 2:
                    print("Skipping this batch after 3 failed attempts.")

    if not all_batches:
        print("No data was downloaded. Exiting.")
        return pd.DataFrame()

    # Combine all batches into a single DataFrame
    return pd.concat(all_batches, axis=1)

def get_macro_tickers():
    """
    Returns a dictionary mapping macroeconomic indicator tickers to their descriptions.
    
    These tickers are used to fetch macroeconomic data that can influence stock prices,
    such as market volatility, interest rates, commodities, and currency exchange rates.
    
    Returns:
        dict: Dictionary mapping ticker symbols to human-readable descriptions
    """
    macro_tickers = {
        '^VIX': 'Volatility Index',  # CBOE Volatility Index (fear gauge)
        '^TNX': '10-Year Treasury Yield',  # 10-Year U.S. Treasury Note yield
        'CL=F': 'Crude Oil',  # WTI Crude Oil futures
        'GC=F': 'Gold',  # Gold futures
        'EURUSD=X': 'EUR/USD Exchange Rate',  # Euro to U.S. Dollar
        'JPY=X': 'USD/JPY Exchange Rate'  # U.S. Dollar to Japanese Yen
    }
    return macro_tickers

def fetch_macro_data(macro_ticker_list, start_date=None):
    """
    Downloads historical closing prices for macroeconomic indicators.
    
    Fetches daily closing prices for indicators like VIX, Treasury yields,
    commodities, and currency exchange rates from Yahoo Finance.
    
    Args:
        macro_ticker_list (list): List of macro indicator tickers (e.g., ['^VIX', 'CL=F'])
        start_date (str, optional): Start date in 'YYYY-MM-DD' format. If None, downloads
                                    all available historical data
    
    Returns:
        pd.DataFrame: DataFrame with dates as index, tickers as columns, and closing prices as values.
                      Returns empty DataFrame on error.
    """
    print(f"Fetching macro data for {', '.join(macro_ticker_list)}...")

    download_params = {
        'interval': '1d'  # Daily data
    }
    if start_date:
        download_params['start'] = start_date
    else:
        download_params['period'] = 'max'  # Download all available history

    try:
        # Download data and extract only the 'Close' column
        data = yf.download(macro_ticker_list, **download_params)['Close']
        return data
    except Exception as e:
        print(f"ERROR while fetching macro data: {e}")
        return pd.DataFrame()

def get_rss_feeds():
    """
    Returns a dictionary of RSS feed URLs for financial news sources.
    
    These feeds provide market news and financial headlines that can impact
    stock prices and market sentiment.
    
    Returns:
        dict: Dictionary mapping source names to RSS feed URLs
    """
    feeds = {
        'Investing_Stock': 'https://www.investing.com/rss/stock.rss',
        'Investing_News': 'https://www.investing.com/rss/news.rss',
        'Yahoo_Finance': 'https://finance.yahoo.com/rss/topstories',
        'Financial_Times': 'https://www.ft.com/rss/home/international',
        'Fortune_News': 'https://fortune.com/feed/fortune-feeds/?id=3230629'
    }
    return feeds

def fetch_market_news_rss():
    """
    Fetches financial news articles from multiple RSS feeds.
    
    Parses RSS feeds from various financial news sources, extracts headlines
    and publication dates, and returns a cleaned DataFrame. Removes duplicates
    and invalid entries.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['published_at', 'source_name', 'headline'].
                      All timestamps are in UTC. Returns empty DataFrame if no articles found.
    """
    feeds = get_rss_feeds()
    all_articles = []
    print("Fetching news articles from RSS feeds...")

    # Iterate through each RSS feed source
    for source, url in tqdm(feeds.items(), desc="Loading RSS"):
        try:
            feed = feedparser.parse(url)
            # Extract article information from each feed entry
            for entry in feed.entries:
                if 'published_parsed' in entry and 'title' in entry:
                    all_articles.append({
                        'published_at': entry.published_parsed,  # Time tuple from RSS
                        'source_name': source,
                        'headline': entry.title
                    })
        except Exception as e:
            print(f"ERROR while fetching news articles from {url}: {e}")
            continue
    
    if not all_articles:
        print("No news articles found. Exiting.")
        return pd.DataFrame()

    news_df = pd.DataFrame(all_articles)

    # Convert published_parsed time tuples to UTC datetime
    news_df['published_at'] = pd.to_datetime(
        news_df['published_at'].apply(lambda x: pd.Timestamp.fromtimestamp(mktime(x)) if x else pd.NaT),
        utc=True
    )

    # Clean data: remove rows with missing timestamps or headlines
    news_df = news_df.dropna(subset=['published_at', 'headline'])
    # Remove duplicate articles (same headline and publication time)
    news_df = news_df.drop_duplicates(subset=['headline', 'published_at'])

    print(f"Fetched {len(news_df)} unique headlines.")
    return news_df