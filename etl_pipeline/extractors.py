import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm
from io import StringIO
import feedparser
from time import mktime


def clean_sp500_data(records):
    allowed_columns = {'ticker', 'company_name', 'sector', 'industry'}
    cleaned = [{k: v for k, v in record.items() if k in allowed_columns} for record in records]
    return cleaned


def get_sp500_tickers():
    print('Getting sp500 tickers from Wikipedia...')

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/117.0 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))

        sp500_df = None
        for table in tables:
            if 'Symbol' in table.columns:
                sp500_df = table
                break
        if sp500_df is None:
            print("Cannot find S&P 500 table on Wikipedia.")
            return []

        sp500_df.rename(columns={
            'Symbol': 'ticker',
            'Security': 'company_name',
            'GICS Sector': 'sector',
            'GICS Sub-Industry': 'industry'
        }, inplace=True)

        records = sp500_df.to_dict('records')
        records = clean_sp500_data(records)

        return records

    except Exception as e:
        print(f"ERROR while getting sp500 tickers from Wikipedia: {e}")
        return []

def fetch_stock_data(ticker_list, start_date=None):
    print("Downloading historical data (may take a while)...")
    batch_size = 50
    all_batches = []

    download_params = {
        'interval': '1d',
        'auto_adjust': True
    }
    if start_date:
        download_params['start'] = start_date
    else:
        download_params['period'] = 'max'

    for i in tqdm(range(0, len(ticker_list), batch_size), desc="Downloading batches"):
        batch = ticker_list[i:i+batch_size]
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

    return pd.concat(all_batches, axis=1)

def get_macro_tickers():
    macro_tickers = {
        '^VIX': 'Volatility Index',
        '^TNX': '10-Year Treasury Yield',
        'CL=F': 'Crude Oil',
        'GC=F': 'Gold',
        'EURUSD=X': 'EUR/USD Exchange Rate',
        'JPY=X': 'USD/JPY Exchange Rate'
    }
    return macro_tickers

def fetch_macro_data(macro_ticker_list, start_date=None):
    print(f"Fetching macro data for {', '.join(macro_ticker_list)}...")

    download_params = {
        'interval': '1d'
    }
    if start_date:
        download_params['start'] = start_date
    else:
        download_params['period'] = 'max'

    try:
        data = yf.download(macro_ticker_list, **download_params)['Close']
        return data
    except Exception as e:
        print(f"ERROR while fetching macro data: {e}")
        return pd.DataFrame()

def get_rss_feeds():
    feeds = {
        'Investing_Stock': 'https://www.investing.com/rss/stock.rss',
        'Investing_News': 'https://www.investing.com/rss/news.rss',
        'Yahoo_Finance': 'https://finance.yahoo.com/rss/topstories',
        'Financial_Times': 'https://www.ft.com/rss/home/international',
        'Fortune_News': 'https://fortune.com/feed/fortune-feeds/?id=3230629'
    }
    return feeds

def fetch_market_news_rss():
    feeds = get_rss_feeds()
    all_articles = []
    print("Fetching news articles from RSS feeds...")

    for source, url in tqdm(feeds.items(), desc="Loading RSS"):
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if 'published_parsed' in entry and 'title' in entry:
                    all_articles.append({
                        'published_at': entry.published_parsed,
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

    news_df['published_at'] = pd.to_datetime(
        news_df['published_at'].apply(lambda x: pd.Timestamp.fromtimestamp(mktime(x)) if x else pd.NaT),
        utc=True
    )

    news_df = news_df.dropna(subset=['published_at', 'headline'])
    news_df = news_df.drop_duplicates(subset=['headline', 'published_at'])

    print(f"Fetched {len(news_df)} unique headlines.")
    return news_df