import pandas as pd
import requests
from io import StringIO


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


print(get_sp500_tickers()[:5])