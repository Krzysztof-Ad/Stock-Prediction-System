"""
Run FinBERT on stored market news headlines, aggregate the scores per day,
and push them into the database for downstream features.
"""

import pandas as pd
import torch
from transformers import pipeline
from sqlalchemy import text
from tqdm import tqdm
from etl_pipeline.db_manager import get_db_engine, create_sentiment_table

def run_sentiment_analysis():
    """
    Fetch news, score each headline, roll up daily sentiment, and store it.
    """
    engine = get_db_engine()

    # Make sure the table exists before we write anything
    print("Creating sentiment table...")
    create_sentiment_table()

    # Check the last day we handled so we skip repeats
    print("Checking for last processed date...")
    try:
        with engine.connect() as connection:
            last_date = connection.execute(text("SELECT MAX(date) FROM sentiment_daily")).scalar()

        if last_date:
            print(f"Last sentiment from: {last_date}. Loading fresh news...")

            query = text("SELECT published_at, headline FROM market_news WHERE published_at >= :last_date")
            news_df = pd.read_sql(query, engine, params={"last_date": last_date})
        else:
            print("No sentiment data to load. Using all news...")
            query = text("SELECT published_at, headline FROM market_news")
            news_df = pd.read_sql(query, engine)

    except Exception as e:
        print(f"Error fetching news: {e}")
        return

    if news_df.empty:
        print("No news found. Run ETL first...")
        return

    print(f"Fetched {len(news_df)} headlines from database. Loading FinBERT...")

    # Use a GPU if we can, otherwise stick with the CPU
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print(f"Discovered GPU device: {torch.cuda.get_device_name(0)}. Using GPU.")
    else:
        print("GPU not found. Using CPU.")

    # Load FinBERT once and reuse it for every batch
    try:
        sentiment_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Analyzing sentiment...")
    label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    results = []

    batch_size = 64
    headlines = news_df['headline'].tolist()

    # Handle the headlines in small batches so memory stays stable
    for i in tqdm(range(0, len(headlines), batch_size), desc="FinBERT Processing"):
        batch = headlines[i:i + batch_size]
        try:
            predictions = sentiment_pipeline(batch, truncation=True, max_length=512)
            for pred in predictions:
                results.append(label_map[pred['label']] * pred['score'])
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            results.extend([0.0] * len(batch))

    news_df['sentiment_score'] = results

    # Snap every timestamp to midnight so grouping is easy
    news_df['date'] = pd.to_datetime(news_df['published_at']).dt.normalize()

    daily_sentiment = news_df.groupby(['date']).agg(
        avg_sentiment_score=('sentiment_score', 'mean'),
        article_count=('sentiment_score', 'count')
    ).reset_index()

    daily_sentiment['ticker'] = 'SP500'
    daily_sentiment['source_name'] = 'RSS_Aggregated'

    print(f"Saving {len(daily_sentiment)} daily summaries to database...")

    # Count how many daily rows we successfully saved
    success_count = 0
    for index, row in tqdm(daily_sentiment.iterrows(), total=len(daily_sentiment), desc="Saving to DB"):
        try:
            pd.DataFrame([row]).to_sql('sentiment_daily', engine, if_exists='append', index=False)
            success_count += 1
        except Exception:
            pass

    print(f"Successfully saved {success_count} sentiment analysis results.")

if __name__ == "__main__":
    run_sentiment_analysis()