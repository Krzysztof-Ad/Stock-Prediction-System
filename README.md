# Stock Prediction System

An educational project to build an end-to-end Machine Learning system for
predicting stock price direction.

> **⚠️ Disclaimer:** This project is for educational purposes only and does not
> constitute financial advice. All models are experimental, and trading on
> financial markets carries significant risk.

## Project Goals

The main objective is to build a complete system that:
1.  Fetches and processes financial data using `yfinance` and `pandas`.
2.  Trains a machine learning model (`scikit-learn`) to predict price direction (Up/Down).
3.  Serves this model's predictions via a simple REST API (using `Flask`).

## Current Focus

We are currently in **Phase 2: Feature Engineering**:
* [x] Build a new (`feature_pipeline`) to read data from the (`stock_data_daily`) table.
* [x] **Time-Series Features**: Generate technical indicators and lagged returns for all 500 stocks.
* [ ] **Sentiment Features**: Analyze headlines from the (`market_news`) table and create a daily sentiment score.
* [ ] **Cross-Sectional Features**: Calculate relative strength vs sector and market-wide feature rankings.
* [ ] Save all computed features to the new (`stock_data_features`) table.

## Achievements
**Phase 1: Data Universe (ETL pipeline)**

* [x] Setup PostgreSQL + TimescaleDB using Docker.
* [x] Build etl_pipeline to fetch S&P 500 tickers (from Wikipedia).
* [x] Build ETL script to download and store all historical OHLCV data.
* [x] Add macro data extractor.
* [x] Add news data extractor (RSS).
* [x] Implement delta load to only account for fresh data.