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

## Current Focus & Achievements

We are currently in **Phase 1: Data Universe (ETL pipeline)**:

* [x] Setup PostgreSQL + TimescaleDB using Docker.

* [x] Build etl_pipeline to fetch S&P 500 tickers (from Wikipedia).

* [x] Build ETL script to download and store all historical OHLCV data.

* [x] Add macro data extractor.

* [ ] Add news/sentiment data extractor (NewsAPI).