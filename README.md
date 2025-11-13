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
4.  (Future) Includes basic user authentication and portfolio tracking.

## Tech Stack

* **Python 3.11** (Managed via `venv`)
* **Data Science:** Pandas, NumPy, Scikit-learn
* **Data Retrieval:** yfinance
* **Backend API:** Flask
* **Database:** Flask-SQLAlchemy (with SQLite for development)

## Current Focus & Next Steps

We are currently in **Phase 1: ML Prototyping**.

The immediate goal is to develop and validate the core prediction model in a Jupyter Notebook.

* [ ] **Data Acquisition:** Fetch historical data for a sample ticker (e.g., `MSFT`).
* [ ] **Feature Engineering:** Create technical indicators to use as features (e.g., Moving Averages, RSI).
* [ ] **Initial Model:** Train a baseline model (like `RandomForestClassifier`).
* [ ] **Validation:** Properly evaluate the model using time-series-aware cross-validation (e.g., `TimeSeriesSplit`).