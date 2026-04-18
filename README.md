# Indian Stock Price Prediction
An interactive Flask website for a data learning project. It pulls historical stock data from `yfinance`, focuses on Indian companies, engineers a few beginner-friendly features, trains a simple linear regression model, and shows a next-session price prediction with charts and model metrics.

## Features
- Search by Indian stock ticker like `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, or `HDFCBANK.NS`
- Open a searchable built-in list of Indian company tickers
- Choose history windows from 6 months to 5 years
- View close price with 7-day and 21-day moving averages
- See prices in Indian rupees and a simple next-close prediction band
- Learn from model metrics like MAE and R2

## Project structure
- `app.py`: Flask server and API routes
- `predictor.py`: yfinance download, feature engineering, and regression model
- `templates/index.html`: Website layout
- `static/styles.css`: Custom styling
- `static/app.js`: Frontend interactions and Chart.js rendering

## Run locally

## Important

Run this command only after all dependencies are installed:

```powershell
.\.venv\Scripts\python.exe app.py
```

Run this command only after all dependencies are installed:
<!-- powershell -->
.\.venv\Scripts\python.exe app.py



Install everything first, then run the website.

1. Create a virtual environment:
<!-- powershell -->
python -m venv .venv


2. Install all required packages:
<!-- powershell -->
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

3. Only after installation is complete, run the website:
<!-- powershell -->
.\.venv\Scripts\python.exe app.py


4. Open `http://127.0.0.1:5000`
