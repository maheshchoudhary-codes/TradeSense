# TradeSense: Explainable Stock Decision Engine

# Overview

TradeSense is a stock analysis tool designed to help make better trading decisions. Instead of only giving BUY or SELL signals, it also explains the reasoning behind each decision.

The system uses technical indicators such as Moving Averages (MA50, MA200), RSI, and VWAP to analyze market trends and momentum. Based on this, it generates signals along with a confidence score.

# Features

- Identifies market trend (Bullish / Bearish)
- Analyzes momentum using RSI
- Uses VWAP to understand price positioning
- Generates BUY / SELL / HOLD signals
- Provides clear reasoning for each signal
- Includes confidence scoring
- Detects potential BTST (Buy Today Sell Tomorrow) opportunities
  
# Tech Stack

- Python  
- Flask  
- Pandas  
- NumPy  
- yfinance
  
# How to Run

1. Install dependencies:
   pip install flask flask-cors pandas numpy yfinance matplotlib
   
3. Run the application:
   python app.py
   
5. Open in browser:
   http://localhost:5000


# Output

The system provides:
- Final Signal (BUY / SELL / HOLD)
- Explanation of the decision
- Confidence score

# Purpose

This project was built to create a simple, data-driven trading assistant that helps users make more logical and informed decisions instead of relying on guesswork.

# Disclaimer

This project is for educational purposes only and should not be considered financial advice.
       
