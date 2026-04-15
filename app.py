"""
TradeSense - Breakout Signal Engine + BTST/STBT Scanner
Author: Mahesh Choudhary

Strategy: Breakout specialist
- 20-day high breakout + volume confirmation = main signal
- MA trend filter to avoid false breakouts
- RSI used only to avoid overbought entries (not for signal generation)

BTST logic from Moneycontrol:
- BTST: today close > yesterday high + last 2 candles green
- STBT: today close < yesterday low + last 2 candles red

TODO: add intraday VWAP-based BTST later
TODO: backtest needs more data, currently 1 year only
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import time

app = Flask(__name__, static_folder='.')
CORS(app)

# simple in-memory cache - 5 min TTL
_cache = {}
CACHE_TTL = 300  # seconds

INDICES = {
    'NIFTY 50':     '^NSEI',
    'BANK NIFTY':   '^NSEBANK',
    'INDIA VIX':    '^INDIAVIX',
    'NIFTY NEXT50': '^NSMIDCP'
}

# top NSE stocks to scan for BTST/STBT
SCAN_LIST = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
    'SBIN', 'WIPRO', 'AXISBANK', 'BAJFINANCE', 'MARUTI',
    'SUNPHARMA', 'TATAMOTORS', 'NTPC', 'ONGC', 'POWERGRID',
    'LT', 'TITAN', 'ADANIENT', 'COALINDIA', 'ULTRACEMCO'
]

def safe(val):
    try:
        return round(float(val), 2)
    except:
        return None

def from_cache(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
    return None

def to_cache(key, data):
    _cache[key] = (data, time.time())

def fetch_ohlcv(symbol, period="1y"):
    import yfinance as yf
    cached = from_cache(symbol + period)
    if cached is not None:
        return cached
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No data for: " + symbol)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    to_cache(symbol + period, df)
    return df

def add_indicators(df):
    df = df.copy()
    df['MA50']  = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l
    df['RSI']      = 100 - (100 / (1 + rs))
    df['VWAP']     = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['vol_avg20'] = df['Volume'].rolling(20).mean()
    df['high20']    = df['High'].rolling(20).max()
    df['low20']     = df['Low'].rolling(20).min()
    return df

def get_trade_signal(df):
    """
    Breakout-first strategy.
    Primary signals: 20-day breakout + volume spike
    Trend filter: MA50 > MA200
    RSI used only to reject overbought entries
    """
    reasons  = []
    warnings = []
    score    = 0

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    close    = safe(latest['Close'])
    ma50     = safe(latest['MA50'])
    ma200    = safe(latest['MA200'])
    rsi      = safe(latest['RSI'])
    vol      = safe(latest['Volume'])
    vol_avg  = safe(latest['vol_avg20'])
    high20   = safe(df['High'].rolling(20).max().iloc[-2])  # prev day 20-high
    low20    = safe(df['Low'].rolling(20).min().iloc[-2])

    # --- trend filter (mandatory for breakout to be valid) ---
    trend_ok = False
    if ma50 and ma200:
        if ma50 > ma200:
            trend_ok = True
            score += 1
            reasons.append("Trend filter passed - MA50 above MA200, uptrend intact")
        else:
            score -= 1
            reasons.append("Trend filter failed - MA50 below MA200, risky to buy breakouts here")

    # --- main breakout signal ---
    if close and high20:
        if close > high20:
            score += 3  # this is the core signal, high weight
            reasons.append("Breaking out above 20-day high (" + str(int(high20)) + ") - strong momentum")
        elif close and low20 and close < low20:
            score -= 3
            reasons.append("Breaking down below 20-day low (" + str(int(low20)) + ") - bearish breakdown")

    # --- volume confirmation - breakout without volume is weak ---
    if vol and vol_avg:
        vol_ratio = round(vol / vol_avg, 1)
        if vol_ratio > 1.5:
            score += 2  # I trust volume-confirmed breakouts a lot
            reasons.append("Volume " + str(vol_ratio) + "x above average - institutions participating")
        elif vol_ratio < 0.8:
            score -= 1
            reasons.append("Volume low (" + str(vol_ratio) + "x avg) - breakout not confirmed, be careful")
            warnings.append("Low volume breakout - higher chance of fakeout")

    # --- RSI: only used to filter overbought (not for signal) ---
    if rsi:
        if rsi > 75:
            score -= 1
            reasons.append("RSI at " + str(rsi) + " - very overbought, late entry risk")
            warnings.append("RSI above 75 - stock may pull back before continuing")
        elif rsi < 35:
            score += 1
            reasons.append("RSI at " + str(rsi) + " - oversold bounce possible")

    # price vs MA50 - secondary confirmation
    if close and ma50:
        if close > ma50:
            score += 1
            reasons.append("Price above MA50 - short-term trend supportive")
        else:
            score -= 1
            reasons.append("Price below MA50 - short-term trend not supportive")

    if score >= 4:
        signal = "BUY"
    elif score <= -3:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = min(int((abs(score) / 8.0) * 100), 95)
    risk_note  = warnings[0] if warnings else "No major red flags."

    # entry / SL / target
    entry = close
    if signal == "BUY":
        sl     = round(close * 0.97, 2)   # 3% SL
        target = round(close * 1.06, 2)   # 6% target, 2:1 RR
    elif signal == "SELL":
        sl     = round(close * 1.03, 2)
        target = round(close * 0.94, 2)
    else:
        sl     = None
        target = None

    return {
        "signal":     signal,
        "score":      round(score, 2),
        "confidence": confidence,
        "reasons":    reasons,
        "risk_note":  risk_note,
        "entry":      entry,
        "stop_loss":  sl,
        "target":     target
    }

def check_btst(df):
    """
    BTST/STBT logic - Moneycontrol method
    BTST: today close > yesterday high AND last 2 candles green
    STBT: today close < yesterday low AND last 2 candles red
    """
    if len(df) < 3:
        return {"signal": "NONE", "reason": "Not enough data"}

    today     = df.iloc[-1]
    yesterday = df.iloc[-2]
    day_before= df.iloc[-3]

    today_close      = safe(today['Close'])
    yesterday_high   = safe(yesterday['High'])
    yesterday_low    = safe(yesterday['Low'])
    yesterday_open   = safe(yesterday['Open'])
    today_open       = safe(today['Open'])
    day_before_open  = safe(day_before['Open'])
    day_before_close = safe(day_before['Close'])

    # check last 2 candles green (close > open)
    candle1_green = today['Close'] > today['Open']
    candle2_green = yesterday['Close'] > yesterday['Open']
    two_green = candle1_green and candle2_green

    # check last 2 candles red
    candle1_red = today['Close'] < today['Open']
    candle2_red = yesterday['Close'] < yesterday['Open']
    two_red = candle1_red and candle2_red

    # BTST condition
    if today_close and yesterday_high:
        if today_close > yesterday_high and two_green:
            target_btst = round(today_close * 1.025, 2)  # 2-3% target next morning
            sl_btst     = round(today_close * 0.985, 2)
            return {
                "signal":    "BTST",
                "entry":     today_close,
                "target":    target_btst,
                "stop_loss": sl_btst,
                "reason":    "Today close (" + str(today_close) + ") above yesterday high (" + str(yesterday_high) + ") + 2 green candles confirmed",
                "timing":    "Buy today 3:25-3:30 PM, sell tomorrow 9:15-10:00 AM"
            }

    # STBT condition
    if today_close and yesterday_low:
        if today_close < yesterday_low and two_red:
            target_stbt = round(today_close * 0.975, 2)
            sl_stbt     = round(today_close * 1.015, 2)
            return {
                "signal":    "STBT",
                "entry":     today_close,
                "target":    target_stbt,
                "stop_loss": sl_stbt,
                "reason":    "Today close (" + str(today_close) + ") below yesterday low (" + str(yesterday_low) + ") + 2 red candles confirmed",
                "timing":    "Sell today 3:25-3:30 PM, buy back tomorrow 9:15-10:00 AM"
            }

    return {"signal": "NONE", "reason": "No BTST/STBT setup today"}

def run_backtest(df):
    """
    Simple backtest - check signal accuracy over past data
    For each point, generate signal, check price after 5 days
    Only need ~50 samples to get a rough idea
    """
    correct = 0
    total   = 0
    wins    = []

    # need at least 220 rows (200 for MA + buffer)
    if len(df) < 220:
        return {"accuracy": None, "total_trades": 0, "note": "Not enough data for backtest"}

    for i in range(200, len(df) - 6, 3):  # step 3 to avoid overlapping trades
        slice_df = df.iloc[:i+1].copy()
        slice_df = add_indicators(slice_df)

        try:
            sig = get_trade_signal(slice_df)
        except:
            continue

        if sig['signal'] not in ['BUY', 'SELL']:
            continue

        future  = safe(df['Close'].iloc[i+5])
        current = safe(df['Close'].iloc[i])
        if not future or not current:
            continue

        total += 1
        pct_change = ((future - current) / current) * 100

        if sig['signal'] == 'BUY' and future > current:
            correct += 1
            wins.append(pct_change)
        elif sig['signal'] == 'SELL' and future < current:
            correct += 1
            wins.append(abs(pct_change))

    accuracy  = round((correct / total * 100), 1) if total > 0 else None
    avg_win   = round(sum(wins) / len(wins), 2) if wins else None

    return {
        "accuracy":     accuracy,
        "total_trades": total,
        "correct":      correct,
        "avg_win_pct":  avg_win,
        "note":         "5-day forward return based backtest"
    }

def prepare_response(name, symbol, df, is_index=False, include_backtest=False):
    df = add_indicators(df)
    latest = df.iloc[-1]

    hist = df[['Close', 'MA50', 'MA200', 'RSI']].tail(100).reset_index()
    hist['Date'] = hist['Date'].astype(str)
    history_clean = []
    for _, row in hist.iterrows():
        history_clean.append({
            'Date':  row['Date'],
            'Close': safe(row['Close']),
            'MA50':  safe(row['MA50']),
            'MA200': safe(row['MA200']),
            'RSI':   safe(row['RSI'])
        })

    result = {
        "name":    name,
        "ticker":  symbol,
        "date":    str(df.index[-1].date()),
        "price":   safe(latest['Close']),
        "ma50":    safe(latest['MA50']),
        "ma200":   safe(latest['MA200']),
        "rsi":     safe(latest['RSI']),
        "vwap":    safe(latest['VWAP']),
        "history": history_clean
    }

    if not is_index:
        sig  = get_trade_signal(df)
        btst = check_btst(df)
        result.update(sig)
        result["btst"] = btst
        if include_backtest:
            result["backtest"] = run_backtest(df)
    else:
        result.update({
            "signal": "INDEX", "confidence": None,
            "reasons": [], "risk_note": "Market context only.",
            "entry": None, "stop_loss": None, "target": None,
            "btst": None
        })

    return result

# --- routes ---

@app.route('/')
def home():
    return send_from_directory('.', 'dashboard.html')

@app.route('/api/indices')
def api_indices():
    results = []
    errors  = []
    for name, symbol in INDICES.items():
        try:
            df = fetch_ohlcv(symbol)
            results.append(prepare_response(name, symbol, df, is_index=True))
        except Exception as e:
            errors.append({"name": name, "error": str(e)})
    return jsonify({"data": results, "errors": errors})

@app.route('/api/search')
def api_search():
    ticker  = request.args.get('ticker', '').strip().upper()
    backtest = request.args.get('backtest', 'false').lower() == 'true'
    if not ticker:
        return jsonify({"error": "ticker param missing"}), 400
    symbol = ticker if ticker.startswith('^') else ticker + ".NS"
    try:
        df     = fetch_ohlcv(symbol, period="2y")  # 2y for backtest
        result = prepare_response(ticker, symbol, df, include_backtest=backtest)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/btst_scan')
def api_btst_scan():
    """Scan SCAN_LIST and return all BTST/STBT candidates"""
    results = []
    errors  = []
    for ticker in SCAN_LIST:
        symbol = ticker + ".NS"
        try:
            df   = fetch_ohlcv(symbol, period="1mo")
            df_i = add_indicators(df)
            btst = check_btst(df_i)
            if btst['signal'] in ['BTST', 'STBT']:
                results.append({
                    "ticker":    ticker,
                    "price":     safe(df.iloc[-1]['Close']),
                    "signal":    btst['signal'],
                    "entry":     btst['entry'],
                    "target":    btst['target'],
                    "stop_loss": btst['stop_loss'],
                    "reason":    btst['reason'],
                    "timing":    btst['timing']
                })
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
    return jsonify({"data": results, "errors": errors, "scanned": len(SCAN_LIST)})

if __name__ == '__main__':
    print("")
    print("=" * 50)
    print("  TradeSense - Breakout Signal Engine")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    print("")
    app.run(debug=True, port=5000)
