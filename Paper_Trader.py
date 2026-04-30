import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import ta
import xgboost as xgb
import joblib
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()
if not os.getenv("ALPACA_API_KEY"):
    load_dotenv(r"C:\Users\ryanc\OneDrive\Desktop\algo-trading-sp500\.env")

# ── Credentials ────────────────────────────────────────────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

print("Script starting...")
print("API_KEY found:", API_KEY is not None)
print("SECRET_KEY found:", SECRET_KEY is not None)

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# ── Verify connection ──────────────────────────────────────────────────────
account         = api.get_account()
portfolio_value = float(account.portfolio_value)
print(f"Account status : {account.status}")
print(f"Portfolio value: ${portfolio_value:,.2f}")
print(f"Buying power   : ${float(account.buying_power):,.2f}")
print(f"Cash           : ${float(account.cash):,.2f}")

# ── S&P 500 ticker cache file ──────────────────────────────────────────────
CACHE_FILE = "sp500_cache.json"

def save_cache(tickers):
    with open(CACHE_FILE, "w") as f:
        json.dump({
            "tickers"  : tickers,
            "updated"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f)
    print(f"  ✓ Cache saved with {len(tickers)} tickers")

def load_cache():
    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        print(f"  ✓ Loaded cached list from {data['updated']} ({len(data['tickers'])} tickers)")
        return data["tickers"]
    except:
        return None

def get_sp500_tickers():
    print("\nFetching S&P 500 from iShares IVV holdings...")
    try:
        url = (
            "https://www.ishares.com/us/products/239726/IVV/"
            "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
        )
        df = pd.read_csv(url, skiprows=9)
        df = df[df["Asset Class"] == "Equity"]
        tickers = df["Ticker"].dropna().tolist()
        tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]

        if len(tickers) < 400:
            raise ValueError(f"Only got {len(tickers)} tickers — likely incomplete")

        save_cache(tickers)
        print(f"  ✓ Successfully fetched {len(tickers)} stocks from iShares IVV")
        return tickers

    except Exception as e:
        print(f"  ⚠ iShares fetch failed: {e}")
        print("  Falling back to last successful cache...")

        cached = load_cache()
        if cached:
            return cached

        print("  ⚠ No cache found — using hardcoded fallback list")
        return [
            "AAPL","MSFT","GOOGL","AMZN","META",
            "NVDA","V","JPM","ORCL","COST",
            "ADBE","CRM","AMD","NFLX","PYPL",
            "MA","UNH","HD","BAC","QCOM"
        ]

# ── Feature engineering ────────────────────────────────────────────────────
def get_features(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period="1y", progress=False)
            if df.empty:
                raise ValueError("Empty dataframe")
            df.columns = df.columns.get_level_values(0)
            df["rsi"]         = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["macd"]        = ta.trend.MACD(df["Close"]).macd()
            df["macd_sig"]    = ta.trend.MACD(df["Close"]).macd_signal()
            df["bb_high"]     = ta.volatility.BollingerBands(df["Close"]).bollinger_hband()
            df["bb_low"]      = ta.volatility.BollingerBands(df["Close"]).bollinger_lband()
            df["vol_ma"]      = df["Volume"].rolling(20).mean()
            df["returns_1d"]  = df["Close"].pct_change(1)
            df["returns_5d"]  = df["Close"].pct_change(5)
            df["returns_20d"] = df["Close"].pct_change(20)
            df["volatility"]  = df["returns_1d"].rolling(20).std()
            df["ma_20"]       = df["Close"].rolling(20).mean()
            df["ma_50"]       = df["Close"].rolling(50).mean()
            df["ma_cross"]    = (df["ma_20"] > df["ma_50"]).astype(int)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"  ⚠ Attempt {attempt+1}/3 failed for {ticker}: {e}")
            time.sleep(5)
    return pd.DataFrame()

# ── Pre-screen filter ──────────────────────────────────────────────────────
def passes_screen(df, ticker):
    try:
        if len(df) < 60:
            return False, "not enough history"

        latest = df.iloc[-1]

        if latest["Close"] < 10:
            return False, f"price too low (${latest['Close']:.2f})"

        avg_vol = df["Volume"].mean()
        if avg_vol < 500_000:
            return False, f"low volume ({avg_vol:,.0f} avg)"

        avg_dollar_vol = (df["Close"] * df["Volume"]).mean()
        if avg_dollar_vol < 5_000_000:
            return False, f"low dollar volume (${avg_dollar_vol:,.0f} avg)"

        rsi = df["rsi"].iloc[-1]
        if rsi < 30 or rsi > 75:
            return False, f"RSI out of range ({rsi:.1f})"

        if latest["Close"] < latest["ma_50"] * 0.95:
            return False, "price below 50MA"

        vol = df["volatility"].iloc[-1]
        if vol > 0.05:
            return False, f"too volatile ({vol:.3f})"

        return True, "passed"

    except Exception as e:
        return False, f"screen error: {e}"

# ── Load model ─────────────────────────────────────────────────────────────
features = ["rsi","macd","macd_sig","bb_high","bb_low",
            "vol_ma","returns_1d","returns_5d","returns_20d",
            "volatility","ma_20","ma_50","ma_cross"]

model = joblib.load("models/aapl_xgb_model.joblib")
print("Model loaded successfully")

# ── Get signal ─────────────────────────────────────────────────────────────
def get_signal(ticker, skip_screen=False):
    try:
        df = get_features(ticker)
        if df.empty or len(df) < 50:
            return None

        if not skip_screen:
            passed, reason = passes_screen(df, ticker)
            if not passed:
                return None

        latest     = df[features].iloc[-1:]
        pred       = model.predict(latest)[0]
        prob       = model.predict_proba(latest)[0]
        price      = df["Close"].iloc[-1]
        confidence = prob[1] if pred == 1 else prob[0]

        return {
            "ticker"    : ticker,
            "signal"    : "BUY" if pred == 1 else "SELL",
            "confidence": confidence,
            "price"     : price,
            "time"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"  ✗ Error getting signal for {ticker}: {e}")
        return None

# ── Position sizing ────────────────────────────────────────────────────────
def get_position_size(confidence, price, portfolio_value):
    if confidence >= 0.80:
        dollars = portfolio_value * 0.05
    elif confidence >= 0.70:
        dollars = portfolio_value * 0.03
    elif confidence >= 0.65:
        dollars = portfolio_value * 0.02
    else:
        dollars = portfolio_value * 0.01
    qty = max(1, int(dollars / price))
    return qty, dollars

# ── Portfolio exposure ─────────────────────────────────────────────────────
def get_portfolio_exposure():
    positions      = api.list_positions()
    total_invested = sum(float(p.market_value) for p in positions)
    return total_invested

# ── Place trade ────────────────────────────────────────────────────────────
def place_trade(ticker, signal, confidence, portfolio_value):
    try:
        total_invested = get_portfolio_exposure()
        exposure_pct   = total_invested / portfolio_value
        max_exposure   = 0.40

        try:
            position      = api.get_position(ticker)
            has_position  = True
            current_qty   = int(float(position.qty))
            unrealized_pl = float(position.unrealized_pl)
            current_price = float(position.current_price)
        except:
            has_position  = False
            current_qty   = 0
            unrealized_pl = 0
            current_price = 0

        if signal == "BUY" and confidence > 0.60:
            if exposure_pct >= max_exposure:
                print(f"  — Portfolio at {exposure_pct:.1%} exposure (max 40%) — skipping")
                return None

            price        = current_price if has_position else get_signal(ticker, skip_screen=True)["price"]
            qty, dollars = get_position_size(confidence, price, portfolio_value)

            if not has_position:
                order = api.submit_order(
                    symbol        = ticker,
                    qty           = qty,
                    side          = "buy",
                    type          = "market",
                    time_in_force = "day"
                )
                print(f"  ✓ BUY {qty} share(s) of {ticker} (~${dollars:,.0f}) — {exposure_pct:.1%} deployed")
                return order

            elif unrealized_pl > 0 or confidence >= 0.65:
                order = api.submit_order(
                    symbol        = ticker,
                    qty           = qty,
                    side          = "buy",
                    type          = "market",
                    time_in_force = "day"
                )
                print(f"  ✓ Adding {qty} share(s) to {ticker} (~${dollars:,.0f}) — {exposure_pct:.1%} deployed")
                return order

            else:
                print(f"  — Position at loss, confidence below 65% — holding")
                return None

        elif signal == "SELL" and has_position:
            order = api.submit_order(
                symbol        = ticker,
                qty           = current_qty,
                side          = "sell",
                type          = "market",
                time_in_force = "day"
            )
            print(f"  ✓ SELL closing full position ({current_qty} shares) of {ticker}")
            return order

        else:
            print(f"  — No action: signal={signal}, has_position={has_position}, confidence={confidence:.1%}")
            return None

    except Exception as e:
        print(f"  ✗ Order failed: {e}")
        return None

# ── Ghost list handler ─────────────────────────────────────────────────────
def handle_ghost_list(sp500):
    positions    = api.list_positions()
    held_tickers = [p.symbol for p in positions]
    ghost_list   = []

    for ticker in held_tickers:
        if ticker not in sp500:
            ghost_list.append((ticker, "left S&P 500"))
            continue

        df = get_features(ticker)
        if not df.empty:
            passed, reason = passes_screen(df, ticker)
            if not passed:
                ghost_list.append((ticker, f"failed screen: {reason}"))

    if not ghost_list:
        print("  ✓ All held positions still in active universe")
        return

    print(f"\n── Ghost List Scan (Sell Only) ────────────")
    for ticker, reason in ghost_list:
        print(f"\n  {ticker} — {reason}")
        try:
            position      = api.get_position(ticker)
            current_qty   = int(float(position.qty))
            unrealized_pl = float(position.unrealized_pl)
            current_price = float(position.current_price)

            if current_price < 1.00:
                print(f"  ⚠ Price below $1 — force closing")
                api.submit_order(
                    symbol        = ticker,
                    qty           = current_qty,
                    side          = "sell",
                    type          = "market",
                    time_in_force = "day"
                )
                continue

            signal_data = get_signal(ticker, skip_screen=True)
            if signal_data is None:
                print(f"  — No data, holding for now (P&L: ${unrealized_pl:+.2f})")
                continue

            print(f"  Signal    : {signal_data['signal']}")
            print(f"  Confidence: {signal_data['confidence']:.1%}")
            print(f"  P&L       : ${unrealized_pl:+.2f}")

            if signal_data["signal"] == "SELL":
                api.submit_order(
                    symbol        = ticker,
                    qty           = current_qty,
                    side          = "sell",
                    type          = "market",
                    time_in_force = "day"
                )
                print(f"  ✓ SELL — closed ghost position ({current_qty} shares)")
            else:
                print(f"  — Holding, signal still BUY")

        except Exception as e:
            print(f"  ✗ Error: {e}")

# ── Portfolio summary ──────────────────────────────────────────────────────
def print_portfolio():
    print("\n── Current Positions ─────────────────────")
    positions = api.list_positions()
    if not positions:
        print("  No open positions")
    total_pnl      = 0
    total_invested = 0
    for p in positions:
        pnl            = float(p.unrealized_pl)
        qty            = int(float(p.qty))
        value          = float(p.market_value)
        total_pnl     += pnl
        total_invested += value
        print(f"  {p.symbol:<6} {qty} shares | Value: ${value:,.2f} | P&L: ${pnl:+.2f}")
    exposure = (total_invested / portfolio_value) * 100
    print(f"\n  Total invested    : ${total_invested:,.2f} ({exposure:.1f}% of portfolio)")
    print(f"  Total P&L         : ${total_pnl:+.2f}")
    print(f"  Cash remaining    : ${float(account.cash):,.2f}")
    print(f"  Max exposure limit: ${portfolio_value * 0.40:,.2f} (40%)")
    print("\n── Recent Orders ──────────────────────────")
    orders = api.list_orders(status="all", limit=10)
    for o in orders:
        print(f"  {o.symbol} {o.side.upper()} {o.qty} shares — {o.status} @ {o.created_at}")

# ── Main ───────────────────────────────────────────────────────────────────
sp500 = get_sp500_tickers()

# Handle ghost list first
print("\n── Checking Ghost List ────────────────────")
handle_ghost_list(sp500)

# ── Phase 1: Scan all stocks and collect signals ───────────────────────────
total        = len(sp500)
screened     = 0
passed_count = 0
all_signals  = []

print(f"\n── Phase 1: Scanning {total} S&P 500 stocks ──")

for ticker in sp500:
    screened += 1
    signal_data = get_signal(ticker)
    if signal_data is None:
        continue
    passed_count += 1
    all_signals.append(signal_data)
    print(f"  {ticker} | {signal_data['signal']} | {signal_data['confidence']:.1%} | ${signal_data['price']:.2f}")

# ── Phase 2: Sort by confidence and trade best opportunities first ─────────
buy_signals  = [s for s in all_signals if s["signal"] == "BUY"]
sell_signals = [s for s in all_signals if s["signal"] == "SELL"]

# Sort buys highest confidence first
buy_signals.sort(key=lambda x: x["confidence"], reverse=True)

print(f"\n── Phase 2: Executing Trades (Best First) ─")
print(f"  Buy signals  : {len(buy_signals)}")
print(f"  Sell signals : {len(sell_signals)}")

# Execute sells first to free up capital
print(f"\n  -- Processing SELL signals --")
for signal_data in sell_signals:
    print(f"\n  {signal_data['ticker']} | SELL | {signal_data['confidence']:.1%}")
    place_trade(
        signal_data["ticker"],
        signal_data["signal"],
        signal_data["confidence"],
        portfolio_value
    )

# Execute buys in order of confidence (highest first)
print(f"\n  -- Processing BUY signals (highest confidence first) --")
for signal_data in buy_signals:
    total_invested = get_portfolio_exposure()
    exposure_pct   = total_invested / portfolio_value
    if exposure_pct >= 0.40:
        print(f"\n  Portfolio at {exposure_pct:.1%} — capacity reached, stopping")
        print(f"  Remaining signals skipped: {len(buy_signals) - buy_signals.index(signal_data)}")
        break
    print(f"\n  {signal_data['ticker']} | BUY | {signal_data['confidence']:.1%} | ${signal_data['price']:.2f}")
    place_trade(
        signal_data["ticker"],
        signal_data["signal"],
        signal_data["confidence"],
        portfolio_value
    )

# ── Scan summary ───────────────────────────────────────────────────────────
print(f"\n── Scan Summary ───────────────────────────")
print(f"  Total scanned  : {screened}")
print(f"  Passed screen  : {passed_count}")
print(f"  Filtered out   : {screened - passed_count}")
print(f"  Buy signals    : {len(buy_signals)}")
print(f"  Sell signals   : {len(sell_signals)}")
if buy_signals:
    print(f"  Top BUY        : {buy_signals[0]['ticker']} at {buy_signals[0]['confidence']:.1%}")
    print(f"  Lowest BUY     : {buy_signals[-1]['ticker']} at {buy_signals[-1]['confidence']:.1%}")

print_portfolio()
print("\nDone. Bot will run again next trading day at 9:30 AM EST.")