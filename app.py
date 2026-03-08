from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

def fetch_data(ticker, period_years=5):
    end = datetime.today()
    start = end - timedelta(days=period_years * 365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    return df

def compute_features(df):
    close = df["Close"].squeeze()
    log_returns = np.log(close / close.shift(1)).dropna()
    rolling_vol = log_returns.rolling(window=21).std().dropna()
    common_idx = log_returns.index.intersection(rolling_vol.index)
    log_returns = log_returns.loc[common_idx]
    rolling_vol = rolling_vol.loc[common_idx]
    features = np.column_stack([log_returns.values, rolling_vol.values])
    dates = common_idx
    prices = close.loc[common_idx].values
    return features, dates, prices, log_returns.values, rolling_vol.values

def label_regimes(model, states, n_states=3):
    state_means = model.means_[:, 0]
    sorted_states = np.argsort(state_means)
    label_map = {}
    labels_ordered = ["Bear", "Sideways", "Bull"]
    for rank, state_id in enumerate(sorted_states):
        label_map[state_id] = labels_ordered[rank]
    regime_labels = [label_map[s] for s in states]
    return regime_labels, label_map

def run_hmm(ticker, n_components=3, n_iter=1000):
    df = fetch_data(ticker)
    if df.empty or len(df) < 100:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the symbol and try again.")
    features, dates, prices, log_returns, volatility = compute_features(df)

    # Normalize features to prevent covariance issues
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=42
    )
    model.fit(features)

    states = model.predict(features)
    regime_labels, label_map = label_regimes(model, states, n_components)
    log_prob = model.score(features)
    transmat = model.transmat_.tolist()

    state_stats = {}
    for state_id, label in label_map.items():
        mask = states == state_id
        state_stats[label] = {
            "mean_return": float(np.mean(log_returns[mask])) * 100,
            "mean_vol": float(np.mean(volatility[mask])) * 100,
            "count": int(np.sum(mask)),
            "pct": float(np.sum(mask)) / len(states) * 100,
            "color": {"Bull": "#00d4a8", "Sideways": "#f5c842", "Bear": "#ff4d6d"}[label]
        }

    result_dates = [d.strftime("%Y-%m-%d") for d in dates]
    result_prices = prices.tolist()
    result_regimes = regime_labels
    result_log_returns = (log_returns * 100).tolist()
    result_volatility = (volatility * 100).tolist()

    regime_changes = []
    current = result_regimes[0]
    start_idx = 0
    for i in range(1, len(result_regimes)):
        if result_regimes[i] != current:
            regime_changes.append({
                "start": result_dates[start_idx],
                "end": result_dates[i - 1],
                "regime": current,
                "color": state_stats[current]["color"]
            })
            current = result_regimes[i]
            start_idx = i
    regime_changes.append({
        "start": result_dates[start_idx],
        "end": result_dates[-1],
        "regime": current,
        "color": state_stats[current]["color"]
    })

    return {
        "ticker": ticker.upper(),
        "dates": result_dates,
        "prices": result_prices,
        "regimes": result_regimes,
        "log_returns": result_log_returns,
        "volatility": result_volatility,
        "state_stats": state_stats,
        "log_prob": float(log_prob),
        "transmat": transmat,
        "regime_changes": regime_changes,
        "n_states": n_components,
        "data_points": len(result_dates)
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    ticker = data.get("ticker", "SPY").strip().upper()
    try:
        result = run_hmm(ticker)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)