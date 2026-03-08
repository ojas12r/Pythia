#  Market Regime Detection — Hidden Markov Model

> Unsupervised Gaussian HMM for regime-switching detection in financial time-series data.

## Features

- **Gaussian HMM** with 3 hidden states: Bull / Sideways / Bear
- **Baum-Welch** algorithm for maximum-likelihood parameter estimation
- **Viterbi decoding** for most-probable hidden state sequence
- **Features**: log-returns + 21-day rolling volatility
- **Live data** via `yfinance` (5 years daily OHLCV)
- **Interactive charts**: price + regime overlay, log-returns, volatility
- **Transition matrix** visualization

---

## Project Structure

```
hmm-regime-detector/
├── app.py              # Flask backend + HMM logic
├── templates/
│   └── index.html      # Frontend (Plotly charts)
├── requirements.txt
├── Procfile            # For Render/Heroku
├── render.yaml         # Render config
└── README.md
```

---

## How It Works

### Feature Engineering
```python
log_returns = log(P_t / P_{t-1})          # Daily log returns
volatility  = rolling_std(log_returns, 21) # 21-day rolling σ
features    = [log_returns, volatility]    # 2D observation sequence
```

### Model Training (Baum-Welch)
```python
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(features)  # EM algorithm (Baum-Welch)
```

### State Inference (Viterbi)
```python
states = model.predict(features)  # Viterbi decoding
```

### Regime Labeling
States are sorted by mean log-return:
- Lowest mean return → **Bear**
- Middle mean return → **Sideways**  
- Highest mean return → **Bull**

---

## Resume Bullet Points
- Implemented unsupervised Gaussian HMM for regime-switching detection in financial time-series
- Trained model using Baum-Welch (EM) algorithm; inferred hidden states via Viterbi decoding
- Engineered log-return and 21-day rolling volatility features for probabilistic state modeling
- Built interactive Flask web app with Plotly visualizations; deployed on Render
