# 📈 Market Regime Detection — Hidden Markov Model

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

## Run Locally

```bash
# 1. Clone / unzip the project
cd hmm-regime-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py

# 4. Open http://localhost:5000
```

---

## Deploy to Render (FREE — ~15 min)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial HMM regime detector"
# Create a new repo at github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/hmm-regime-detector.git
git push -u origin main
```

### Step 2 — Deploy on Render
1. Go to **https://render.com** → Sign up / Log in
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. Wait ~3-5 min for build → your app is LIVE at `https://hmm-regime-detector.onrender.com`

> **Free tier note**: Render free tier spins down after 15 min of inactivity. First request after sleep takes ~30s.

---

## Deploy to Railway (alternative, also FREE)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login & deploy
railway login
railway init
railway up
```

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
