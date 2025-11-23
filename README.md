
# Advanced Position Sizing & Dynamic Leveraging for Algorithmic Trading

> A research-oriented project that combines reinforcement learning with
> advanced capital allocation models (position sizing & dynamic leverage)
> to maximize long-term growth while controlling drawdowns.

---

## 1. Project Overview

This repository implements a **research-oriented algorithmic trading system** that focuses on:

- Separating **signal quality (expectancy)** from **capital allocation (position sizing & leverage)**  
- Designing **dynamic, path-dependent position sizing** rules based on recent performance  
- Integrating **dynamic leverage** as a controlled risk amplifier  
- Providing a framework to compare **AI decisions vs human decisions** over time

The system is not intended for live real-money trading.  
It is designed for **simulation, academic research, and strategy analysis**.

---

## 2. Motivation: Capital Allocation Matters

Most traders focus on *what* and *when* to trade.  
This project focuses on **how much to trade each time**, under the assumption that:

- You already have a strategy with **positive expectancy**  
  (e.g. win rate ~30%, payoff ratio ~3:1)
- The key question becomes:
  > Given a positive edge, how should we size positions and apply leverage  
  > to maximize CAGR while keeping Maximum Drawdown (MDD) under control?

In other words:

- **Expectancy** answers: “Is this strategy statistically profitable?”  
- **Capital allocation** answers: “How aggressively should we bet on it?”

Even with the same signals, different position sizing rules can lead to  
dramatically different equity curves and risk profiles.

---

## 3. Classical Models: A Critical Baseline

Before introducing dynamic models, the project examines classical, **static** bet sizing frameworks:

### 3.1 Fixed Fractional Sizing

- Risk a fixed percentage of current equity per trade (e.g. 2%)
- Position size formula (conceptually):

**Position Size Formula**

Position Size = (Equity × Risk Fraction) / (Entry Price – Stop Loss)



- Pros:
  - Built-in drawdown protection (risk shrinks as equity shrinks)
- Cons:
  - **Path-agnostic**: does not care about recent performance or regime changes
  - Treats all signals equally, even if the recent environment has changed

### 3.2 Kelly Criterion

f* = W − (1 − W) / R

- \( W \): win probability  
- \( R \): payoff ratio (average win / average loss)

Example: with \( W = 0.3 \), \( R = 3 \):

f* = 0.3 − (0.7 / 3)
f* ≈ 0.067  → about 6.7% of equity


- Pros:
  - Theoretically maximizes long-term geometric growth
- Cons:
  - Highly sensitive to estimation errors in \( W \) and \( R \)
  - Assumes ideal conditions (no costs, no slippage, stable parameters)
  - In practice, **full Kelly is often dangerously aggressive**

Partial Kelly and other variants still inherit this **sensitivity** problem.

Static models are useful as benchmarks, but they ignore **path-dependency** and
the evolving relationship between strategy and market environment.

---

## 4. Evolving to Dynamic, Performance-Based Sizing

To overcome the limitations of static models, this project uses **dynamic,
path-dependent** position sizing based on recent performance.

### 4.1 Why Path-Dependency?

The **equity curve itself** contains information about how well the strategy
is currently aligned with the market.

- Smoothly rising equity → strategy is in sync → reasonable to scale up risk  
- Choppy or falling equity → possible regime change or degradation → scale down risk

Static models ignore this information; dynamic models explicitly use it.

### 4.2 Choosing a Feedback Signal

Several candidates for the feedback signal:

- **Rolling Win Ratio**
  - Intuitive and easy to compute
  - But ignores the *magnitude* of wins vs losses

- **Rolling Profit Factor**
  - Total gross profit / total gross loss
  - Includes both frequency and size of outcomes
  - Can be distorted by a single large outlier win

- **Equity curve slope / Sharpe-like metrics**
  - Most theoretically complete (risk-adjusted return)
  - More complex & sensitive to parameter choices

In this project, we focus on:

> **Rolling Win Ratio** as the main feedback signal,  
> under the assumption of a reasonably stable payoff ratio.

We then **map** this rolling win ratio into a risk multiplier using a carefully
designed function.

---

## 5. Core Model: Win-Ratio Modulated Dynamic Sizing & Leverage

The central idea:

1. Define a conservative **base risk fraction** derived from long-term statistics  
   (e.g. a fraction of Kelly, like 0.25 × Kelly)  
2. Modulate this base risk using a function of **Rolling Win Ratio**  
3. Apply **additional non-linear scaling** when using leverage

---

## 📚 Key Features

### ✔ 1. Daily Trading Decision (Using PPO Model)
The system loads the trained PPO model stored under:


models/.../model.zip
Daily execution includes:
1. Fetching the latest market data  
2. Constructing a fixed 32-dimensional observation  
3. Predicting an action from the RL agent  
   - `0 = HOLD`  
   - `1 = BUY`  
   - `2 = SELL`  
4. Outputting the final recommended position

Only one decision per day.

---

### ✔ 2. Dynamic Position Sizing  
Implemented using the **WinRatioModulatedSizer**:
- Uses rolling win ratio  
- Sigmoid-based risk scaling  
- Dynamic leverage adjustments  
- Outputs:
  - maximum recommended position size  
  - dynamic risk fraction  
  - leverage multiplier  

This stabilizes risk and provides more realistic trading recommendations.

---

### ✔ 3. Daily CSV Logging

Every execution produces:

Each row logs:
- timestamp  
- symbol  
- current price  
- action (buy/sell/hold)  
- position before/after  
- trade size  
- equity before/after  
- max position  
- leverage  
- rolling win ratio  
- dynamic risk fraction  

This makes post-analysis simple and reliable.

---

### ✔ 4. AI vs. Human Decision Comparison

Because the system logs all AI-generated decisions,  
a human trader can also record their daily actions.

Later comparisons may include:
- Differences in action choice  
- PnL comparison  
- Position sizing differences  
- Equity curve comparison  
- Behavioral patterns  

This project is suitable for quantitative behavior analysis.

---

## 📂 Project Structure
```bash
rl-trading-googl/
│
├── models/
│   └── .../model.zip
│
├── logs/
│   └── live_trading_YYYY-MM-DD.csv
│
├── src/
│   └── trading_rl/
│       ├── live_trading_googl_once.py     # Daily execution script
│       ├── position_sizing.py             # Dynamic position sizing logic
│       ├── main_train_ppo.py              # PPO training script
│       └── live_trading_googl.py          # Real-time loop (legacy)
│
└── README.md
```


---

## 🚀 Daily Automation (macOS)

### **1. Create an Automator App**

1. Open **Automator**  
2. Choose **Application**  
3. Add **Run Shell Script**  
4. Insert:

```bash
cd /Users/yourname/Documents/rl-trading-googl
source .venv/bin/activate
python -m src.trading_rl.live_trading_googl_once
```

📊 Data Analysis

To analyze logged data:

```bash
import pandas as pd

df = pd.read_csv("logs/live_trading_2025-11-22.csv")
print(df.head())
```





