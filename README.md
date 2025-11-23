Reinforcement Learning–Based Daily Trading Decision System
(Paper-Trading Mode with Position Sizing, Logging, and Daily Automation)

This project implements a reinforcement-learning (PPO)–based system that generates one trading decision per day and logs all results for later quantitative analysis.

The system does not execute any real trades.
It only computes daily recommended actions and saves them for research.

📁 Project Overview
1. Purpose

Generate daily trading decisions (BUY / SELL / HOLD) using a trained RL model

Integrate dynamic position sizing based on recent performance (rolling win rate)

Log all results to daily CSV files for further analysis

Provide a foundation for comparing RL-based decisions with human trades

📚 Key Features
✔ 1) Decision-making using a trained PPO model

Loads a pre-trained policy from models/.../model.zip

Uses a fixed 32-dimensional observation vector

Computes one action per day

✔ 2) Daily automatic execution

Script: live_trading_googl_once.py

Designed to run once per day, not continuously

Fetches the latest market data

Builds the observation

Runs the RL model

Runs position sizing

Produces the final recommended target position

Saves all results in a daily log

Actions:

0 → HOLD

1 → BUY

2 → SELL

✔ 3) Dynamic Position Sizing

Module: WinRatioModulatedSizer

This component calculates:

Maximum recommended position size (max_position_size)

Dynamic risk fraction (based on a sigmoid of rolling win rate)

Dynamic leverage

Target position based on RL output + risk model

This makes the RL output more realistic by controlling risk exposure.

✔ 4) Daily CSV Logging

Every daily run generates a CSV file:

Format example:

logs/live_trading_2025-11-22.csv


Each row contains:

UTC timestamp

Symbol

Mode (paper)

Price

Action (HOLD / BUY / SELL)

Position before

Position after

Trade size

Equity before

Equity after

Max position size

Leverage

Rolling win rate

Dynamic risk fraction

These logs can later be used for analysis and visualization.

✔ 5) Framework for Human vs AI Decision Comparison

This system produces:

AI’s target positions

AI’s recommended trades

Daily equity curves

RL risk parameters

A human trader can log their own trades.
Later both datasets can be compared:

Differences in timing

Differences in number of shares

Divergence in risk exposure

Performance comparison

Behavioral differences

This provides a strong foundation for research on human vs AI decision-making.

📂 Project Structure
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

🚀 Automating Daily Execution on macOS
Step 1 — Create an Automator Application

Open Automator

Choose Application

Insert a Run Shell Script action

Add:

cd /Users/yourname/Documents/rl-trading-googl
source .venv/bin/activate
python -m src.trading_rl.live_trading_googl_once


Save as:
RL_Daily_Trading.app

Step 2 — Use Calendar to run it daily

Open Calendar

Create an event at your chosen time

Click Alert → Custom → Open file

Choose the Automator app

Now the system runs automatically at that time each day (as long as your Mac is awake).

📈 Example Output
===== RL Daily Decision Report =====
Time (UTC):        2025-11-22 03:29:07
Symbol:            GOOGL
Current Price:     299.66 USD
Current Position:  34 shares
Current Equity:    10304.03 USD

RL Action:         BUY (raw=1)
Sizer max_pos:     5 shares (leverage=1.71)
Target Position:   34 shares
Trade Size:        +0 shares

Equity Before/After: 10304.03 → 10304.03
rolling_win_ratio = 0.500
dynamic_risk_fraction = 0.0265
====================================

🔍 Research Possibilities

With the daily logs, you can analyze:

Tracking human vs AI position curves

Daily PnL comparison

Risk exposure differences

Trade frequency differences

Equity curve visualization

Rolling statistics (Sharpe, drawdown, leverage usage)

🛠 Dependencies
pip install stable-baselines3 gymnasium yfinance numpy pandas matplotlib joblib

📌 Notes

This project is for research and paper-trading simulation only.

No real orders are executed.

The system has been designed to run once per day and record all decisions.
