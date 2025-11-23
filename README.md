# Reinforcement Learning–Based Daily Trading Decision System  
### (Paper-Trading Mode with Dynamic Position Sizing, Logging, and Daily Automation)

This project implements a reinforcement-learning (PPO)–based system that generates **one trading recommendation per day**, applies **dynamic position sizing**, and saves all results to CSV for later analysis.  
All decisions are **paper mode only** — no real trades are executed.

---

## 📁 Project Overview

### **Purpose**
- Generate daily trading actions using a trained RL model  
- Apply performance-based dynamic position sizing  
- Log all decisions in CSV format  
- Enable future analysis comparing **AI decisions vs. Human decisions**

This system is designed for research and experimentation.

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





