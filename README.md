# Trading Competition Meta-Learning Project

## Overview
This 2-day long development project implements an innovative trading strategy that secured 8th place in a Masters-level trading competition at UChicago. By combining meta-learning techniques with traditional quantitative analysis, the system achieved top performance among low-frequency equities trading strategies during live paper trading.

The strategy builds upon learnings from the M6 Financial Forecasting Competition, incorporating:

Advanced meta-learning architecture for stock prediction
Comprehensive feature engineering pipeline
Multi-timeframe analysis framework
Dynamic portfolio allocation strategy
Robust backtesting methodology

The system processes historical data from Alpaca Markets to train a neural network model that predicts weekly stock quintiles, enabling data-driven portfolio allocation decisions. During the one-week competition period, the strategy demonstrated superior performance in live market conditions, particularly excelling in handling market volatility and regime changes, though losing out to competitors trading cryptocurrencies (BTC hit 100,000 that week).

## Key Features
- Meta-learning based neural network for stock prediction
- Extensive feature engineering including technical indicators
- Multi-timeframe analysis (daily and weekly data)
- Rolling window backtesting framework
- Dynamic portfolio allocation strategy
- Performance comparison with benchmark strategies

## Project Structure
```
trading_competition/
├─ data/
│  ├─ raw/                # Raw data fetched from Alpaca
│  ├─ processed_data.csv  # Processed features for training
│  └─ stock_universe.csv  # Universe of stocks used
├─ models/
│  └─ model.pth          # Saved trained model parameters
├─ results/
│  ├─ next_week_predictions.csv  # Quintile predictions
│  ├─ strategy_returns.csv       # Backtest returns
│  ├─ strategy_metrics.csv       # Performance metrics
│  └─ backtest_results.png      # Performance visualization
├─ quintile.py           # Model training and prediction
├─ backtest.py           # Backtesting framework
├─ allocation.py         # Portfolio allocation logic
├─ requirements.txt      # Python dependencies
└─ README.md            # This file
```

## Core Components

### 1. Meta-Learning Model (`quintile.py`)
The model architecture consists of:
- Base neural network for feature processing
- Meta-learning layer for stock-specific adjustments
- Feature engineering pipeline including:
  - Technical indicators (MACD, RSI, Bollinger Bands)
  - Price momentum features
  - Volatility metrics
  - Volume indicators

```python
# Example: Initialize and train the model
forecaster = initialize_forecaster()
train_loader, val_loader, _ = forecaster.prepare_data(data)
forecaster.train(train_loader, val_loader)

# Generate predictions
predictions = forecaster.predict_filtered(forecaster.meta_model, test_loader)
aggregated_predictions = forecaster.aggregate_predictions(predictions)
```

### 2. Backtesting Framework (`backtest.py`)
Features:
- Rolling window backtest implementation
- Multiple strategy comparison:
  - Equal Weight portfolio
  - Top Quintile selection
  - MACD-based strategies
- Performance metrics calculation
- Visualization of results

```python
# Example: Run backtest
backtester = EnhancedBacktester(weekly_data, daily_data)
returns, metrics = backtester.rolling_backtest(
    window_size=1,
    num_predictions=20
)
```

### 3. Portfolio Allocation (`allocation.py`)
Implements a two-tier allocation strategy:
- 70% allocation to top quintile stocks
- 30% allocation to second quintile stocks
- Intelligent reallocation of remaining balance

```python
# Example: Allocate funds
remaining_balance, data = allocate_funds(data, 70000, top_quintile)
remaining_balance += allocate_funds(data, 30000, second_quintile)[0]
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading_competition.git
cd trading_competition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Alpaca API credentials:
Create a `.env` file in the project root:
```env
ALPACA_API_BASE_URL="https://paper-api.alpaca.markets"
ALPACA_API_KEY_ID="YOUR_API_KEY"
ALPACA_SECRET_KEY="YOUR_SECRET_KEY"
```

## Usage Guide

### 1. Data Preparation and Model Training
```bash
python quintile.py
```
This will:
- Fetch historical data from Alpaca
- Process features and train the model
- Generate predictions for the next week
- Save results in `results/next_week_predictions.csv`

### 2. Run Backtesting
```bash
python backtest.py
```
This will:
- Perform rolling window backtests
- Generate performance metrics
- Create visualization plots
- Save results in `results/`

### 3. Execute Trade Allocation
```bash
python allocation.py
```
This will:
- Read the latest predictions
- Fetch current market prices
- Place orders according to the allocation strategy

## Performance Metrics
Based on backtesting from July 2024 to November 2024:

| Metric | Equal Weight | Top Quintile | MACD All | MACD Top Quintile |
|--------|-------------|--------------|-----------|-------------------|
| Total Return | 7.88% | 3.39% | 0.80% | -7.29% |
| Annualized Return | 20.55% | 9.43% | 2.59% | -18.54% |
| Sharpe Ratio | 1.593 | 0.744 | 0.250 | -1.233 |
| Max Drawdown | -2.83% | -3.77% | -5.95% | -9.58% |

## Requirements
- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- alpaca-trade-api
- matplotlib
- python-dotenv

## Disclaimer
This project is for educational and research purposes only. Past performance does not guarantee future results. Always perform your own due diligence before making investment decisions.

## License
MIT License

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.
