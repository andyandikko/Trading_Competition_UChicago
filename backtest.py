import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from trading_competition.quintile_prediction import initialize_forecaster, FeatureEngine 
# Load environment variables
load_dotenv(dotenv_path='./trading_competition/.env')
ALPACA_API_KEY_ID = os.getenv('ALPACA_API_KEY_ID')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_API_BASE_URL')

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Data management for backtesting"""
    def __init__(self, data_dir: str = './trading_competition/data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.alpaca = tradeapi.REST(
            key_id=ALPACA_API_KEY_ID,
            secret_key=ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL
        )
    
    def fetch_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes"""
        weekly_data = {}
        daily_data = {}
        
        for symbol in symbols:
            try:
                # Fetch weekly data
                weekly_bars = self.alpaca.get_bars(
                    symbol, 
                    timeframe='1Week',
                    start=start_date,
                    end=end_date
                ).df
                weekly_data[symbol] = weekly_bars
                
                # Fetch daily data
                daily_bars = self.alpaca.get_bars(
                    symbol,
                    timeframe='1Day',
                    start=start_date,
                    end=end_date
                ).df
                daily_data[symbol] = daily_bars
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        return weekly_data, daily_data

class EnhancedBacktester:
    """Enhanced backtester with rolling predictions"""
    def __init__(self, weekly_data: Dict[str, pd.DataFrame], daily_data: Dict[str, pd.DataFrame]):
        self.weekly_data = weekly_data
        self.daily_data = daily_data
        self.forecaster = initialize_forecaster()

    def get_predictions_for_window(self, window_end: pd.Timestamp, window_start: pd.Timestamp) -> pd.DataFrame:
        """Generate predictions using data up to window end."""
        logger.info(f"\nGenerating predictions for window {window_start} to {window_end}")
        
        # Process data for prediction with proper date range
        window_data = {}
        for ticker, df in self.weekly_data.items():
            mask = (df.index >= window_start) & (df.index <= window_end)
            window_df = df[mask].copy()
            if len(window_df) > 0:
                window_data[ticker] = window_df
        
        if not window_data:
            logger.error("No data available for window")
            return pd.DataFrame()
        
        # Process features and generate predictions
        feature_engine = FeatureEngine()
        processed_data = feature_engine.process_data_with_shifts(window_data, shifts=[0])
        
        if processed_data.empty:
            logger.error("No data available for prediction after processing")
            return pd.DataFrame()
        
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        try:
            # Train model and get predictions
            train_loader, val_loader, test_loader = self.forecaster.prepare_data(processed_data)
            self.forecaster.train(train_loader, val_loader)
            predictions = self.forecaster.predict_filtered(self.forecaster.meta_model, test_loader)

            # Ensure predictions are properly aggregated
            if 'Probabilities' in predictions.columns:
                aggregated_predictions = self.forecaster.aggregate_predictions(predictions)
                aggregated_predictions['PredictedQuintile'] = aggregated_predictions['FinalQuintile']
                return aggregated_predictions

            logger.error("PredictedQuintile column could not be computed.")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in prediction generation: {e}")
            return pd.DataFrame()



    def calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate MACD indicators for the entire price series
        Returns DataFrame with MACD, signal line, and trading positions
        """
        # Calculate EMAs
        fast_ema = prices.ewm(span=12, adjust=False).mean()
        slow_ema = prices.ewm(span=26, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return pd.DataFrame({
            'macd': macd,
            'signal': signal,
            'position': np.where(macd > signal, 1, -1)
        })

    def calculate_returns(self, df: pd.DataFrame, positions: pd.Series) -> pd.Series:
        """Calculate strategy returns using close prices and positions"""
        returns = df['close'].pct_change()
        strategy_returns = positions.shift(1) * returns
        return strategy_returns.fillna(0)

    def run_strategy_backtest(self, 
                          window_start: pd.Timestamp, 
                          window_end: pd.Timestamp,
                          predictions_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"\nRunning backtest for window: {window_start} to {window_end}")
        
        # Weekly Data Strategies
        eq_returns = pd.DataFrame({
            ticker: df['close'].loc[(df.index >= window_start) & (df.index <= window_end)].pct_change()
            for ticker, df in self.weekly_data.items()
            if not df.empty
        }).mean(axis=1)

        # Quintile Strategy
        MIN_STOCKS_PER_QUINTILE = 30
        top_tickers = predictions_df[
            predictions_df['PredictedQuintile'] == predictions_df['PredictedQuintile'].max()
        ]['Ticker'].tolist()

        if len(top_tickers) < MIN_STOCKS_PER_QUINTILE:
            # print(predictions_df.head(10))
            logger.warning(f"Insufficient stocks in top quintile ({len(top_tickers)}). Using fallback.")
            sorted_predictions = predictions_df.sort_values('PredictedQuintile', ascending=False)
            top_tickers = sorted_predictions.head(MIN_STOCKS_PER_QUINTILE)['Ticker'].tolist()

        quintile_returns = pd.DataFrame({
            ticker: self.weekly_data[ticker]['close'].loc[(self.weekly_data[ticker].index >= window_start) & 
                                                        (self.weekly_data[ticker].index <= window_end)].pct_change()
            for ticker in top_tickers
            if ticker in self.weekly_data
        }).mean(axis=1)

        # MACD Strategies
        all_daily_returns = []
        top_quintile_daily_returns = []
        
        # All stocks MACD
        for ticker, df in self.daily_data.items():
            if not df.empty:
                # Calculate MACD for full history
                full_macd = self.calculate_macd(df['close'])
                
                # Get window data and corresponding MACD signals
                window_data = df.loc[(df.index >= window_start) & (df.index <= window_end)]
                if not window_data.empty:
                    window_signals = full_macd.loc[window_data.index]
                    returns = window_data['close'].pct_change() * window_signals['position'].shift(1)
                    all_daily_returns.append(returns)

        # Top quintile MACD
        for ticker in top_tickers:
            if ticker in self.daily_data and not self.daily_data[ticker].empty:
                df = self.daily_data[ticker]
                full_macd = self.calculate_macd(df['close'])
                
                window_data = df.loc[(df.index >= window_start) & (df.index <= window_end)]
                if not window_data.empty:
                    window_signals = full_macd.loc[window_data.index]
                    returns = window_data['close'].pct_change() * window_signals['position'].shift(1)
                    top_quintile_daily_returns.append(returns)

        # Average daily returns across stocks
        if all_daily_returns:
            # Average across stocks and handle NaNs
            macd_all = pd.concat(all_daily_returns, axis=1).mean(axis=1).apply(lambda x: 0 if np.isnan(x) else x)
            # Calculate weekly return using compound returns
            macd_weekly_return = (1 + macd_all).prod() - 1  # This gives us the total return for the week
            
            logger.info(f"MACD All strategy using {len(all_daily_returns)} stocks with return: {macd_weekly_return}")
        else:
            macd_weekly_return = 0
            logger.warning("No stocks available for MACD All strategy")

        if top_quintile_daily_returns:
            # Average across stocks and handle NaNs
            macd_quintile = pd.concat(top_quintile_daily_returns, axis=1).mean(axis=1).apply(lambda x: 0 if np.isnan(x) else x)
            # Calculate weekly return using compound returns
            macd_quintile_weekly_return = (1 + macd_quintile).prod() - 1  # This gives us the total return for the week
            
            logger.info(f"MACD Top Quintile strategy using {len(top_quintile_daily_returns)} stocks with return: {macd_quintile_weekly_return}")
        else:
            macd_quintile_weekly_return = 0
            logger.warning("No stocks available for MACD Top Quintile strategy")

        # Create results DataFrame
        results = pd.DataFrame({
            'Equal Weight': eq_returns.iloc[-1],
            'Top Quintile': quintile_returns.iloc[-1],
            'MACD All': macd_weekly_return,
            'MACD Top Quintile': macd_quintile_weekly_return
        }, index=[window_end])

        # Remove duplicate indices and keep only the window end date
        results = results[~results.index.duplicated(keep='last')]
        results = results.loc[[window_end]]

        logger.info("\nFinal Strategy Statistics:")
        logger.info(f"Equal Weight: {len(self.weekly_data)} stocks")
        logger.info(f"Top Quintile: {len(top_tickers)} stocks")
        logger.info(f"MACD All: {len(all_daily_returns)} stocks")
        logger.info(f"MACD Top Quintile: {len(top_quintile_daily_returns)} stocks")

        return results

        

    def rolling_backtest(self, window_size: int = 1, num_predictions: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform rolling window backtest"""
        logger.info("\nStarting rolling backtest...")
        all_returns = []
        
        # Get sorted unique dates across all stocks
        dates = sorted(list(set([date for data in self.weekly_data.values() for date in data.index])))
        
        if len(dates) < num_predictions + window_size:
            raise ValueError(f"Insufficient data for {num_predictions} predictions")
        
        # Only test the last num_predictions periods
        start_idx = len(dates) - num_predictions - window_size
        test_dates = dates[start_idx:]
        
        logger.info(f"Backtest period: {test_dates[0]} to {test_dates[-1]}")
        logger.info(f"Number of test windows: {num_predictions}")
        
        for i in range(len(test_dates) - window_size):
            window_start = test_dates[i]
            window_end = test_dates[i + window_size]
            
            logger.info(f"\nProcessing window {i+1}/{num_predictions}")
            logger.info(f"Window period: {window_start} to {window_end}")
            
            try:
                # Generate predictions
                predictions = self.get_predictions_for_window(window_end, window_start)
                print(predictions.head(10))
                if predictions.empty:
                    logger.warning(f"Skipping window due to prediction failure")
                    continue
                
                # Run strategies
                window_returns = self.run_strategy_backtest(
                    window_start,
                    window_end,
                    predictions
                )
                
                if not window_returns.empty:
                    all_returns.append(window_returns)
                    logger.info(f"Successfully processed window {i+1}")
                    
            except Exception as e:
                logger.error(f"Error in window {window_start} to {window_end}: {e}")
                continue
        
        if not all_returns:
            raise ValueError("No valid results generated in backtest")
        
        # Combine all results
        combined_returns = pd.concat(all_returns)
        combined_returns = combined_returns.sort_index()
        
        # Calculate metrics
        metrics = self.calculate_metrics(combined_returns)
        
        logger.info("\nBacktest completed successfully")
        logger.info(f"Total periods processed: {len(combined_returns)}")
        
        return combined_returns, metrics

    def calculate_metrics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance metrics"""
        metrics = {}
        
        for column in returns.columns:
            series = returns[column].fillna(0)
            
            # Calculate metrics with error handling
            try:
                metrics[column] = {
                    'Total Return': (1 + series).prod() - 1,
                    'Annualized Return': series.mean() * 52,  # Annualize weekly returns
                    'Annualized Vol': series.std() * np.sqrt(52),
                    'Sharpe Ratio': (series.mean() / series.std() * np.sqrt(52)) if series.std() != 0 else 0,
                    'Max Drawdown': (series.cumsum() - series.cumsum().expanding().max()).min(),
                    'Win Rate': (series > 0).mean(),
                    'Periods': len(series)
                }
            except Exception as e:
                logger.error(f"Error calculating metrics for {column}: {e}")
                continue
        
        return pd.DataFrame(metrics)

def plot_results(returns: pd.DataFrame, metrics: pd.DataFrame, save_path: str):
    """Plot cumulative returns and strategy performance metrics."""
    # Set up figure with clear allocation for graph and table
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(16, 14))
    
    # Cumulative returns plot (top 70% of figure)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    (1 + returns).cumprod().plot(ax=ax1, linewidth=2)
    ax1.set_title('Cumulative Strategy Returns (Last 10 Weeks)', fontsize=16, pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Metrics table (bottom 30% of figure)
    ax2 = plt.subplot2grid((4, 1), (3, 0))
    ax2.axis('off')

    # Format metrics for table
    metrics_styled = metrics.round(4)
    table = ax2.table(
        cellText=metrics_styled.values,
        rowLabels=metrics_styled.index,
        colLabels=metrics_styled.columns,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.0, 0.8, 1.0]  # Adjust table size and position
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)  # Adjust cell size for readability

    # Add bold header and highlight
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#E6E6E6')

    ax2.set_title('Strategy Performance Metrics', fontsize=14, pad=10)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for table and title
    plt.subplots_adjust(hspace=0.4)  # Add space between subplots

    # Save plot
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
def main():
    try:
        logger.info("Starting backtest execution...")
        
        # Setup directories
        Path('./trading_competition/data').mkdir(parents=True, exist_ok=True)
        Path('./trading_competition/results').mkdir(parents=True, exist_ok=True)
        
        # Load universe
        universe = pd.read_csv('./trading_competition/results/next_week_predictions.csv')
        symbols = universe['Ticker'].tolist()
        # Initialize managers
        data_manager = DataManager()
        
        # Fetch data
        logger.info("Fetching data...")
        weekly_data, daily_data = data_manager.fetch_data(
            symbols=symbols,
            start_date='2018-01-01',
            end_date='2024-11-30'
        )
        
        # Initialize and run backtester
        logger.info("Running backtest...")
        backtester = EnhancedBacktester(weekly_data, daily_data)
        returns, metrics = backtester.rolling_backtest(window_size=1, num_predictions=20)
        
        plot_results(
            returns=returns,
            metrics=metrics,
            save_path='./trading_competition/results/backtest_results.png'
        )
        
        # Save results
        returns.to_csv('./trading_competition/results/strategy_returns.csv')
        metrics.to_csv('./trading_competition/results/strategy_metrics.csv')
        
        # Print summary
        print("\nStrategy Performance Metrics:")
        print(metrics.round(4))
        
        print("\nBacktest Period:")
        print(f"Start: {returns.index[0]}")
        print(f"End: {returns.index[-1]}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
    
    