import os
import torch
from torch import nn
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas_ta as ta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""
    input_size: int
    hidden_sizes: List[int]
    output_size: int = 5  # Number of quintiles
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 50
    mesa_dim: int = 1
    dropout_rate: float = 0.2
    shifts: List[int] = None  # Will be set in post_init

    def __post_init__(self):
        if self.shifts is None:
            self.shifts = [0, 7, 14, 21]  # Default shifts in days

class FinancialDataset(Dataset):
    """Custom Dataset for financial time series data"""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, tickers: torch.Tensor):
        self.features = features
        self.labels = labels
        self.tickers = tickers

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.tickers[idx]

class FeatureEngine:
    """Enhanced feature engineering for financial data"""
    def __init__(self):
        self.feature_scalers = {}
        self.imputer = SimpleImputer(strategy='mean')
    
    def realized_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Compute realized volatility over a rolling window.
        
        Args:
            returns (pd.Series): Daily returns (e.g., percentage change).
            window (int): Rolling window size in days.
        
        Returns:
            pd.Series: Realized volatility.
        """
        squared_returns = returns**2
        rolling_sum = squared_returns.rolling(window=window).sum()
        realized_vol = np.sqrt(rolling_sum)
        return realized_vol
    
    def compute_features(self, df: pd.DataFrame, ticker: str, fit: bool = True) -> pd.DataFrame:
        """Compute comprehensive set of features including realized volatility."""
        features = pd.DataFrame()
        
        # Basic price features
        features['return'] = df['Adj Close'].pct_change()
        
        # Realized volatility (20-day rolling window)
        features['realized_vol_20d'] = self.realized_volatility(features['return'], window=20)
        
        # Lagged volatility and returns
        for lag in range(1, 8):
            features[f'volatility_lag_{lag}'] = features['return'].rolling(window=lag).std()
            features[f'return_lag_{lag}'] = features['return'].shift(lag)
        
        # Volume features
        features['volume_change'] = df['Volume'].pct_change()
        features['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Price momentum
        for window in [5, 10, 20]:
            features[f'momentum_{window}d'] = df['Adj Close'].pct_change(window)
        
        # Moving averages and crosses
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = df['Adj Close'].rolling(window).mean()
            features[f'ema_{window}'] = ta.ema(df['Adj Close'], length=window)
            features[f'ma_ratio_{window}'] = df['Adj Close'] / features[f'sma_{window}']
        
        # Technical indicators
        features['rsi_14'] = ta.rsi(df['Adj Close'], length=14)
        macd = ta.macd(df['Adj Close'])
        bbands = ta.bbands(df['Adj Close'], length=20, std=2.0)
        features = pd.concat([features, macd, bbands], axis=1)
        
        # Handle infinite and NaN values
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(0, inplace=True)
        
        # Scale features
        if fit:
            self.feature_scalers[ticker] = StandardScaler()
            features_scaled = self.feature_scalers[ticker].fit_transform(features)
        else:
            if ticker not in self.feature_scalers:
                raise ValueError(f"No scaler found for ticker {ticker}")
            features_scaled = self.feature_scalers[ticker].transform(features)
        
        return pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
    
    def get_feature_count(self) -> int:
        """Calculate the number of features that will be generated."""
        count = 0
        count += 2  # return and realized_vol_20d
        count += 14  # lagged volatility and returns (7 each)
        count += 2  # volume features
        count += 3  # momentum features
        count += 12  # moving averages and ratios (4 windows * 3)
        count += 1  # RSI
        count += 3  # MACD
        count += 5  # Bollinger Bands
        return count

    def process_data_with_shifts(self, data: Dict[str, pd.DataFrame], shifts: List[int]) -> pd.DataFrame:
        """
        Process data with shifts and combine into a single DataFrame.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary of stock data by ticker.
            shifts (List[int]): List of shifts in days.
        
        Returns:
            pd.DataFrame: Combined dataset with all shifts.
        """
        all_data = []
        
        for shift in shifts:
            for ticker, df in data.items():
                df_copy = df.copy()
                features = self.compute_features(df_copy, ticker)
                
                # Add metadata
                features['Ticker'] = ticker
                features['Shift'] = shift
                features['Date'] = df_copy.index
                features['Return'] = df_copy['Adj Close'].pct_change()
                features['IsETF'] = 1 if ticker.upper() in ['SPY', 'QQQ', 'TLT', 'IWM', 'EFA'] else 0
                
                # Shift dates
                features.index = features.index + pd.Timedelta(days=shift)
                
                all_data.append(features)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.set_index('Date', inplace=True)
        combined_data.sort_index(inplace=True)
        
        return combined_data


class FinancialForecaster:
    """Main class for financial forecasting"""
    def __init__(self, config: ModelConfig = None):
        self.feature_engine = FeatureEngine()
        if config is None:
            config = ModelConfig(
                input_size=self.feature_engine.get_feature_count(),
                hidden_sizes=[32, 16],
                output_size=5,
                learning_rate=0.001,
                batch_size=256,
                epochs=50
            )
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, data: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training"""
        # Extract features and labels
        feature_cols = [col for col in data.columns if col not in ['Ticker', 'Shift', 'Return', 'IsETF']]
        X = torch.FloatTensor(data[feature_cols].values.copy())
        y = torch.LongTensor(self._compute_quintiles(data['Return']).copy())
        
        # Create and store ticker mapping
        unique_tickers = data['Ticker'].unique()
        self.ticker_to_idx = {ticker: idx for idx, ticker in enumerate(unique_tickers)}
        self.idx_to_ticker = {idx: ticker for ticker, idx in self.ticker_to_idx.items()}
        
        ticker_indices = [self.ticker_to_idx[ticker] for ticker in data['Ticker']]
        ticker_indices = torch.LongTensor(ticker_indices)
        
        # Create dataset
        dataset = FinancialDataset(X, y, ticker_indices)
        dataset.idx_to_ticker = self.idx_to_ticker  # Store mapping in dataset
        
        # Split data
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        return train_loader, val_loader, test_loader

    def _compute_quintiles(self, returns: pd.Series) -> np.ndarray:
        """Compute return quintiles"""
        returns = returns.fillna(0)
        quintiles = pd.qcut(returns, q=5, labels=False, duplicates='drop').fillna(2).astype(int).values
        return quintiles

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[nn.Module, nn.Module]:
        """Train base and meta models"""
        base_model = BaseModel(self.config).to(self.device)
        num_tasks = len(set(train_loader.dataset.dataset.tickers.tolist()))
        meta_model = MetaModel(base_model, num_tasks, self.config).to(self.device)
        criterion = nn.CrossEntropyLoss()
        base_optimizer = torch.optim.Adam(base_model.parameters(), lr=self.config.learning_rate)
        meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=self.config.learning_rate)
        best_val_loss = float('inf')
        for epoch in range(self.config.epochs):
            self._train_epoch(base_model, meta_model, train_loader, base_optimizer, meta_optimizer, criterion)
            val_loss = self._validate(meta_model, val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(meta_model.state_dict(), './trading_competition/results/best_meta_model.pt')
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs} - Validation Loss: {val_loss:.4f}")
        return base_model, meta_model

    def _train_epoch(self, base_model: nn.Module, meta_model: nn.Module,
                     train_loader: DataLoader, base_optimizer: torch.optim.Optimizer,
                     meta_optimizer: torch.optim.Optimizer, criterion: nn.Module):
        """Train for one epoch"""
        base_model.train()
        meta_model.train()
        for batch_X, batch_y, batch_tickers in train_loader:
            batch_X, batch_y, batch_tickers = batch_X.to(self.device), batch_y.to(self.device), batch_tickers.to(self.device)
            base_optimizer.zero_grad()
            base_outputs = base_model(batch_X)
            base_loss = criterion(base_outputs, batch_y)
            base_loss.backward()
            base_optimizer.step()
            meta_optimizer.zero_grad()
            meta_outputs = meta_model(batch_X, batch_tickers)
            meta_loss = criterion(meta_outputs, batch_y)
            meta_loss.backward()
            meta_optimizer.step()

    def _validate(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model"""
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y, batch_tickers in val_loader:
                batch_X, batch_y, batch_tickers = batch_X.to(self.device), batch_y.to(self.device), batch_tickers.to(self.device)
                outputs = model(batch_X, batch_tickers)
                total_loss += criterion(outputs, batch_y).item()
        return total_loss / len(val_loader)

    def predict_filtered(self, model: nn.Module, test_loader: DataLoader, shift: int = 7) -> pd.DataFrame:
        """Generate filtered predictions with actual ticker names"""
        model.eval()
        all_predictions = []
        
        # Get ticker mapping
        idx_to_ticker = test_loader.dataset.dataset.idx_to_ticker
        
        with torch.no_grad():
            for batch_X, batch_y, batch_tickers in test_loader:
                batch_X = batch_X.to(self.device)
                batch_tickers = batch_tickers.to(self.device)
                outputs = model(batch_X, batch_tickers)
                
                # Get predictions and convert to numpy
                pred_quintiles = outputs.argmax(dim=1).cpu().numpy()
                ticker_indices = batch_tickers.cpu().numpy()
                
                # Map indices to actual ticker names
                batch_tickers = [idx_to_ticker[int(idx)] for idx in ticker_indices]
                
                # Store batch results
                batch_df = pd.DataFrame({
                    'Ticker': batch_tickers,
                    'PredictedQuintile': pred_quintiles
                })
                all_predictions.append(batch_df)
        
        # Combine all predictions
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Aggregate predictions by ticker
        final_predictions = predictions_df.groupby('Ticker').agg({
            'PredictedQuintile': 'mean'
        }).reset_index()
        
        # Round predictions to nearest integer
        final_predictions['PredictedQuintile'] = final_predictions['PredictedQuintile'].round().astype(int)
        
        # Sort by ticker
        final_predictions = final_predictions.sort_values('Ticker')
        
        return final_predictions[['Ticker', 'PredictedQuintile']]

class BaseModel(nn.Module):
    """Base neural network model"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        layers = []
        prev_size = config.input_size
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, config.output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.network(x), dim=1)

class MetaModel(nn.Module):
    """Meta-learning model"""
    def __init__(self, base_model: BaseModel, num_tasks: int, config: ModelConfig):
        super().__init__()
        self.base_model = base_model
        self.mesa_params = nn.Parameter(torch.randn(num_tasks, config.mesa_dim))
        self.meta_layer = nn.Linear(config.mesa_dim, config.output_size, bias=False)

    def forward(self, x, task_indices):
        base_pred = self.base_model(x)
        task_params = self.mesa_params[task_indices]
        adjustments = self.meta_layer(task_params)
        return torch.softmax(base_pred + adjustments, dim=1)


class DataManager:
    """Handles data fetching and storage"""
    def __init__(self, data_dir: str = './trading_competition/data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_data(self, symbols: List[str], start_date: str, end_date: str, frequency: str = 'W') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols and save to CSV, resampling to the desired frequency.
        
        Args:
            symbols (List[str]): List of stock symbols.
            start_date (str): Start date for data fetching.
            end_date (str): End date for data fetching.
            frequency (str): Data frequency ('D' for daily, 'W' for weekly).
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock data by ticker.
        """
        data = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                csv_path = self.data_dir / f"{symbol}.csv"
                
                if csv_path.exists():
                    logger.info(f"Loading cached data for {symbol}")
                    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
                else:
                    logger.info(f"Downloading data for {symbol}")
                    df = yf.download(symbol, start=start_date, end=end_date)
                    if df.empty:
                        logger.warning(f"No data found for {symbol}")
                        failed_symbols.append(symbol)
                        continue
                    df.to_csv(csv_path)
                
                # Resample to weekly frequency
                df = df.resample(frequency).last()
                data[symbol] = df
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)
                continue
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for symbols: {failed_symbols}")
        
        logger.info(f"Successfully fetched data for {len(data)} symbols")
        return data


class ResultManager:
    """Handles saving and loading of results"""
    def __init__(self, result_dir: str = './trading_competition/results'):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
    def save_predictions(self, predictions: pd.DataFrame, filename: str = 'test_predictions.csv'):
        """Save predictions to CSV"""
        file_path = self.result_dir / filename
        predictions.to_csv(file_path, index=False)
        logger.info(f"Predictions saved to {file_path}")
        
    def save_model(self, model: nn.Module, filename: str):
        """Save model state"""
        file_path = self.result_dir / f"{filename}.pth"
        torch.save(model.state_dict(), file_path)
        logger.info(f"Model saved to {file_path}")
        
    def save_config(self, config: ModelConfig, filename: str = 'model_config.json'):
        """Save model configuration to JSON"""
        file_path = self.result_dir / filename
        with open(file_path, 'w') as f:
            json.dump(config.__dict__, f, indent=4)
        logger.info(f"Configuration saved to {file_path}")
        
    def load_model(self, model: nn.Module, filename: str) -> nn.Module:
        """Load model state"""
        file_path = self.result_dir / f"{filename}.pth"
        model.load_state_dict(torch.load(file_path))
        logger.info(f"Model loaded from {file_path}")
        return model


def main():
    data_manager = DataManager('./trading_competition/data')
    universe = pd.read_csv('./trading_competition/data/stock_universe_mini.csv')
    symbols = universe['symbol'].tolist()

    # Fetch and process data
    data = data_manager.fetch_data(symbols, '2018-01-01', '2023-12-31', frequency='W')
    feature_engine = FeatureEngine()
    processed_data = feature_engine.process_data_with_shifts(data, [0, 1, 2, 3, 4, 5, 6, 7])

    # Prepare and train models
    config = ModelConfig(input_size=feature_engine.get_feature_count(), hidden_sizes=[32, 16])
    forecaster = FinancialForecaster(config=config)
    train_loader, val_loader, test_loader = forecaster.prepare_data(processed_data)
    base_model, meta_model = forecaster.train(train_loader, val_loader)

    # Get predictions
    predictions = forecaster.predict_filtered(meta_model, test_loader, shift=7)

    # Save predictions
    result_manager = ResultManager('./trading_competition/results')
    result_manager.save_predictions(predictions, filename='next_week_predictions.csv')
    
    logger.info(f"Generated predictions for {len(predictions)} tickers")

if __name__ == "__main__":
    main()
