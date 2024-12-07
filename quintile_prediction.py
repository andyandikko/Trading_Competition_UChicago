import os
import torch
from torch import nn
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging
from pathlib import Path
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
            self.shifts = [0, 1, 2, 3, 4, 5, 6, 7]  # Default shifts

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
    """Feature engineering for financial data."""
    def __init__(self):
        self.feature_scalers = {}
        self.imputer = SimpleImputer(strategy='mean')

    def realized_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Compute realized volatility over a rolling window."""
        squared_returns = returns**2
        rolling_sum = squared_returns.rolling(window=window).sum()
        realized_vol = np.sqrt(rolling_sum)
        return realized_vol

    def calculate_rsi(self, series: pd.Series, length: int = 14) -> pd.Series:
        """Calculate the Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)
        avg_gain = gain.rolling(window=length).mean()
        avg_loss = loss.rolling(window=length).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9) -> pd.DataFrame:
        """Calculate the MACD indicator."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({'macd': macd, 'signal': signal_line})

    def calculate_bbands(self, series: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper_band = sma + num_std * rolling_std
        lower_band = sma - num_std * rolling_std
        return pd.DataFrame({'bb_upper': upper_band, 'bb_lower': lower_band, 'bb_width': upper_band - lower_band})

    def compute_features(self, df: pd.DataFrame, ticker: str, fit: bool = True) -> pd.DataFrame:
        """Compute features for a given DataFrame."""
        features = pd.DataFrame()
        
        # Basic price and volume features
        features['return'] = df['close'].pct_change()
        features['hlv'] = (df['high'] - df['low']) / df['volume']
        features['close_to_high'] = (df['high'] - df['close']) / df['high']
        features['close_to_low'] = (df['close'] - df['low']) / df['low']
        
        # Realized volatility
        features['realized_vol_20d'] = self.realized_volatility(features['return'], window=20)
        
        # Lagged features
        for lag in range(1, 8):
            features[f'volatility_lag_{lag}'] = features['return'].rolling(window=lag).std()
            features[f'return_lag_{lag}'] = features['return'].shift(lag)
            features[f'volume_lag_{lag}'] = df['volume'].pct_change().shift(lag)
        
        # Volume and price momentum
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        for window in [5, 10, 20]:
            features[f'momentum_{window}d'] = df['close'].pct_change(window)
            features[f'volume_momentum_{window}d'] = df['volume'].pct_change(window)
        
        # Moving averages and technical indicators
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = df['close'].rolling(window).mean()
            features[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            features[f'ma_ratio_{window}'] = df['close'] / features[f'sma_{window}']
        
        # RSI
        features['rsi_14'] = self.calculate_rsi(df['close'], length=14)
        
        # MACD
        macd_data = self.calculate_macd(df['close'])
        features['macd'] = macd_data['macd']
        features['macd_signal'] = macd_data['signal']
        features['macd_hist'] = macd_data['macd'] - macd_data['signal']
        
        # Bollinger Bands
        bbands_data = self.calculate_bbands(df['close'])
        features = pd.concat([features, bbands_data], axis=1)
        
        # Handle missing values
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features = features.ffill().bfill().fillna(0)
        
        # Scale features
        if fit:
            self.feature_scalers[ticker] = StandardScaler()
            features_scaled = self.feature_scalers[ticker].fit_transform(features)
        else:
            if ticker not in self.feature_scalers:
                raise ValueError(f"No scaler found for ticker {ticker}")
            features_scaled = self.feature_scalers[ticker].transform(features)
        
        return pd.DataFrame(features_scaled, index=features.index, columns=features.columns)

    def process_data_with_shifts(self, data: Dict[str, pd.DataFrame], shifts: List[int] = None) -> pd.DataFrame:
        """Process data with shifts and align targets."""
        if shifts is None:
            shifts = [0]
        
        all_data = []
        for shift in shifts:
            for ticker, df in data.items():
                df_copy = df.copy()
                features = self.compute_features(df_copy, ticker)
                
                shifted_features = features.shift(shift)
                shifted_features['Return'] = df_copy['close'].pct_change().shift(-2)
                shifted_features['Ticker'] = ticker
                shifted_features['Shift'] = shift
                shifted_features['Date'] = df_copy.index
                
                shifted_features.dropna(inplace=True)
                all_data.append(shifted_features)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.set_index('Date', inplace=True)
        combined_data.sort_index(inplace=True)
        
        return combined_data

    def get_feature_count(self) -> int:
        """Calculate the total number of features that will be generated."""
        count = 0

        # Basic price and volume features
        count += 4  # return, hlv, close_to_high, close_to_low

        # Realized volatility
        count += 1  # realized_vol_20d

        # Lagged features (7 lags each for volatility, return, and volume)
        count += 7  # volatility lags
        count += 7  # return lags
        count += 7  # volume lags

        # Volume and price momentum
        count += 2  # volume_change, volume_ma_ratio
        count += 6  # momentum and volume momentum for [5,10,20]

        # Moving averages and ratios (4 windows: 5,10,20,50)
        count += 12  # SMA, EMA, and MA ratio for each window

        # RSI
        count += 1  # rsi_14

        # MACD
        count += 3  # macd, macd_signal, macd_hist

        # Bollinger Bands
        count += 3  # bb_upper, bb_lower, bb_width

        logger.info(f"Total features calculated: {count}")
        return count



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
    """Meta-learning model for asset-specific adjustments"""
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
                epochs=10
            )
        self.config = config
        self.device = self.device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
        self.meta_model = None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training"""
        # Extract features and labels
        feature_cols = [col for col in data.columns if col not in ['Ticker', 'Shift', 'Return']]
        X = torch.FloatTensor(data[feature_cols].values)
        y = torch.LongTensor(self._compute_quintiles(data['Return']))
        
        # Create ticker mapping
        unique_tickers = data['Ticker'].unique()
        self.ticker_to_idx = {ticker: idx for idx, ticker in enumerate(unique_tickers)}
        self.idx_to_ticker = {idx: ticker for ticker, idx in self.ticker_to_idx.items()}
        
        ticker_indices = torch.LongTensor([self.ticker_to_idx[ticker] for ticker in data['Ticker']])
        
        # Create dataset and splits
        dataset = FinancialDataset(X, y, ticker_indices)
        dataset.idx_to_ticker = self.idx_to_ticker
        
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        return train_loader, val_loader, test_loader

    def _compute_quintiles(self, returns: pd.Series) -> np.ndarray:
        """Compute return quintiles"""
        returns = returns.fillna(0)
        quintiles = pd.qcut(returns, q=5, labels=False, duplicates='drop').fillna(2).astype(int)
        return quintiles.values

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train the model"""
        logger.info("Training started...")
        
        base_model = BaseModel(self.config).to(self.device)
        num_tasks = len(self.ticker_to_idx)
        meta_model = MetaModel(base_model, num_tasks, self.config).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        base_optimizer = torch.optim.Adam(base_model.parameters(), lr=self.config.learning_rate)
        meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=self.config.learning_rate)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training
            meta_model.train()
            train_loss = 0
            for X, y, tickers in train_loader:
                X, y, tickers = X.to(self.device), y.to(self.device), tickers.to(self.device)
                
                meta_optimizer.zero_grad()
                outputs = meta_model(X, tickers)
                loss = criterion(outputs, y)
                loss.backward()
                meta_optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self._validate(meta_model, val_loader, criterion)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.meta_model = meta_model
                logger.info(f"Epoch {epoch+1}: New best model saved")
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
                logger.info(f"Training Loss: {train_loss/len(train_loader):.4f}")
                logger.info(f"Validation Loss: {val_loss:.4f}")
        
        logger.info("Training completed.")

    def _validate(self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model"""
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y, tickers in val_loader:
                X, y, tickers = X.to(self.device), y.to(self.device), tickers.to(self.device)
                outputs = model(X, tickers)
                loss = criterion(outputs, y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def predict_filtered(self, model: nn.Module, test_loader: DataLoader) -> pd.DataFrame:
        """Generate filtered predictions with softmax probabilities."""
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch_X, _, batch_tickers in test_loader:
                batch_X = batch_X.to(self.device)
                batch_tickers = batch_tickers.to(self.device)
                
                # Get softmax probabilities
                outputs = model(batch_X, batch_tickers).cpu().numpy()  # Shape: [batch_size, output_size]
                
                # Convert ticker indices back to names
                batch_ticker_names = [
                    test_loader.dataset.dataset.idx_to_ticker[int(idx)]
                    for idx in batch_tickers.cpu().numpy()
                ]
                
                # Store predictions
                for ticker, probs in zip(batch_ticker_names, outputs):
                    predictions.append({
                        'Ticker': ticker,
                        'Probabilities': probs  # Store full softmax probabilities
                    })

        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        return predictions_df
    
    def aggregate_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate softmax probabilities for each ticker and assign the final quintile.
        """
        # Split probabilities into separate columns for each quintile
        probabilities_df = pd.DataFrame(predictions_df['Probabilities'].tolist())
        probabilities_df.columns = [f'Q{i}' for i in range(5)]  # Q0 to Q4
        probabilities_df['Ticker'] = predictions_df['Ticker']
        
        # Group by ticker and calculate mean probabilities
        aggregated_probs = probabilities_df.groupby('Ticker').mean().reset_index()
        
        # Determine the final quintile based on the highest mean probability
        aggregated_probs['FinalQuintile'] = aggregated_probs.iloc[:, 1:6].idxmax(axis=1).str[-1].astype(int)
        
        return aggregated_probs[['Ticker', 'FinalQuintile']]




    def save_model(self, path: str = './trading_competition/models') -> None:
        """Save the trained model."""
        if self.meta_model is None:
            raise ValueError("No trained model to save. Please train the model first.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'meta_model_state_dict': self.meta_model.state_dict(),
            'base_model_state_dict': self.meta_model.base_model.state_dict(),
            'ticker_to_idx': self.ticker_to_idx,
            'config': self.config
        }, save_path / 'model.pth')
        
        logger.info(f"Model saved to {save_path / 'model.pth'}")

    def load_model(self, path: str = './trading_competition/models/model.pth') -> None:
        """Load a trained model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Initialize models
        base_model = BaseModel(self.config).to(self.device)
        num_tasks = len(checkpoint['ticker_to_idx'])
        self.meta_model = MetaModel(base_model, num_tasks, self.config).to(self.device)
        
        # Load state dicts
        self.meta_model.load_state_dict(checkpoint['meta_model_state_dict'])
        self.ticker_to_idx = checkpoint['ticker_to_idx']
        
        logger.info(f"Model loaded from {path}")

def initialize_forecaster(input_size: int = None) -> FinancialForecaster:
    """Initialize the forecaster with default configuration."""
    if input_size is None:
        feature_engine = FeatureEngine()
        input_size = feature_engine.get_feature_count()


    config = ModelConfig(
        input_size=feature_engine.get_feature_count(),
        hidden_sizes=[64, 32],
        output_size=5,
        learning_rate=0.001,
        batch_size=256,
        epochs=50,
        mesa_dim=2,
        dropout_rate=0.2
)

    
    return FinancialForecaster(config=config)

if __name__ == "__main__":

    
    # Load environment variables
    load_dotenv(dotenv_path='./trading_competition/.env')
    ALPACA_API_BASE_URL = os.getenv("ALPACA_API_BASE_URL")
    ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID")
    ALPACA_API_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    
    # Initialize Alpaca API
    alpaca = tradeapi.REST(
        key_id=ALPACA_API_KEY_ID,
        secret_key=ALPACA_API_SECRET_KEY,
        base_url=ALPACA_API_BASE_URL
    )
    
    # Initialize components
    universe = pd.read_csv('./trading_competition/data/stock_universe.csv')
    symbols = universe['symbol'].tolist()
    
    # Initialize forecaster
    forecaster = initialize_forecaster()
    
    logger.info("Starting training workflow...")
    
    try:
        # Setup data paths
        data_path = Path('./trading_competition/data/processed_data.csv')
        raw_data_dir = Path('./trading_competition/data/raw')
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        if data_path.exists():
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            logger.info("Loaded processed data from file")
        else:
            # Fetch and process data from Alpaca
            raw_data = {}
            for symbol in symbols:
                try:
                    logger.info(f"Fetching data for {symbol}")
                    bars = alpaca.get_bars(
                        symbol,
                        timeframe='1Week',
                        start='2018-01-01',
                        end='2024-12-02'
                    ).df

                    if not bars.empty:
                        # Save raw data
                        bars.to_csv(raw_data_dir / f"{symbol}.csv")
                        raw_data[symbol] = bars
                    else:
                        logger.warning(f"No data found for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")

            # Process the data
            feature_engine = FeatureEngine()
            processed_data = feature_engine.process_data_with_shifts(raw_data, shifts=[0, 1, 2, 3, 4, 5, 6, 7])

            # Save processed data
            processed_data.to_csv(data_path)
            data = processed_data
            logger.info("Processed new data and saved to file")

        # Prepare data and train model
        train_loader, val_loader, test_loader = forecaster.prepare_data(data)

        # Train the model
        logger.info("Training the model...")
        forecaster.train(train_loader, val_loader)

        # Ensure meta_model is initialized
        if forecaster.meta_model is None:
            logger.error("Model training did not complete successfully.")
            raise ValueError("Meta model is not initialized. Ensure training is completed before prediction.")

        # Generate predictions for all shifts
        logger.info("Generating predictions for all shifts...")
        predictions = forecaster.predict_filtered(
            forecaster.meta_model,
            test_loader
        )

        # Aggregate predictions using confidence-based logic
        logger.info("Aggregating predictions across shifts...")
        aggregated_predictions = forecaster.aggregate_predictions(predictions)

        # Save aggregated results
        results_path = Path('./trading_competition/results')
        results_path.mkdir(parents=True, exist_ok=True)

        aggregated_predictions.to_csv(results_path / 'next_week_predictions.csv', index=False)
        logger.info(f"Saved aggregated predictions to {results_path / 'next_week_predictions.csv'}")

        # Save trained model
        model_path = Path('./trading_competition/models')
        model_path.mkdir(parents=True, exist_ok=True)
        forecaster.save_model(path=str(model_path))

        logger.info("Training workflow completed successfully")

    except Exception as e:
        logger.error(f"Error in training workflow: {str(e)}")
        raise

    
    