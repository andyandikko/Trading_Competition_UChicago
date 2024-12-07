import pandas as pd
from alpaca_trade_api.rest import REST
import os

# Constants for Alpaca API
API_KEY = 'PKSGLWGX3JG73FGO3AGM'
API_SECRET = 'Wz3UVEBqmPMLE7J5NCUl3NTkz77CXvvauau6JMXJ'
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def fetch_latest_prices(stock_list):
    """
    Fetch the latest prices for the given stock list.
    Returns a dictionary of {symbol: latest_price}.
    """
    latest_prices = {}
    active_stocks = []
    for symbol in stock_list:
        try:
            latest_trade = api.get_latest_trade(symbol)
            latest_prices[symbol] = latest_trade.price
            active_stocks.append(symbol)
            print(f"Latest price for {symbol}: {latest_trade.price}")
        except Exception as e:
            print(f"Error fetching latest price for {symbol}: {e}")
    return latest_prices, active_stocks

def place_orders(stock_allocations, total_balance):
    """
    Place orders based on stock allocations and total balance.
    Dynamically adjusts remaining balance for the last stock.
    Returns the remaining balance after all orders are placed.
    """
    num_stocks = len(stock_allocations)
    allocation_per_stock = total_balance // num_stocks
    remaining_balance = total_balance

    for idx, (symbol, latest_price) in enumerate(stock_allocations.items()):
        try:
            # For the last stock, use the minimum of the remaining balance allocation and the equal per-stock allocation
            if idx == num_stocks - 1:
                qty = min(int(remaining_balance // latest_price), int(allocation_per_stock // latest_price))
            else:
                qty = int(allocation_per_stock // latest_price)

            # Adjust remaining balance
            remaining_balance -= qty * latest_price

            # Submit a limit buy order
            if qty > 0:
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='limit',
                    time_in_force='gtc',
                    limit_price=round(latest_price * 1.01, 2),
                )
                print(f"Buy order for {qty} shares of {symbol} placed at {round(latest_price * 1.01, 2)}")
            else:
                print(f"Skipping order for {symbol}: qty must be > 0")
        except Exception as e:
            print(f"An error occurred while placing order for {symbol}: {e}")
    return remaining_balance

def filter_active_stocks(data):
    """
    Filter out inactive stocks from the DataFrame.
    """
    stock_list = data['Ticker'].tolist()
    _, active_stocks = fetch_latest_prices(stock_list)
    return data[data['Ticker'].isin(active_stocks)]

def allocate_funds(data, total_balance, quintile):
    """
    Allocate funds to stocks in the given quintile.
    Returns remaining balance after allocation and the modified data.
    """
    quintile_data = data[data['PredictedQuintile'] == quintile]
    quintile_data.loc[quintile_data['Ticker'] == 'FB', 'Ticker'] = 'META'
    stock_list = quintile_data['Ticker'].tolist()
    
    latest_prices, _ = fetch_latest_prices(stock_list)
    remaining_balance = place_orders(latest_prices, total_balance)
    return remaining_balance, data

def reallocate_remaining_balance(data, remaining_balance):
    """
    Allocate remaining balance to the top quintile stock with the lowest price.
    """
    top_quintile = data['PredictedQuintile'].max()
    top_quintile_data = data[data['PredictedQuintile'] == top_quintile]
    top_quintile_data.loc[top_quintile_data['Ticker'] == 'FB', 'Ticker'] = 'META'
    lowest_price_stock = top_quintile_data.sort_values(by='Ticker')['Ticker'].iloc[0]
    
    latest_prices, _ = fetch_latest_prices([lowest_price_stock])
    if latest_prices:
        place_orders(latest_prices, remaining_balance)

if __name__ == "__main__":
    if "trading_competition" in os.getcwd():
        os.chdir("..")
    
    # Load data
    csv_file_path = './trading_competition/results/next_week_predictions.csv'
    data = pd.read_csv(csv_file_path)

    # Filter out inactive stocks
    data = filter_active_stocks(data)

    # Allocate $70,000 to top quintile stocks
    total_balance = 70000
    remaining_balance, data = allocate_funds(data, total_balance, quintile=data['PredictedQuintile'].max())

    # Allocate $40,000 to the second top quintile stocks
    total_balance = 30000
    remaining_balance += allocate_funds(data, total_balance, quintile=data['PredictedQuintile'].max() - 1)[0]

    # Reallocate remaining balance to the top quintile stock with the lowest price
    if remaining_balance > 100:  # Minimum threshold for allocation
        reallocate_remaining_balance(data, remaining_balance)

    print("Order placement process completed.")
    