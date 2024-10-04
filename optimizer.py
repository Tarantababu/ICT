import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexSignalOptimizer:
    def __init__(self, pairs, api_key, sl_pips_range, risk_reward_range, pip_values):
        self.pairs = pairs
        self.api_key = api_key
        self.sl_pips_range = sl_pips_range
        self.risk_reward_range = risk_reward_range
        self.pip_values = pip_values
        self.data = {pair: {} for pair in pairs}
        self.signals = {pair: [] for pair in pairs}
        self.optimized_results = {pair: [] for pair in pairs}

    def fetch_data(self):
        for pair in self.pairs:
            from_symbol, to_symbol = pair[:3], pair[3:]
            for timeframe in ['60min', '15min', '5min']:
                url = f'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={from_symbol}&to_symbol={to_symbol}&interval={timeframe}&outputsize=full&apikey={self.api_key}'
                
                try:
                    r = requests.get(url)
                    r.raise_for_status()
                    data = r.json()

                    if f'Time Series FX ({timeframe})' in data:
                        df = pd.DataFrame(data[f'Time Series FX ({timeframe})']).T
                        df.index = pd.to_datetime(df.index)
                        df = df.sort_index()
                        df = df.astype(float)
                        df.columns = ['Open', 'High', 'Low', 'Close']
                        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
                        self.data[pair][timeframe] = df
                    else:
                        st.error(f"Failed to fetch data for {pair} at {timeframe} timeframe. Please check your API key and try again.")
                except Exception as e:
                    st.error(f"Error fetching data for {pair} at {timeframe} timeframe: {e}")

    def simulate_trade(self, pair, entry_price, sl_pips, risk_reward, pip_value):
        # Simulate setting stop loss and take profit
        stop_loss = entry_price - (sl_pips * pip_value)
        take_profit = entry_price + (sl_pips * risk_reward * pip_value)
        
        # Placeholder for trade exit logic
        # (This is where you would simulate exit conditions based on future price movements)
        # For simplicity, we'll just assume a successful trade hitting take profit
        exit_price = take_profit

        return stop_loss, take_profit, exit_price

    def optimize_signals(self):
        for pair in self.pairs:
            for sl_pips in self.sl_pips_range:
                for risk_reward in self.risk_reward_range:
                    # Placeholder logic for optimizing signals
                    # (You could include further customization based on your trading logic)
                    for i in range(len(self.data[pair]['5min']) - 1):
                        entry_price = self.data[pair]['5min'].iloc[i]['Close']
                        stop_loss, take_profit, exit_price = self.simulate_trade(
                            pair, entry_price, sl_pips, risk_reward, self.pip_values[pair])

                        self.optimized_results[pair].append({
                            'sl_pips': sl_pips,
                            'risk_reward': risk_reward,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'exit_price': exit_price,
                        })

    def display_optimized_results(self):
        for pair in self.pairs:
            st.subheader(f'Optimized Results for {pair}')
            st.write(pd.DataFrame(self.optimized_results[pair]))

def main():
    st.title('Forex Signal Optimizer')

    # Sidebar for user inputs
    st.sidebar.header('Settings')
    api_key = st.sidebar.text_input('Enter your Alpha Vantage API key:', type='password')
    pairs = st.sidebar.multiselect('Select currency pairs:', ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDCAD', 'NZDUSD', 'CADCHF', 'EURCAD', 'GBPAUD', 'AUDJPY'])

    # Range inputs for stop loss and risk-reward ratio
    sl_pips_range = st.sidebar.slider('Stop Loss Range (in pips):', 0, 100, (20, 50), step=1)
    risk_reward_range = st.sidebar.slider('Risk-Reward Range:', 1.0, 5.0, (2.0, 4.0), step=0.1)

    # Pip value input for each selected pair
    pip_values = {}
    for pair in pairs:
        pip_values[pair] = st.sidebar.number_input(f'Pip Value for {pair}:', min_value=0.0001, max_value=0.01, value=0.0001, format='%f', key=f'pip_{pair}')

    if not api_key:
        st.warning('Please enter your Alpha Vantage API key to proceed.')
        return

    bot = ForexSignalOptimizer(pairs, api_key, range(*sl_pips_range), np.arange(*risk_reward_range, 0.1), pip_values)

    if st.button('Optimize Signals'):
        with st.spinner('Fetching data and optimizing signals...'):
            bot.fetch_data()
            bot.optimize_signals()
            bot.display_optimized_results()

if __name__ == "__main__":
    main()
