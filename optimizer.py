import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexSignalOptimizer:
    def __init__(self, pairs, api_key, sl_pips_range, risk_reward_range, pip_values, initial_capital, risk_per_trade):
        self.pairs = pairs
        self.api_key = api_key
        self.sl_pips_range = sl_pips_range
        self.risk_reward_range = risk_reward_range
        self.pip_values = pip_values
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.data = {pair: {} for pair in pairs}
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

    def simulate_trade(self, pair, entry_price, sl_pips, risk_reward, pip_value, capital):
        # Calculate stop loss and take profit prices
        stop_loss = entry_price - (sl_pips * pip_value)
        take_profit = entry_price + (sl_pips * risk_reward * pip_value)

        # Calculate position size based on the percentage risk and stop loss
        risk_amount = capital * self.risk_per_trade
        position_size = risk_amount / (sl_pips * pip_value)

        # Placeholder for trade outcome
        # Assume a simple random outcome: 70% chance of hitting take profit, 30% chance of hitting stop loss
        outcome = np.random.choice(['tp', 'sl'], p=[0.7, 0.3])

        if outcome == 'tp':
            profit = (take_profit - entry_price) * position_size
            capital += profit
        else:
            loss = (entry_price - stop_loss) * position_size
            capital -= loss

        return capital

    def optimize_signals(self):
        for pair in self.pairs:
            best_final_capital = self.initial_capital
            best_sl_pips = None
            best_risk_reward = None

            for sl_pips in self.sl_pips_range:
                for risk_reward in self.risk_reward_range:
                    capital = self.initial_capital

                    # Simulate trades across the available data
                    for i in range(len(self.data[pair]['5min']) - 1):
                        entry_price = self.data[pair]['5min'].iloc[i]['Close']
                        capital = self.simulate_trade(
                            pair, entry_price, sl_pips, risk_reward, self.pip_values[pair], capital)

                    # Track the best stop loss and risk-reward combo based on final capital
                    if capital > best_final_capital:
                        best_final_capital = capital
                        best_sl_pips = sl_pips
                        best_risk_reward = risk_reward

            # Store the best results for this pair
            self.optimized_results[pair] = {
                'best_sl_pips': best_sl_pips,
                'best_risk_reward': best_risk_reward,
                'final_capital': best_final_capital
            }

    def display_optimized_results(self):
        for pair in self.pairs:
            st.subheader(f'Optimized Results for {pair}')
            st.write(f"Best Stop Loss (pips): {self.optimized_results[pair]['best_sl_pips']}")
            st.write(f"Best Risk-Reward Ratio: {self.optimized_results[pair]['best_risk_reward']}")
            st.write(f"Final Capital: {self.optimized_results[pair]['final_capital']:.2f}")

def main():
    st.title('Forex Signal Optimizer')

    # Sidebar for user inputs
    st.sidebar.header('Settings')
    api_key = st.sidebar.text_input('Enter your Alpha Vantage API key:', type='password')
    pairs = st.sidebar.multiselect('Select currency pairs:', ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDCAD', 'NZDUSD', 'CADCHF', 'EURCAD', 'GBPAUD', 'AUDJPY'])

    # Range inputs for stop loss and risk-reward ratio
    sl_pips_range = st.sidebar.slider('Stop Loss Range (in pips):', 10, 100, (20, 50), step=1)
    risk_reward_range = st.sidebar.slider('Risk-Reward Range:', 1.0, 5.0, (2.0, 4.0), step=0.1)

    # Pip value input for each selected pair
    pip_values = {}
    for pair in pairs:
        pip_values[pair] = st.sidebar.number_input(f'Pip Value for {pair}:', min_value=0.0001, max_value=0.01, value=0.0001, format='%f', key=f'pip_{pair}')

    # Initial capital and risk per trade input
    initial_capital = st.sidebar.number_input('Initial Capital:', min_value=1000.0, value=10000.0, step=100.0)
    risk_per_trade = st.sidebar.slider('Risk per Trade (% of capital):', 0.01, 0.05, 0.02, step=0.01)

    if not api_key:
        st.warning('Please enter your Alpha Vantage API key to proceed.')
        return

    bot = ForexSignalOptimizer(pairs, api_key, range(*sl_pips_range), np.arange(*risk_reward_range, 0.1), pip_values, initial_capital, risk_per_trade)

    if st.button('Optimize Signals'):
        with st.spinner('Fetching data and optimizing signals...'):
            bot.fetch_data()
            bot.optimize_signals()
            bot.display_optimized_results()

if __name__ == "__main__":
    main()
