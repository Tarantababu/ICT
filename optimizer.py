import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

class ForexSignalBot:
    def __init__(self, pairs, api_key, sl_pips, pip_values, risk_rewards, initial_capital=10000, risk_per_trade=0.01):
        self.pairs = pairs
        self.data = {pair: {} for pair in pairs}
        self.signals = {pair: [] for pair in pairs}
        self.api_key = api_key
        self.sl_pips = sl_pips
        self.pip_values = pip_values
        self.risk_rewards = risk_rewards
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade  # Risk as a % of current capital
        self.capital = initial_capital

    def fetch_data(self):
        for pair in self.pairs:
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={pair[:3]}&to_symbol={pair[3:]}&apikey={self.api_key}&outputsize=compact"
            response = requests.get(url)
            data = response.json()
            if "Time Series FX (Daily)" in data:
                df = pd.DataFrame(data["Time Series FX (Daily)"]).T
                df.columns = ['open', 'high', 'low', 'close']
                df.index = pd.to_datetime(df.index)
                self.data[pair] = df.astype(float)
            else:
                st.error(f"Failed to fetch data for {pair}")

    def generate_signals(self):
        for pair in self.pairs:
            df = self.data[pair]
            signals = []
            for i in range(1, len(df)):
                entry_price = df['close'].iloc[i-1]
                direction = np.random.choice(['Long', 'Short'])  # Random signal generation
                stop_loss = entry_price - self.sl_pips[pair] * self.pip_values[pair] if direction == 'Long' else entry_price + self.sl_pips[pair] * self.pip_values[pair]
                take_profit = entry_price + self.risk_rewards[pair] * abs(entry_price - stop_loss) if direction == 'Long' else entry_price - self.risk_rewards[pair] * abs(entry_price - stop_loss)

                # Assuming the trade hits either TP or SL (simplified for backtest)
                exit_price = take_profit if np.random.random() > 0.5 else stop_loss
                pips = self.calculate_pips(entry_price, exit_price, direction, self.pip_values[pair])

                signals.append({
                    'pair': pair,
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'exit_price': exit_price,
                    'pips': pips
                })
            self.signals[pair] = signals

    def calculate_pips(self, entry_price, exit_price, direction, pip_value):
        if direction == 'Long':
            return (exit_price - entry_price) / pip_value
        else:  # Short
            return (entry_price - exit_price) / pip_value

    def simulate_trading(self, signals):
        capital = self.initial_capital
        for signal in signals:
            risk_amount = capital * self.risk_per_trade
            pips = signal['pips']
            pip_value = self.pip_values[signal['pair']]
            profit_or_loss = pips * pip_value

            trade_outcome = risk_amount * profit_or_loss / abs(signal['stop_loss'] - signal['entry_price'])
            capital += trade_outcome

        return capital

    def optimize_parameters(self):
        def objective(params):
            stop_loss_pips, risk_reward = params
            for pair in self.pairs:
                self.sl_pips[pair] = stop_loss_pips
                self.risk_rewards[pair] = risk_reward

            self.generate_signals()
            total_capital = 0
            for pair in self.pairs:
                total_capital += self.simulate_trading(self.signals[pair])

            return -total_capital  # We negate to maximize final capital

        result = minimize(objective, [20, 2], bounds=[(1, 100), (1, 5)], method='SLSQP')
        best_sl, best_rr = result.x
        return best_sl, best_rr

    def run(self):
        self.fetch_data()
        best_sl, best_rr = self.optimize_parameters()

        st.write(f"Best Stop Loss (pips): {best_sl:.2f}")
        st.write(f"Best Risk-Reward Ratio: {best_rr:.2f}")

        # Calculate the final capital with the optimized parameters
        final_capital = 0
        for pair in self.pairs:
            final_capital += self.simulate_trading(self.signals[pair])

        st.write(f"Final Capital: {final_capital:,.2f}")

        # Generate chart for the optimized parameters
        for pair in self.pairs:
            chart = create_chart(pair, self.data[pair], self.signals[pair])
            st.plotly_chart(chart)

def create_chart(pair, data, signals):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.3, 0.7])

    # Price data
    fig.add_trace(
        go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Price'),
        row=1, col=1
    )

    # Mark signals on chart
    for signal in signals:
        if signal['direction'] == 'Long':
            fig.add_trace(go.Scatter(x=[data.index[signals.index(signal)]], y=[signal['entry_price']],
                                     mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=[data.index[signals.index(signal)]], y=[signal['entry_price']],
                                     mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'), row=1, col=1)

    fig.update_layout(title=f"{pair} Price Chart with Signals", xaxis_title='Date', yaxis_title='Price')
    return fig

def main():
    st.title('Forex Signal Bot with Optimization')

    # Sidebar for user inputs
    st.sidebar.header('Settings')
    api_key = st.sidebar.text_input('Enter your Alpha Vantage API key:', type='password')
    pairs = st.sidebar.multiselect('Select currency pairs:', ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDCAD', 'NZDUSD', 'CADCHF', 'EURCAD', 'GBPAUD', 'AUDJPY'])

    # Create dictionaries to store pair-specific settings
    sl_pips = {}
    pip_values = {}
    risk_rewards = {}

    # Input fields for each selected pair
    for pair in pairs:
        st.sidebar.subheader(f'Settings for {pair}')
        sl_pips[pair] = st.sidebar.number_input(f'Stop Loss for {pair} (in pips):', min_value=1, max_value=100, value=20, key=f'sl_{pair}')
        pip_values[pair] = st.sidebar.number_input(f'Pip Value for {pair}:', min_value=0.0001, max_value=0.01, value=0.0001, format='%f', key=f'pip_{pair}')
        risk_rewards[pair] = st.sidebar.number_input(f'Risk-Reward Ratio for {pair}:', min_value=1.0, max_value=5.0, value=2.0, key=f'rr_{pair}')

    if not api_key:
        st.warning('Please enter your Alpha Vantage API key to proceed.')
        return

    bot = ForexSignalBot(pairs, api_key, sl_pips, pip_values, risk_rewards)

    if st.button('Run Optimization and Generate Signals'):
        with st.spinner('Fetching data and optimizing parameters...'):
            bot.run()

if __name__ == "__main__":
    main()
