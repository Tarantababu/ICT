import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexSignalBot:
    def __init__(self, pairs, api_key, sl_pips):
        self.pairs = pairs
        self.data = {pair: {} for pair in pairs}
        self.signals = {pair: [] for pair in pairs}
        self.api_key = api_key
        self.sl_pips = sl_pips

    def fetch_data(self):
        for pair in self.pairs:
            from_symbol, to_symbol = pair[:3], pair[3:]
            url = f'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={from_symbol}&to_symbol={to_symbol}&interval=5min&outputsize=full&apikey={self.api_key}'
            
            try:
                r = requests.get(url)
                r.raise_for_status()
                data = r.json()

                if 'Time Series FX (5min)' in data:
                    df = pd.DataFrame(data['Time Series FX (5min)']).T
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    df = df.astype(float)
                    df.columns = ['Open', 'High', 'Low', 'Close']
                    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
                    self.data[pair] = df.iloc[-100:]  # Keep only the last 100 data points
                else:
                    st.error(f"Failed to fetch data for {pair}. Please check your API key and try again.")
            except Exception as e:
                st.error(f"Error fetching data for {pair}: {e}")

    def detect_signal(self, pair):
        df = self.data[pair]
        if len(df) < 2:
            return None

        current_price = df['Close'].iloc[-1]
        previous_price = df['Close'].iloc[-2]
        high = df['High'].max()
        low = df['Low'].min()

        if current_price > high and previous_price <= high:
            return "High sweep"
        elif current_price < low and previous_price >= low:
            return "Low sweep"
        return None

    def generate_signals(self):
        for pair in self.pairs:
            signal = self.detect_signal(pair)
            if signal:
                current_price = self.data[pair]['Close'].iloc[-1]
                pip_value = 0.0001  # Assuming 4 decimal places for forex pairs
                if signal == "High sweep":
                    stop_loss = current_price + self.sl_pips * pip_value
                    take_profit = current_price - self.sl_pips * pip_value * 2
                else:  # Low sweep
                    stop_loss = current_price - self.sl_pips * pip_value
                    take_profit = current_price + self.sl_pips * pip_value * 2

                self.signals[pair].append({
                    "time": self.data[pair].index[-1],
                    "price": current_price,
                    "signal": signal,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                })

    def run_real_time(self):
        self.fetch_data()
        self.generate_signals()

def create_chart(pair, data, signals):
    fig = make_subplots(rows=1, cols=1)

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Price'))

    # Add signals, SL, and TP to the chart
    for signal in signals:
        fig.add_trace(go.Scatter(x=[signal['time']], y=[signal['price']],
                                 mode='markers',
                                 marker=dict(symbol='triangle-up' if signal['signal'] == 'Low sweep' else 'triangle-down',
                                             size=10,
                                             color='green' if signal['signal'] == 'Low sweep' else 'red'),
                                 name=f"{signal['signal']} at {signal['price']:.5f}"))
        
        fig.add_trace(go.Scatter(x=[signal['time'], signal['time']], 
                                 y=[signal['price'], signal['stop_loss']],
                                 mode='lines',
                                 line=dict(color='red', width=1, dash='dash'),
                                 name=f"SL: {signal['stop_loss']:.5f}"))
        
        fig.add_trace(go.Scatter(x=[signal['time'], signal['time']], 
                                 y=[signal['price'], signal['take_profit']],
                                 mode='lines',
                                 line=dict(color='green', width=1, dash='dash'),
                                 name=f"TP: {signal['take_profit']:.5f}"))

    fig.update_layout(title=f'{pair} Chart', xaxis_rangeslider_visible=False)
    return fig

def main():
    st.title('Forex Signal Bot')

    # Sidebar for user inputs
    st.sidebar.header('Settings')
    api_key = st.sidebar.text_input('Enter your Alpha Vantage API key:', type='password')
    pairs = st.sidebar.multiselect('Select currency pairs:', ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'])
    sl_pips = st.sidebar.number_input('Set stop loss (in pips):', min_value=1, max_value=100, value=20)

    if not api_key:
        st.warning('Please enter your Alpha Vantage API key to proceed.')
        return

    bot = ForexSignalBot(pairs, api_key, sl_pips)

    if st.button('Get Signals'):
        with st.spinner('Fetching data and generating signals...'):
            bot.run_real_time()

        # Display active signals
        st.header('Active Signals')
        for pair in pairs:
            if bot.signals[pair]:
                latest_signal = bot.signals[pair][-1]
                st.success(f"{pair}: {latest_signal['signal']} at {latest_signal['price']:.5f}")
                st.info(f"Stop Loss: {latest_signal['stop_loss']:.5f}, Take Profit: {latest_signal['take_profit']:.5f}")
            else:
                st.info(f"No active signals for {pair}")

        # Display charts
        st.header('Charts')
        for pair in pairs:
            if pair in bot.data:
                chart = create_chart(pair, bot.data[pair], bot.signals[pair])
                st.plotly_chart(chart)
            else:
                st.warning(f"No data available for {pair}")

if __name__ == "__main__":
    main()
