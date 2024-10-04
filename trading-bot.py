import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexSignalBot:
    def __init__(self, pairs, api_key, sl_pips):
        self.pairs = pairs
        self.data = {pair: pd.DataFrame() for pair in pairs}
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
                    self.data[pair] = df
                else:
                    st.error(f"Failed to fetch data for {pair}. Please check your API key and try again.")
            except Exception as e:
                st.error(f"Error fetching data for {pair}: {e}")

    def detect_signal(self, pair, index):
        df = self.data[pair]
        if index < 12:  # Need at least 1 hour of data
            return None

        current_price = df['Close'].iloc[index]
        previous_price = df['Close'].iloc[index-1]
        high = df['High'].iloc[index-12:index+1].max()  # 1-hour high (12 * 5min)
        low = df['Low'].iloc[index-12:index+1].min()  # 1-hour low

        if current_price > high and previous_price <= high:
            return "High sweep"
        elif current_price < low and previous_price >= low:
            return "Low sweep"
        return None

    def generate_historical_signals(self):
        for pair in self.pairs:
            df = self.data[pair]
            self.signals[pair] = []
            for i in range(12, len(df)):  # Start from 13th candle (1 hour of data)
                signal = self.detect_signal(pair, i)
                if signal:
                    self.add_signal(pair, i, signal)

    def add_signal(self, pair, index, signal_type):
        df = self.data[pair]
        current_price = df['Close'].iloc[index]
        pip_value = 0.0001 if 'JPY' not in pair else 0.01
        sl_pips = self.sl_pips[pair]
        
        if signal_type == "High sweep":
            stop_loss = current_price + sl_pips * pip_value
            take_profit = current_price - sl_pips * pip_value * 2
        else:  # Low sweep
            stop_loss = current_price - sl_pips * pip_value
            take_profit = current_price + sl_pips * pip_value * 2

        self.signals[pair].append({
            "time": df.index[index],
            "price": current_price,
            "signal": signal_type,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        })

    def update_real_time(self):
        for pair in self.pairs:
            df = self.data[pair]
            latest_index = len(df) - 1
            signal = self.detect_signal(pair, latest_index)
            if signal:
                self.add_signal(pair, latest_index, signal)

    def run(self):
        self.fetch_data()
        self.generate_historical_signals()
        self.update_real_time()

def create_chart(pair, data, signals):
    fig = make_subplots(rows=1, cols=1)

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Price'))

    # Add historical and current signals to the chart
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
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # Hide weekends
    return fig

def main():
    st.title('Forex Signal Bot')

    # Sidebar for user inputs
    st.sidebar.header('Settings')
    api_key = st.sidebar.text_input('Enter your Alpha Vantage API key:', type='password')
    pairs = st.sidebar.multiselect('Select currency pairs:', ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'])
    
    # Create a dictionary to store stop loss for each pair
    sl_pips = {}
    for pair in pairs:
        sl_pips[pair] = st.sidebar.number_input(f'Set stop loss for {pair} (in pips):', min_value=1, max_value=100, value=20, key=pair)

    if not api_key:
        st.warning('Please enter your Alpha Vantage API key to proceed.')
        return

    bot = ForexSignalBot(pairs, api_key, sl_pips)

    if st.button('Get Signals'):
        with st.spinner('Fetching data and generating signals...'):
            bot.run()

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
            if not bot.data[pair].empty:
                chart = create_chart(pair, bot.data[pair], bot.signals[pair])
                st.plotly_chart(chart)
            else:
                st.warning(f"No data available for {pair}")

        # Display all signals
        st.header('All Signals')
        for pair in pairs:
            if bot.signals[pair]:
                st.subheader(f'Signals for {pair}')
                for signal in bot.signals[pair]:
                    st.write(f"Time: {signal['time']}, Signal: {signal['signal']}, Price: {signal['price']:.5f}")
            else:
                st.info(f"No signals generated for {pair}")

if __name__ == "__main__":
    main()
