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
        self.data = {pair: {} for pair in pairs}
        self.signals = {pair: [] for pair in pairs}
        self.api_key = api_key
        self.sl_pips = sl_pips

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

    def mark_highs_lows(self, pair, date):
        session_start = datetime.combine(date, datetime.min.time()).replace(tzinfo=pytz.timezone('America/New_York'))
        session_end = (session_start + timedelta(hours=8, minutes=30))
        session_data = self.data[pair]['60min'].loc[session_start:session_end]
        return session_data['High'].max(), session_data['Low'].min()

    def detect_sweep(self, price, high, low):
        if price > high:
            return "High sweep"
        elif price < low:
            return "Low sweep"
        return None

    def detect_market_structure_shift(self, pair, timeframe, start_index, direction):
        data = self.data[pair][timeframe].loc[start_index:]
        if direction == "High sweep":
            high = data['High'].iloc[0]
            for i in range(1, len(data)):
                if data['High'].iloc[i] < high:
                    low = data['Low'].iloc[i]
                    for j in range(i + 1, len(data)):
                        if data['Low'].iloc[j] < low:
                            return data.index[j]
                elif data['High'].iloc[i] > high:
                    high = data['High'].iloc[i]
        elif direction == "Low sweep":
            low = data['Low'].iloc[0]
            for i in range(1, len(data)):
                if data['Low'].iloc[i] > low:
                    high = data['High'].iloc[i]
                    for j in range(i + 1, len(data)):
                        if data['High'].iloc[j] > high:
                            return data.index[j]
                elif data['Low'].iloc[i] < low:
                    low = data['Low'].iloc[i]
        return None

    def find_fvg(self, pair, timeframe, start_index, direction):
        data = self.data[pair][timeframe].loc[start_index:]
        for i in range(len(data) - 2):
            candle_1 = data.iloc[i]
            candle_2 = data.iloc[i + 1]
            candle_3 = data.iloc[i + 2]

            if direction == "High sweep" and candle_1['Low'] > candle_3['High']:
                return {
                    "start_time": candle_1.name,
                    "end_time": candle_3.name,
                    "gap_start": candle_1['Low'],
                    "gap_end": candle_3['High'],
                    "direction": "bearish",
                    "fvg_high": candle_1['High']
                }
            elif direction == "Low sweep" and candle_1['High'] < candle_3['Low']:
                return {
                    "start_time": candle_1.name,
                    "end_time": candle_3.name,
                    "gap_start": candle_1['High'],
                    "gap_end": candle_3['Low'],
                    "direction": "bullish",
                    "fvg_low": candle_1['Low']
                }
        return None

    def set_stop_loss_and_take_profit(self, entry_price, fvg, direction):
        pip_value = 0.0001  # Assuming 4 decimal places for forex pairs
        if direction == "High sweep":  # Short trade
            stop_loss = entry_price + self.sl_pips * pip_value
            take_profit = entry_price - self.sl_pips * pip_value * 2  # 1:2 risk-reward ratio
        else:  # Low sweep, Long trade
            stop_loss = entry_price - self.sl_pips * pip_value
            take_profit = entry_price + self.sl_pips * pip_value * 2  # 1:2 risk-reward ratio
        return stop_loss, take_profit

    def generate_signals(self):
        for pair in self.pairs:
            data_5m = self.data[pair]['5min']
            data_1h = self.data[pair]['60min']
            self.signals[pair] = []

            for i in range(len(data_5m) - 1):
                current_time = data_5m.index[i]
                if current_time.time() < pd.Timestamp("08:30").time() or current_time.time() >= pd.Timestamp("11:00").time():
                    continue

                high, low = self.mark_highs_lows(pair, current_time.date())

                current_price = data_5m.iloc[i]['Close']
                sweep = self.detect_sweep(current_price, high, low)

                if sweep:
                    choch = self.detect_market_structure_shift(pair, '5min', current_time, sweep)
                    if choch:
                        fvg = self.find_fvg(pair, '5min', choch, sweep)
                        if fvg:
                            entry_price = (fvg['gap_start'] + fvg['gap_end']) / 2
                            stop_loss, take_profit = self.set_stop_loss_and_take_profit(entry_price, fvg, sweep)

                            # Simulate trade exit
                            exit_price = None
                            exit_time = None
                            for j in range(i + 1, len(data_5m)):
                                future_price = data_5m.iloc[j]
                                if sweep == "High sweep":  # Short trade
                                    if future_price['High'] >= stop_loss:
                                        exit_price = stop_loss
                                        exit_time = data_5m.index[j]
                                        break
                                    elif future_price['Low'] <= take_profit:
                                        exit_price = take_profit
                                        exit_time = data_5m.index[j]
                                        break
                                else:  # Low sweep, Long trade
                                    if future_price['Low'] <= stop_loss:
                                        exit_price = stop_loss
                                        exit_time = data_5m.index[j]
                                        break
                                    elif future_price['High'] >= take_profit:
                                        exit_price = take_profit
                                        exit_time = data_5m.index[j]
                                        break

                            self.signals[pair].append({
                                "time": current_time,
                                "price": current_price,
                                "sweep": sweep,
                                "choch_time": choch,
                                "fvg": fvg,
                                "entry_price": entry_price,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "direction": "Short" if sweep == "High sweep" else "Long",
                                "exit_price": exit_price,
                                "exit_time": exit_time
                            })

    def run(self):
        self.fetch_data()
        self.generate_signals()

def calculate_stats(signals):
    if not signals:
        return 0, 0, 0

    wins = 0
    total_pips = 0
    pip_value = 0.0001  # Assuming 4 decimal places for forex pairs

    for signal in signals:
        entry_price = signal['entry_price']
        exit_price = signal['exit_price'] if signal['exit_price'] is not None else entry_price
        
        if signal['direction'] == 'Long':
            pips = (exit_price - entry_price) / pip_value
        else:  # Short
            pips = (entry_price - exit_price) / pip_value
        
        total_pips += pips
        if pips > 0:
            wins += 1

    win_rate = (wins / len(signals)) * 100
    avg_pips = total_pips / len(signals)

    return win_rate, avg_pips, len(signals)

def create_chart(pair, data, signals):
    fig = make_subplots(rows=1, cols=1)

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data['5min'].index,
                                 open=data['5min']['Open'],
                                 high=data['5min']['High'],
                                 low=data['5min']['Low'],
                                 close=data['5min']['Close'],
                                 name='Price'))

    for signal in signals:
        # Entry point
        fig.add_trace(go.Scatter(x=[signal['time']], y=[signal['entry_price']],
                                 mode='markers',
                                 marker=dict(symbol='circle', size=8, color='blue'),
                                 name='Entry'))

        # Exit point (if available)
        if signal['exit_price'] is not None and signal['exit_time'] is not None:
            color = 'green' if (signal['direction'] == 'Long' and signal['exit_price'] > signal['entry_price']) or \
                               (signal['direction'] == 'Short' and signal['exit_price'] < signal['entry_price']) else 'red'
            fig.add_trace(go.Scatter(x=[signal['exit_time']], y=[signal['exit_price']],
                                     mode='markers',
                                     marker=dict(symbol='circle', size=8, color=color),
                                     name='Exit'))

    fig.update_layout(title=f'{pair} Chart', xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # Hide weekends
    return fig

def main():
    st.title('Simplified Forex Signal Bot')

    # Sidebar for user inputs
    st.sidebar.header('Settings')
    api_key = st.sidebar.text_input('Enter your Alpha Vantage API key:', type='password')
    pairs = st.sidebar.multiselect('Select currency pairs:', ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'])
    sl_pips = st.sidebar.number_input('Set stop loss (in pips):', min_value=1, max_value=100, value=20)

    if not api_key:
        st.warning('Please enter your Alpha Vantage API key to proceed.')
        return

    bot = ForexSignalBot(pairs, api_key, sl_pips)

    if st.button('Generate Signals'):
        with st.spinner('Fetching data and generating signals...'):
            bot.run()

        # Display signal statistics
        st.header('Signal Statistics')
        for pair in pairs:
            if bot.signals[pair]:
                win_rate, avg_pips, total_signals = calculate_stats(bot.signals[pair])
                st.subheader(f'Statistics for {pair}')
                st.write(f"Total Signals: {total_signals}")
                st.write(f"Win Rate: {win_rate:.2f}%")
                st.write(f"Average Pips: {avg_pips:.2f}")
                st.write("---")
            else:
                st.info(f"No signals generated for {pair}")

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
