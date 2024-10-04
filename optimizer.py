import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexSignalBot:
    def __init__(self, pairs, api_key, sl_pips, pip_values, risk_rewards):
        self.pairs = pairs
        self.data = {pair: {} for pair in pairs}
        self.signals = {pair: [] for pair in pairs}
        self.api_key = api_key
        self.sl_pips = sl_pips
        self.pip_values = pip_values
        self.risk_rewards = risk_rewards

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

    def set_stop_loss_and_take_profit(self, pair, entry_price, fvg, direction):
        pip_value = self.pip_values[pair]
        risk_reward = self.risk_rewards[pair]
        sl_pips = self.sl_pips[pair]

        if direction == "Short":  # High sweep
            fvg_high = fvg['fvg_high']
            stop_loss = fvg_high + (sl_pips * pip_value)  # SL above FVG high
            take_profit = entry_price - (stop_loss - entry_price) * risk_reward
        else:  # Long trade (Low sweep)
            fvg_low = fvg['fvg_low']
            stop_loss = fvg_low - (sl_pips * pip_value)  # SL below FVG low
            take_profit = entry_price + (entry_price - stop_loss) * risk_reward

        return stop_loss, take_profit

    def generate_signals(self):
        for pair in self.pairs:
            data_5m = self.data[pair]['5min']
            data_1h = self.data[pair]['60min']
            self.signals[pair] = []
            signal_count = 1
            entry_prices = set()

            # Initialize a variable to track if a trade is currently active
            active_trade = False

            for i in range(len(data_5m) - 1):
                current_time = data_5m.index[i]
                if current_time.time() < pd.Timestamp("08:30").time() or current_time.time() >= pd.Timestamp("11:00").time():
                    continue

                high, low = self.mark_highs_lows(pair, current_time.date())

                current_price = data_5m.iloc[i]['Close']
                sweep = self.detect_sweep(current_price, high, low)

                if sweep and not active_trade:  # Check if there is no active trade
                    choch = self.detect_market_structure_shift(pair, '5min', current_time, sweep)
                    if choch:
                        fvg = self.find_fvg(pair, '5min', choch, sweep)
                        if fvg:
                            entry_candle = data_5m.loc[current_time]
                            entry_price = entry_candle['Close']
                            
                            if entry_price in entry_prices:
                                continue
                            
                            entry_prices.add(entry_price)

                            direction = "Short" if sweep == "High sweep" else "Long"
                            stop_loss, take_profit = self.set_stop_loss_and_take_profit(pair, entry_price, fvg, direction)

                            # Simulate trade exit
                            exit_price = None
                            exit_time = None
                            for j in range(i + 1, len(data_5m)):
                                future_candle = data_5m.iloc[j]
                                if direction == "Short":
                                    if future_candle['High'] >= stop_loss:
                                        exit_price = stop_loss
                                        exit_time = data_5m.index[j]
                                        break
                                    elif future_candle['Low'] <= take_profit:
                                        exit_price = take_profit
                                        exit_time = data_5m.index[j]
                                        break
                                else:  # Long trade
                                    if future_candle['Low'] <= stop_loss:
                                        exit_price = stop_loss
                                        exit_time = data_5m.index[j]
                                        break
                                    elif future_candle['High'] >= take_profit:
                                        exit_price = take_profit
                                        exit_time = data_5m.index[j]
                                        break

                            # If no exit was triggered, use the last available price
                            if exit_price is None:
                                exit_price = data_5m.iloc[-1]['Close']
                                exit_time = data_5m.index[-1]

                            self.signals[pair].append({
                                "signal_number": signal_count,
                                "time": current_time,
                                "price": current_price,
                                "sweep": sweep,
                                "choch_time": choch,
                                "fvg": fvg,
                                "entry_price": entry_price,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "direction": direction,
                                "exit_price": exit_price,
                                "exit_time": exit_time
                            })
                            active_trade = True  # Mark that a trade has been opened
                            signal_count += 1

                # Reset active trade if we are outside the active trade period
                if active_trade and current_time.time() >= pd.Timestamp("11:00").time():
                    active_trade = False  # Allow a new trade to be opened after 11:00

    def display_signals(self):
        for pair, signals in self.signals.items():
            st.subheader(f"Signals for {pair}")
            for signal in signals:
                st.write(f"Signal Number: {signal['signal_number']}")
                st.write(f"Time: {signal['time']}, Price: {signal['price']}, Sweep: {signal['sweep']}")
                st.write(f"CHOCH Time: {signal['choch_time']}, Entry Price: {signal['entry_price']}")
                st.write(f"Stop Loss: {signal['stop_loss']}, Take Profit: {signal['take_profit']}, Direction: {signal['direction']}")
                st.write(f"Exit Price: {signal['exit_price']} at {signal['exit_time']}")
                st.write("---")

def run_bot():
    pairs = ['EURUSD', 'GBPUSD']  # List of currency pairs
    api_key = st.text_input("Enter your Alpha Vantage API Key:")
    
    if api_key:
        sl_pips = {'EURUSD': 15, 'GBPUSD': 15}  # Stop loss in pips
        pip_values = {'EURUSD': 0.0001, 'GBPUSD': 0.0001}  # Value of a pip
        risk_rewards = {'EURUSD': 1.5, 'GBPUSD': 1.5}  # Risk-reward ratios

        bot = ForexSignalBot(pairs, api_key, sl_pips, pip_values, risk_rewards)

        if st.button("Fetch Data"):
            bot.fetch_data()
            bot.generate_signals()
            bot.display_signals()

# Run the bot in the Streamlit app
if __name__ == "__main__":
    run_bot()
