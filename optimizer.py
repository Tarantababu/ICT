import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForexSignalBot:
    def __init__(self, pairs, api_key, sl_pips, pip_values, risk_rewards, max_risk_per_trade):
        self.pairs = pairs
        self.data = {pair: {} for pair in pairs}
        self.signals = {pair: [] for pair in pairs}
        self.api_key = api_key
        self.sl_pips = sl_pips
        self.pip_values = pip_values
        self.risk_rewards = risk_rewards
        self.max_risk_per_trade = max_risk_per_trade

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
                        df.index = df.index.tz_localize(None)
                        self.data[pair][timeframe] = df
                    else:
                        st.error(f"Failed to fetch data for {pair} at {timeframe} timeframe. Please check your API key and try again.")
                except Exception as e:
                    st.error(f"Error fetching data for {pair} at {timeframe} timeframe: {e}")

    def to_datetime(self, timestamp):
        """Convert any timestamp to Python datetime object."""
        if isinstance(timestamp, pd.Timestamp):
            return timestamp.to_pydatetime()
        elif isinstance(timestamp, datetime):
            return timestamp
        else:
            return pd.to_datetime(timestamp).to_pydatetime()

    def mark_highs_lows(self, pair, date):
        session_start = datetime.combine(date, datetime.min.time())
        session_end = session_start + timedelta(hours=8, minutes=30)
        session_data = self.data[pair]['60min'].loc[session_start:session_end]
        return session_data['High'].max(), session_data['Low'].min()

    def detect_sweep(self, price, high, low):
        if price > high:
            return "High sweep"
        elif price < low:
            return "Low sweep"
        return None

    def detect_market_structure_shift(self, pair, timeframe, start_index, direction):
        data = self.data[pair][timeframe].loc[self.to_datetime(start_index):]
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
        data = self.data[pair][timeframe].loc[self.to_datetime(start_index):]
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

    def detect_trend(self, pair, timeframe, current_time):
        timeframe = '60min' if timeframe == '1h' else timeframe  # Convert '1h' to '60min'
        if timeframe not in self.data[pair]:
            return "No trend"  # Return "No trend" if the timeframe data is not available

        data = self.data[pair][timeframe]
        window = 20  # Use 20 periods for trend detection
        current_index = data.index.get_loc(current_time, method='nearest')
        if current_index < window:
            return "No trend"  # Not enough data for trend detection
        
        ma_short = data['Close'].rolling(window=window//2).mean()
        ma_long = data['Close'].rolling(window=window).mean()
        
        if ma_short.iloc[current_index] > ma_long.iloc[current_index]:
            return "Uptrend"
        elif ma_short.iloc[current_index] < ma_long.iloc[current_index]:
            return "Downtrend"
        else:
            return "No trend"

    def calculate_risk_amount(self, pair, entry_price, stop_loss):
        return abs(entry_price - stop_loss) / self.pip_values[pair]

    def generate_signals(self):
        for pair in self.pairs:
            data_5m = self.data[pair]['5min']
            data_1h = self.data[pair]['60min']  # Changed from '1h' to '60min'
            self.signals[pair] = []
            signal_count = 1
            active_trade = False
            cooldown_period = timedelta(hours=1)
            last_trade_time = self.to_datetime(data_5m.index[0]) - cooldown_period

            for i in range(len(data_5m) - 1):
                current_time = self.to_datetime(data_5m.index[i])
                
                # Only consider trading during specific hours
                trading_start = current_time.replace(hour=8, minute=30, second=0, microsecond=0)
                trading_end = current_time.replace(hour=11, minute=0, second=0, microsecond=0)
                if current_time < trading_start or current_time >= trading_end:
                    continue
                
                # Enforce cooldown period between trades
                if (current_time - last_trade_time) < cooldown_period:
                    continue

                # Check if there's an active trade
                if active_trade:
                    # Check for exit conditions
                    current_price = data_5m.iloc[i]['Close']
                    active_signal = self.signals[pair][-1]
                    
                    if (active_signal['direction'] == 'Long' and current_price >= active_signal['take_profit']) or \
                       (active_signal['direction'] == 'Short' and current_price <= active_signal['take_profit']):
                        active_signal['exit_price'] = active_signal['take_profit']
                        active_signal['exit_time'] = current_time
                        active_trade = False
                        last_trade_time = current_time
                    elif (active_signal['direction'] == 'Long' and current_price <= active_signal['stop_loss']) or \
                         (active_signal['direction'] == 'Short' and current_price >= active_signal['stop_loss']):
                        active_signal['exit_price'] = active_signal['stop_loss']
                        active_signal['exit_time'] = current_time
                        active_trade = False
                        last_trade_time = current_time
                    
                    continue  # Skip to next iteration if there's an active trade

                # Entry criteria
                high, low = self.mark_highs_lows(pair, current_time.date())
                current_price = data_5m.iloc[i]['Close']
                sweep = self.detect_sweep(current_price, high, low)

                if sweep:
                    choch = self.detect_market_structure_shift(pair, '5min', current_time, sweep)
                    if choch:
                        fvg = self.find_fvg(pair, '5min', choch, sweep)
                        if fvg:
                            # Additional entry filters
                            trend = self.detect_trend(pair, '60min', current_time)  # Changed from '1h' to '60min'
                            if (sweep == "High sweep" and trend == "Downtrend") or (sweep == "Low sweep" and trend == "Uptrend"):
                                entry_price = data_5m.iloc[i+1]['Open']  # Enter on next candle open
                                direction = "Short" if sweep == "High sweep" else "Long"
                                stop_loss, take_profit = self.set_stop_loss_and_take_profit(pair, entry_price, fvg, direction)
                                
                                # Risk management
                                risk_amount = self.calculate_risk_amount(pair, entry_price, stop_loss)
                                if risk_amount > self.max_risk_per_trade:
                                    continue  # Skip this trade if risk is too high

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
                                    "exit_price": None,
                                    "exit_time": None,
                                    "risk_amount": risk_amount
                                })
                                signal_count += 1
                                active_trade = True
                                last_trade_time = current_time

            # Close any open trade at the end of the period
            if active_trade:
                active_signal = self.signals[pair][-1]
                last_price = data_5m.iloc[-1]['Close']
                active_signal['exit_price'] = last_price
                active_signal['exit_time'] = self.to_datetime(data_5m.index[-1])

    def run(self):
        self.fetch_data()
        self.generate_signals()
        
        # Calculate pips for each signal
        for pair in self.pairs:
            for signal in self.signals[pair]:
                if signal['exit_price'] is not None:
                    signal['pips'] = calculate_pips(
                        signal['entry_price'],
                        signal['exit_price'],
                        signal['direction'],
                        self.pip_values[pair]
                    )

def calculate_pips(entry_price, exit_price, direction, pip_value):
    if direction == 'Long':
        return (exit_price - entry_price) / pip_value
    else:  # Short
        return (entry_price - exit_price) / pip_value

def calculate_stats(signals, pip_value):
    if not signals:
        return 0, 0, 0, 0

    wins = 0
    total_pips_gained = 0
    total_pips_lost = 0

    for signal in signals:
        pips = signal['pips']
        
        if pips > 0:
            total_pips_gained += pips
            wins += 1
        else:
            total_pips_lost += abs(pips)

    win_rate = (wins / len(signals)) * 100 if signals else 0

    return win_rate, total_pips_gained, total_pips_lost, len(signals)

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
        # Entry point - blue for buy, orange for sell
        entry_color = 'blue' if signal['direction'] == 'Long' else 'orange'
        fig.add_trace(go.Scatter(x=[signal['time']], y=[signal['entry_price']],
                                 mode='markers+text',
                                 marker=dict(symbol='circle', size=8, color=entry_color),
                                 text=[str(signal['signal_number'])],
                                 textposition="top center",
                                 name=f"{signal['direction']} Entry {signal['signal_number']}"))

        # Exit point
        if signal['exit_time'] is not None:
            exit_color = 'green' if signal['pips'] > 0 else 'red'
            fig.add_trace(go.Scatter(x=[signal['exit_time']], y=[signal['exit_price']],
                                     mode='markers+text',
                                     marker=dict(symbol='circle', size=8, color=exit_color),
                                     text=[str(signal['signal_number'])],
                                     textposition="top center",
                                     name=f"Exit {signal['signal_number']}"))

        # Add stop loss and take profit lines
        fig.add_trace(go.Scatter(x=[signal['time'], signal['exit_time'] or data['5min'].index[-1]],
                                 y=[signal['stop_loss'], signal['stop_loss']],
                                 mode='lines',
                                 line=dict(color='red', dash='dash'),
                                 name=f"SL {signal['signal_number']}"))
        
        fig.add_trace(go.Scatter(x=[signal['time'], signal['exit_time'] or data['5min'].index[-1]],
                                 y=[signal['take_profit'], signal['take_profit']],
                                 mode='lines',
                                 line=dict(color='green', dash='dash'),
                                 name=f"TP {signal['signal_number']}"))

    fig.update_layout(title=f'{pair} Chart', xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # Hide weekends
    return fig

def calculate_end_capital(signals, initial_capital, risk_per_trade):
    capital = initial_capital
    for signal in signals:
        if 'pips' not in signal:
            # Calculate pips if not already present
            signal['pips'] = calculate_pips(signal['entry_price'], signal['exit_price'], signal['direction'], signal['risk_amount'] / (risk_per_trade * capital))
        
        pips = signal['pips']
        risk_amount = min(capital * risk_per_trade, signal['risk_amount'])
        pip_value = risk_amount / abs(signal['entry_price'] - signal['stop_loss'])
        pnl = pips * pip_value
        capital += pnl
    return capital

def optimize_parameters(bot, pair, sl_range, rr_range, initial_capital, risk_per_trade):
    results = []
    
    for sl_pips in sl_range:
        for risk_reward in rr_range:
            bot.sl_pips[pair] = sl_pips
            bot.risk_rewards[pair] = risk_reward
            
            bot.generate_signals()
            
            # Calculate pips for each signal
            for signal in bot.signals[pair]:
                if signal['exit_price'] is not None:
                    signal['pips'] = calculate_pips(
                        signal['entry_price'],
                        signal['exit_price'],
                        signal['direction'],
                        bot.pip_values[pair]
                    )
            
            end_capital = calculate_end_capital(bot.signals[pair], initial_capital, risk_per_trade)
            
            results.append({
                'SL (pips)': sl_pips,
                'Risk-Reward': risk_reward,
                'End Capital': end_capital
            })
    
    return pd.DataFrame(results)

def main():
    st.title('Advanced Forex Signal Bot with Optimizer')

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

    max_risk_per_trade = st.sidebar.number_input('Maximum Risk per Trade (in pips):', min_value=1, max_value=100, value=20)

    if not api_key:
        st.warning('Please enter your Alpha Vantage API key to proceed.')
        return

    bot = ForexSignalBot(pairs, api_key, sl_pips, pip_values, risk_rewards, max_risk_per_trade)

    # Optimization settings
    st.sidebar.header('Optimization Settings')
    optimize = st.sidebar.checkbox('Enable Optimization')
    if optimize:
        pair_to_optimize = st.sidebar.selectbox('Select pair to optimize:', pairs)
        initial_capital = st.sidebar.number_input('Initial Capital:', min_value=100, value=10000)
        risk_per_trade = st.sidebar.slider('Risk per trade (%):', min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100

        sl_min = st.sidebar.number_input('Minimum Stop Loss (pips):', min_value=5, value=10)
        sl_max = st.sidebar.number_input('Maximum Stop Loss (pips):', min_value=sl_min + 5, value=50)
        sl_step = st.sidebar.number_input('Stop Loss Step:', min_value=1, value=5)

        rr_min = st.sidebar.number_input('Minimum Risk-Reward:', min_value=0.5, value=1.0, step=0.1)
        rr_max = st.sidebar.number_input('Maximum Risk-Reward:', min_value=rr_min + 0.5, value=3.0, step=0.1)
        rr_step = st.sidebar.number_input('Risk-Reward Step:', min_value=0.1, value=0.2, step=0.1)

    if st.button('Generate Signals' if not optimize else 'Optimize and Generate Signals'):
        with st.spinner('Fetching data and generating signals...'):
            bot.run()

        if optimize:
            st.header('Optimization Results')
            sl_range = np.arange(sl_min, sl_max + sl_step, sl_step)
            rr_range = np.arange(rr_min, rr_max + rr_step, rr_step)

            results_df = optimize_parameters(bot, pair_to_optimize, sl_range, rr_range, initial_capital, risk_per_trade)

            st.dataframe(results_df)

            best_result = results_df.loc[results_df['End Capital'].idxmax()]
            st.success(f"Best parameters found for {pair_to_optimize}:\n"
                       f"Stop Loss: {best_result['SL (pips)']:.2f} pips\n"
                       f"Risk-Reward: {best_result['Risk-Reward']:.2f}\n"
                       f"End Capital: ${best_result['End Capital']:.2f}")

            st.header('Results Visualization')
            pivot_table = results_df.pivot(index='SL (pips)', columns='Risk-Reward', values='End Capital')
            fig = go.Figure(data=[go.Surface(z=pivot_table.values, x=pivot_table.columns, y=pivot_table.index)])
            fig.update_layout(title='Optimization Results', autosize=False, width=800, height=600,
                              scene=dict(xaxis_title='Risk-Reward', yaxis_title='SL (pips)', zaxis_title='End Capital'))
            st.plotly_chart(fig)

        # Display active signals
        st.header('Active Signals')
        active_signals_exist = False
        for pair in pairs:
            active_signals = [signal for signal in bot.signals[pair] if signal['exit_price'] is None]
            if active_signals:
                active_signals_exist = True
                for signal in active_signals:
                    direction = signal['direction']
                    arrow = "🔼" if direction == "Long" else "🔽"
                    st.write(f"{pair} - {direction}{arrow} - entry: {signal['entry_price']:.5f}, "
                             f"sl: {signal['stop_loss']:.5f}, tp: {signal['take_profit']:.5f}")
        
        if not active_signals_exist:
            st.info("No active signals at the moment.")

        # Display signal statistics
        st.header('Signal Statistics')
        for pair in pairs:
            if bot.signals[pair]:
                win_rate, total_pips_gained, total_pips_lost, total_signals = calculate_stats(bot.signals[pair], bot.pip_values[pair])
                st.subheader(f'Statistics for {pair}')
                st.write(f"Total Signals: {total_signals}")
                st.write(f"Win Rate: {win_rate:.2f}%")
                st.write(f"Total Pips Gained: {total_pips_gained:.2f}")
                st.write(f"Total Pips Lost: {total_pips_lost:.2f}")
                st.write(f"Net Pips: {total_pips_gained - total_pips_lost:.2f}")
                
                # Display individual signal details
                st.write("Individual Signal Details:")
                for signal in bot.signals[pair]:
                    st.write(f"Signal {signal['signal_number']}: {signal['direction']} - "
                             f"Entry: {signal['entry_price']:.5f}, "
                             f"Exit: {signal['exit_price']:.5f}, "
                             f"Pips: {signal['pips']:.2f}")
                
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
