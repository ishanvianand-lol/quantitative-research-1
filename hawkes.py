import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st
import mplfinance as mpf

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, span):
    return data.ewm(span=span, adjust=False).mean()

def calculate_rsi(data, period=14):
    close = data['Close'] 
    delta = close.diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods = period).mean()
    avg_loss = loss.rolling(window=period, min_periods = period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    middle = calculate_sma(data=data['Close'], window=window)
    std = data['Close'].rolling(window=window).std()
    lower = middle - (std * num_std)
    upper = middle + (std * num_std)
    return lower, middle, upper

def calculate_atr(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    return atr

def calculate_ichimoku(high, low, close):
    """Ichimoku Cloud Components"""
    # HINT: You need 5 lines
    # 1. Tenkan-sen (Conversion Line) = (9-period high + 9-period low)/2
    # 2. Kijun-sen (Base Line) = (26-period high + 26-period low)/2
    # 3. Senkou Span A = (Tenkan + Kijun)/2, shifted 26 periods forward
    # 4. Senkou Span B = (52-period high + 52-period low)/2, shifted 26 periods forward
    # 5. Chikou Span = Close, shifted 26 periods backward
    pass  # YOU IMPLEMENT THIS

def Hawkes_Process(parameters, events, T):
    mu, alpha, beta = parameters

    if mu <= 0 or alpha <= 0 or beta <= 0:
        return np.inf
    
    n = len(events)
    intensity_sum = 0.0

    for i, ti in enumerate(events):
        history = events[:i]

        lam = mu + alpha * np.sum(np.exp(-beta * (ti - history)))
        intensity_sum += np.log(lam)

    cmpst = mu * T

    for ti in events:
        cmpst += (alpha/beta) * (1 - np.exp(-beta * (T - ti)))
    
    return -(intensity_sum - cmpst)

def Hawkes_Fitting(events):
    T = events[-1]

    initial_estimation = [0.01, 0.05, 0.2]

    res = minimize(
        Hawkes_Process,
        initial_estimation,
        args=(events, T),
        method='L-BFGS-B',
        bounds=[(1e-6, None), (1e-6, None), (1e-6, None)]
    )

    return res.x

st.set_page_config(layout="wide")

with st.sidebar:
    symbol = st.text_input("Ticker Symbol", value="^GSPC").upper()
    k = st.slider("Shock Threshold", 1.0, 5.0, 2.0)
    rolling_window = st.number_input("Rolling Window for Return Volatility", 20, 100, 50)

tab1, tab2 = st.tabs(["Hawkes Fitting with Toxic Flow", "Technical Indicators"])

sp500 = yf.download(tickers = symbol,
                    period = '60d',
                    interval = '2m',
                    auto_adjust = True)
sp500.columns = sp500.columns.get_level_values(0)

sp500 = sp500.dropna()

sp500["Returns"] = sp500["Close"].pct_change()
sp500["Volatility"] = sp500["Returns"].rolling(rolling_window).std() 

sp500["Shock/Event"] = (
    np.abs(sp500["Returns"]) > k * sp500["Volatility"]
    ).astype(int)

event_times = np.where(sp500["Shock/Event"] == 1)[0]
print("Total shocks:", len(event_times))
print("Shock %:", len(event_times) / len(sp500) * 100)

shock_indices = np.where(sp500["Shock/Event"] == 1)[0]

event_times = event_times.astype(float)
event_times = event_times - event_times.min()

print("First 10 event times:", event_times[:10])
print("Last event time:", event_times[-1])
print("Total events:", len(event_times))

mu, alpha, beta = Hawkes_Fitting(event_times)

lambda_t = np.zeros(len(sp500))
event_times_float = np.where(sp500["Shock/Event"] == 1)[0].astype(float)

for t in range(len(sp500)):
    past_events = event_times_float[event_times_float < t]
    lambda_t[t] = mu + alpha * np.sum(np.exp(-beta * (t - past_events)))


with tab1:
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax1 = plt.subplots(figsize=(14,6))

        ax1.plot(sp500.index, 
                sp500['Close'], 
                color='black', 
                label='Close Price')

        ax1.scatter(sp500.index[sp500['Shock/Event'] == 1], 
                    sp500['Close'][sp500['Shock/Event'] == 1], 
                    color='red', 
                    label='Shocks', 
                    zorder=5)

        scale = sp500['Close'].max() / lambda_t.max()
        ax1.plot(sp500.index, 
                lambda_t * scale, 
                color='blue',
                label='Hawkes Intensity Scaled')

        ax1.set_title(f"Hawkes Process: {symbol}")
        ax1.set_ylabel("Price / Scaled Intensity")
        ax1.set_xlabel("Date")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)
    with col2:
        if len(event_times) > 1:
            inter_event_times = np.diff(event_times)
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.hist(inter_event_times, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(1/mu, color='red', linestyle='--', label=f'Expected (1/μ = {1/mu:.1f})')
            ax2.set_xlabel("Time Between Shocks")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Inter-Event Time Distribution")
            ax2.legend()
            st.pyplot(fig2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("μ (baseline)", f"{mu:.4f}")
    col2.metric("α (excitation)", f"{alpha:.4f}")
    col3.metric("β (decay)", f"{beta:.4f}")
    col4.metric("Branching Ratio (α/β)", f"{alpha/beta:.4f}", 
            help="Expected number of shocks triggered by each shock")

    col1, col2 = st.columns(2)
    current_intensity = lambda_t[-1]
    expected_time_to_next = 1 / current_intensity if current_intensity > 0 else np.inf
    col1.metric("Current intensity", f"{current_intensity:.4f}")
    col2.metric("Expected time to next shock (periods)", f"{expected_time_to_next:.1f}")

with tab2:
    st.header("📈 Technical Analysis")

    chart_data = sp500[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    chart_data['SMA_20'] = calculate_sma(sp500['Close'], 20)
    chart_data['SMA_50'] = calculate_sma(sp500['Close'], 50)
    chart_data['EMA_12'] = calculate_ema(sp500['Close'], 12)
    chart_data['RSI'] = calculate_rsi(sp500)

    lower, middle, upper = calculate_bollinger_bands(sp500)

    chart_data['BB_upper'] = upper
    chart_data['BB_middle'] = middle
    chart_data['BB_lower'] = lower
    chart_data['ATR'] = calculate_atr(sp500)

        
    # Mark Hawkes shocks
    chart_data['Hawkes_Shock'] = sp500['Shock/Event']

    # ==================== CHART 1: CANDLESTICK + MOVING AVERAGES ====================
    st.subheader("🕯️ Candlestick Chart with Moving Averages")

    # Define additional plots (overlays)
    apds = [
        mpf.make_addplot(chart_data['SMA_20'], color='blue', width=1.5, label='SMA 20'),
        mpf.make_addplot(chart_data['SMA_50'], color='red', width=1.5, label='SMA 50'),
        mpf.make_addplot(chart_data['EMA_12'], color='green', width=1, linestyle='--', label='EMA 12'),
    ]

    # Mark Hawkes shocks on price chart
    shock_points = chart_data['Close'].copy()
    shock_points[chart_data['Hawkes_Shock'] == 0] = np.nan
    apds.append(mpf.make_addplot(shock_points, type='scatter', markersize=50, 
                                marker='v', color='red', label='Hawkes Shock'))

    # Create custom style
    mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', 
                            wick='inherit', volume='in', alpha=0.8)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray', 
                        facecolor='white', figcolor='white')

    # Plot
    fig1, axes = mpf.plot(chart_data, type='candle', style=s, addplot=apds,
                        volume=True, figsize=(16, 10), 
                        title='Price Action with Moving Averages & Hawkes Shocks',
                        returnfig=True, warn_too_much_data=10000)

    st.pyplot(fig1)

    # ==================== INTERPRETATION ====================
    st.markdown("### 📊 Moving Average Interpretation")

    current_price = chart_data['Close'].iloc[-1]
    current_sma20 = chart_data['SMA_20'].iloc[-1]
    current_sma50 = chart_data['SMA_50'].iloc[-1]
    current_ema12 = chart_data['EMA_12'].iloc[-1]

    price_vs_sma20 = ((current_price / current_sma20 - 1) * 100)
    price_vs_sma50 = ((current_price / current_sma50 - 1) * 100)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Price vs SMA20", f"{price_vs_sma20:+.2f}%")
    with col2:
        st.metric("Price vs SMA50", f"{price_vs_sma50:+.2f}%")
    with col3:
        trend = "🟢 Bullish" if current_sma20 > current_sma50 else "🔴 Bearish"
        st.metric("Trend", trend)

    st.write(f"""
    **Current Setup:**
    - Price: **${current_price:.2f}**
    - SMA20: **${current_sma20:.2f}** ({price_vs_sma20:+.2f}%)
    - SMA50: **${current_sma50:.2f}** ({price_vs_sma50:+.2f}%)

    **Trend Analysis:**
    {
        '''🔥 **STRONG UPTREND**
    - Price > EMA12 > SMA20 > SMA50 (perfect bull alignment)
    - All MAs rising
    - Strategy: Buy dips to moving averages''' 
        if current_price > current_ema12 > current_sma20 > current_sma50
        else '''❄️ **STRONG DOWNTREND**
    - Price < EMA12 < SMA20 < SMA50 (perfect bear alignment)
    - All MAs falling
    - Strategy: Sell rallies to moving averages'''
        if current_price < current_ema12 < current_sma20 < current_sma50
        else '''⚠️ **CHOPPY/RANGING**
    - MAs not aligned properly
    - Price whipsawing through MAs
    - Strategy: Wait for clear breakout or use range-bound tactics'''
    }

    **Hawkes Shock Analysis:**
    - Recent shocks near MAs suggest: {'Support holding' if price_vs_sma20 > -2 and price_vs_sma20 < 2 else 'Breakout/breakdown in progress'}
    """)

    # ==================== CHART 2: BOLLINGER BANDS ====================
    st.subheader("📊 Bollinger Bands Analysis")

    # Prepare Bollinger Bands
    apds_bb = [
        mpf.make_addplot(chart_data['BB_upper'], color='red', width=1, linestyle='--', label='Upper Band'),
        mpf.make_addplot(chart_data['BB_middle'], color='blue', width=1.5, label='Middle (SMA20)'),
        mpf.make_addplot(chart_data['BB_lower'], color='green', width=1, linestyle='--', label='Lower Band'),
    ]

    # Fill between bands
    # Note: mplfinance doesn't support fill_between directly, so we'll add custom panel
    apds_bb.append(mpf.make_addplot(shock_points, type='scatter', markersize=50,
                                    marker='v', color='red', label='Hawkes Shock'))

    fig2, axes = mpf.plot(chart_data, type='candle', style=s, addplot=apds_bb,
                        volume=True, figsize=(16, 10),
                        title='Bollinger Bands Strategy',
                        returnfig=True, warn_too_much_data=10000)

    st.pyplot(fig2)

    # ==================== BB INTERPRETATION ====================
    st.markdown("### 📊 Bollinger Bands Interpretation")

    current_upper = chart_data['BB_upper'].iloc[-1]
    current_lower = chart_data['BB_lower'].iloc[-1]
    current_middle = chart_data['BB_middle'].iloc[-1]

    bb_width = current_upper - current_lower
    bb_width_pct = (bb_width / current_middle) * 100
    bb_position = (current_price - current_lower) / (current_upper - current_lower) * 100

    # Check for squeeze
    bb_width_ma = (chart_data['BB_upper'] - chart_data['BB_lower']).rolling(50).mean().iloc[-1]
    is_squeeze = bb_width < bb_width_ma * 0.7

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("BB Width", f"${bb_width:.2f}", delta=f"{bb_width_pct:.1f}%")
    with col2:
        st.metric("BB Position", f"{bb_position:.1f}%")
    with col3:
        st.metric("Upper Band", f"${current_upper:.2f}")
    with col4:
        st.metric("Lower Band", f"${current_lower:.2f}")

    st.write(f"""
    **Bollinger Band Analysis:**

    **Current Position:**
    - Price is at **{bb_position:.1f}%** of band range
    - {
        '🔴 Near upper band - Potentially overbought' if bb_position > 80
        else '🟢 Near lower band - Potentially oversold' if bb_position < 20
        else '⚪ Mid-range - No extreme'
    }

    **Volatility State:**
    {
        '''⚠️ **BOLLINGER SQUEEZE DETECTED!**
    - Bands very narrow relative to average
    - Low volatility environment
    - **BIG MOVE BREWING** - direction unknown
    - Wait for breakout above/below bands for direction'''
        if is_squeeze
        else f'''📊 **NORMAL VOLATILITY**
    - Band width: {bb_width_pct:.1f}% of price
    - {'Expanding' if bb_width > bb_width_ma else 'Contracting'} relative to 50-period average'''
    }

    **Trading Setup:**
    - Upper band touch + Hawkes shock → Potential reversal short
    - Lower band touch + Hawkes shock → Potential reversal long
    - Squeeze + Hawkes spike → Breakout confirmation
    - Walk along upper band = strong uptrend (don't short!)
    - Walk along lower band = strong downtrend (don't buy!)
    """)

    # ==================== CHART 3: RSI + MACD ====================
    st.subheader("⚡ Momentum Indicators (RSI)")

    # Create RSI panel
    fig3 = mpf.figure(figsize=(16, 10), style=s)

    # Top panel: Price
    ax1 = fig3.add_subplot(3, 1, 1)
    ax1.plot(chart_data.index, chart_data['Close'], color='black', linewidth=1.5, label='Close')
    ax1.scatter(chart_data.index[chart_data['Hawkes_Shock'] == 1],
                chart_data['Close'][chart_data['Hawkes_Shock'] == 1],
                color='red', s=50, zorder=5, label='Hawkes Shock', alpha=0.7)
    ax1.set_ylabel('Price', fontsize=10)
    ax1.set_title('Price with Hawkes Shocks', fontweight='bold', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Middle panel: RSI
    ax2 = fig3.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(chart_data.index, chart_data['RSI'], color='blue', linewidth=1.5, label='RSI(14)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Overbought')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Oversold')
    ax2.axhline(50, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax2.fill_between(chart_data.index, 30, 70, alpha=0.1, color='gray')
    ax2.set_ylabel('RSI', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title('RSI - Relative Strength Index', fontweight='bold', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Bottom panel: Volume
    ax3 = fig3.add_subplot(3, 1, 3, sharex=ax1)
    colors = ['green' if chart_data['Close'].iloc[i] >= chart_data['Open'].iloc[i] else 'red' 
            for i in range(len(chart_data))]
    ax3.bar(chart_data.index, chart_data['Volume'], color=colors, alpha=0.6, width=0.0008)
    ax3.set_ylabel('Volume', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_title('Volume', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)

    # ==================== RSI INTERPRETATION ====================
    st.markdown("### 📊 RSI Interpretation")

    current_rsi = chart_data['RSI'].iloc[-1]
    prev_rsi = chart_data['RSI'].iloc[-2]
    rsi_ma = chart_data['RSI'].rolling(20).mean().iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current RSI", f"{current_rsi:.2f}", delta=f"{current_rsi - prev_rsi:.2f}")
    with col2:
        st.metric("RSI 20-MA", f"{rsi_ma:.2f}")
    with col3:
        condition = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "⚪ Neutral"
        st.metric("Condition", condition)
    with col4:
        trend = "📈 Bullish" if current_rsi > 50 else "📉 Bearish"
        st.metric("Momentum", trend)

    st.write(f"""
    **RSI Analysis:**
    - Current RSI: **{current_rsi:.2f}**
    - {
        '🔴 **OVERBOUGHT** - RSI > 70, pullback likely' if current_rsi > 70
        else '🟢 **OVERSOLD** - RSI < 30, bounce likely' if current_rsi < 30
        else '⚪ **NEUTRAL ZONE**' if 45 <= current_rsi <= 55
        else '📈 Bullish momentum (50-70)' if current_rsi > 50
        else '📉 Bearish momentum (30-50)'
    }

    **Momentum State:**
    - RSI {'rising' if current_rsi > prev_rsi else 'falling'} - momentum {'building' if abs(current_rsi - prev_rsi) > 2 else 'steady'}
    - RSI vs 20-MA: {'Above (bullish)' if current_rsi > rsi_ma else 'Below (bearish)'}

    **Hawkes-RSI Correlation:**
    - When Hawkes shocks occur at RSI extremes (>70 or <30) → High reversal probability
    - Current setup: {'⚠️ Shock + extreme RSI = reversal setup' if (current_rsi > 70 or current_rsi < 30) and lambda_t[-1] > mu*2 else '✅ Normal conditions'}
    """)