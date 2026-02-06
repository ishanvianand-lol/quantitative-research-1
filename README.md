📈 Hawkes–VPIN Market Microstructure Analyzer
A Streamlit-based quantitative trading and market microstructure dashboard that combines:

Hawkes Self-Exciting Point Process for shock/event modeling

VPIN (Volume-Synchronized Probability of Informed Trading) for order-flow toxicity

Classical Technical Indicators (SMA, EMA, RSI, Bollinger Bands, ATR)

Candlestick + Volume + Momentum visual analytics

This tool is designed to study how informed trading (VPIN) correlates with self-exciting price shocks (Hawkes) and how both interact with traditional technical signals.

This is not a simple TA dashboard — it is a microstructure + stochastic process + technical analysis fusion system.

🧠 Core Idea
Markets do not move randomly. They move because:

Informed traders create toxic order flow → captured by VPIN

This toxicity creates price shocks → detected via return volatility threshold

Shocks are self-exciting (clustered in time) → modeled via Hawkes Process

These events occur at meaningful technical levels → analyzed via TA indicators

This dashboard proves and visualizes that relationship.

⚙️ Features
🔬 Hawkes Process Modeling

Detects return shocks using volatility threshold

Fits Hawkes parameters: μ, α, β

Computes live intensity λ(t)

Shows clustering of events and expected time to next shock

Branching ratio (α/β) interpretation

🧪 VPIN Toxicity Engine

Implements full VPIN bucket logic from microstructure literature

Adaptive bucket sizing based on average volume

Toxicity thresholds (0.3, 0.5, 0.7)

Correlates VPIN values with Hawkes shock events

Intraday VPIN vs shock distribution

📊 Technical Analysis Suite

Candlestick chart with SMA20, SMA50, EMA12

Bollinger Bands with squeeze detection

RSI momentum panel with volume

ATR volatility measure

Automatic interpretation of trend, volatility, and momentum

🔗 Cross-Analysis (unique part)

VPIN distribution at shock vs non-shock periods

VPIN vs Hawkes intensity scatter

Shock locations on all charts

Microstructure patterns by hour of day

🏗️ Architecture
ComponentPurposeyfinanceIntraday data (2m bars, 60 days)VPIN engineOrder flow imbalance bucketsShock detectorVolatility-scaled return thresholdHawkes fittingL-BFGS MLE for μ, α, βmplfinanceProfessional candlestick plottingStreamlitInteractive dashboard

📦 Installation
pip install streamlit yfinance numpy pandas matplotlib scipy mplfinance

▶️ Run the App
streamlit run app.py

🖥️ Controls (Sidebar)
ControlMeaningTicker SymbolAny Yahoo Finance symbol (e.g. ^GSPC, AAPL, TSLA)Shock Threshold (k)Multiplier on rolling volatility to define shockRolling WindowWindow for return volatilityVPIN Bucket VolumeVolume per bucket for toxicity calculation

📐 Mathematical Models
Hawkes Process
λ(t)=μ+α∑ti<te−β(t−ti)\lambda(t) = \mu + \alpha \sum\_{t_i < t} e^{-\beta (t - t_i)}λ(t)=μ+αti​<t∑​e−β(t−ti​)

μ → baseline shock rate

α → excitation strength

β → decay rate

α/β → expected number of shocks triggered by each shock

VPIN
VPIN=∑∣Vbuy−Vsell∣Vbucket×NVPIN = \frac{\sum |V*{buy} - V*{sell}|}{V\_{bucket} \times N}VPIN=Vbucket​×N∑∣Vbuy​−Vsell​∣​
Measures order flow toxicity from volume imbalance.

🧪 What This Dashboard Demonstrates
You will observe:

VPIN rises before clusters of Hawkes shocks

Shocks occur at Bollinger extremes and MA levels

RSI extremes + Hawkes spike → high reversal probability

Intraday open/close show highest toxicity and shock frequency

Low VPIN periods correspond to liquidity provider dominance

📊 Tabs Overview
Tab 1 — Hawkes + VPIN Microstructure

Price with shocks and scaled intensity

Inter-event time histogram

VPIN toxicity plot

VPIN vs shocks statistical comparison

Intraday microstructure patterns

Tab 2 — Technical Analysis

Candlestick + moving averages + shock markers

Bollinger Bands strategy

RSI momentum panel with volume

Automated interpretation of current market state

🧭 Interpretation Guide
ConditionMeaningHigh VPIN + Rising λ(t)Informed traders driving shock clusterShock at Upper BB + RSI > 70Reversal short setupShock at Lower BB + RSI < 30Reversal long setupBB Squeeze + VPIN spikeBreakout brewingα/β > 1Highly self-exciting market regime

🧩 Use Cases

Quant research in market microstructure

Studying informed trading impact

Event clustering analysis

Strategy validation around technical levels

Educational tool for Hawkes processes in finance

⚠️ Notes

Uses 2-minute intraday data → computationally heavy

VPIN bucket size auto-adjusts to symbol volume

Works best on liquid instruments (indices, large caps)

Not financial advice — research/educational tool

📚 Concepts Referenced

Hawkes (1971) self-exciting processes

VPIN (Easley, López de Prado)

Market microstructure theory

Volatility clustering

Technical analysis

✅ Example Symbols to Try
^GSPC (S&P 500)
AAPL
TSLA
NVDA
SPY

🚀 Outcome
This project bridges:

Stochastic processes + Order flow theory + Technical analysis
into one unified, visual, interactive system.

It shows why markets move — not just how they look.
