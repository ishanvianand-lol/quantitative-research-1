````markdown
# Hawkes–VPIN Market Microstructure Dashboard

A Streamlit dashboard to study how **order-flow toxicity (VPIN)** relates to **volatility shocks** and how those shocks **cluster in time** via a **Hawkes self-exciting process**, with technical indicators for price context.

This project combines ideas from **market microstructure**, **stochastic point processes**, and **technical analysis** into a single interactive tool.

---

## Installation

```bash
pip install streamlit yfinance numpy pandas scipy matplotlib mplfinance
```
````

---

## Run

```bash
streamlit run app.py
```

---

## Inputs (Sidebar)

| Parameter           | Meaning                                           |
| ------------------- | ------------------------------------------------- |
| Ticker              | Any Yahoo Finance symbol                          |
| Shock Threshold (k) | Multiplier on rolling volatility to define shocks |
| Rolling Window      | Window for return volatility estimation           |
| VPIN Bucket Volume  | Volume per bucket for VPIN computation            |

---

## Data

- 2-minute bars
- Last 60 days
- OHLCV from Yahoo Finance

---

## Methodology

### 1. Return and Volatility

Returns:

[
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
]

Rolling volatility:

[
\sigma_t = \text{StdDev}(r_{t-w}, \dots, r_t)
]

A **shock/event** is defined when:

[
|r_t| > k \cdot \sigma_t
]

---

### 2. Hawkes Self-Exciting Process

Shock times are modeled as a point process with intensity:

[
\lambda(t) = \mu + \alpha \sum_{t_i < t} e^{-\beta (t - t_i)}
]

Where:

- ( \mu ) — baseline shock rate
- ( \alpha ) — excitation strength
- ( \beta ) — decay rate

**Branching ratio**

[
\frac{\alpha}{\beta}
]

Expected number of shocks triggered by each shock.

Parameters are estimated via **maximum likelihood** using L-BFGS-B.

Log-likelihood:

[
\mathcal{L} = \sum_i \log \lambda(t_i) - \int_0^T \lambda(t),dt
]

---

### 3. VPIN — Volume-Synchronized Probability of Informed Trading

Each bar’s volume is split into estimated buy/sell volume using price location within the candle.

Buckets are filled by fixed volume (V\_{bucket}).

For each bucket:

[
\text{Imbalance} = |V_{buy} - V_{sell}|
]

VPIN over last (N) buckets:

[
VPIN = \frac{\sum_{i=1}^{N} |V_{buy,i} - V_{sell,i}|}{V_{bucket} \times N}
]

Interpretation:

| VPIN    | Meaning                 |
| ------- | ----------------------- |
| < 0.3   | Balanced flow           |
| 0.3–0.5 | Directional pressure    |
| 0.5–0.7 | Informed trading likely |
| > 0.7   | Extreme toxicity        |

---

### 4. Technical Indicators

- **SMA**: ( \text{SMA}\_n = \frac{1}{n} \sum P_t )
- **EMA**: Exponentially weighted mean
- **RSI**:

[
RSI = 100 - \frac{100}{1 + RS}
]

- **Bollinger Bands**

[
\text{Upper} = SMA + k\sigma, \quad
\text{Lower} = SMA - k\sigma
]

- **ATR**: Average true range for volatility

---

## Dashboard Structure

### Tab 1 — Hawkes + VPIN

- Price with shock markers and scaled intensity
- Inter-event time distribution
- VPIN time series with toxicity levels
- VPIN at shocks vs non-shocks
- VPIN vs Hawkes intensity
- Intraday toxicity patterns

### Tab 2 — Technical Analysis

- Candlestick with SMA/EMA and shock markers
- Bollinger Bands with squeeze detection
- RSI with volume
- ATR context

---

## What to Observe

- VPIN tends to rise before clusters of shocks
- Shocks occur near Bollinger extremes and MAs
- RSI extremes + shock → higher reversal probability
- Market open/close show higher toxicity and shock frequency
- High ( \alpha/\beta ) indicates clustered shock regimes

---

## Best Symbols to Try

```
^GSPC, SPY, AAPL, TSLA, NVDA
```
