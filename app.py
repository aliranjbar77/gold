import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator

# --- Page Config ---
st.set_page_config(page_title="AI Trading Terminal", layout="wide")
st.markdown(
    """
    <style>
        .stApp { background-color: #0e1117; color: white; }
        .metric-card { background-color: #1e2130; padding: 12px; border-radius: 10px; border: 1px solid #30334e; }
        .signal-buy { background-color: rgba(0,255,0,0.12); border-left: 5px solid #00ff00; padding: 12px; }
        .signal-sell { background-color: rgba(255,0,0,0.12); border-left: 5px solid #ff0000; padding: 12px; }
        .signal-neutral { background-color: rgba(255,255,255,0.06); border-left: 5px solid #aaaaaa; padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helpers ---
@st.cache_data(ttl=300)
def get_data(symbol: str, period: str, interval: str):
    data = yf.download(symbol, period=period, interval=interval)
    dxy = yf.download("DX-Y.NYB", period=period, interval=interval)

    # Fix MultiIndex columns for newer pandas versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if isinstance(dxy.columns, pd.MultiIndex):
        dxy.columns = dxy.columns.get_level_values(0)
    return data, dxy

@st.cache_data(ttl=300)
def get_higher_timeframe(symbol: str, base_interval: str):
    mapping = {"1h": "4h", "4h": "1d", "1d": "1wk"}
    higher = mapping.get(base_interval, "1d")

    period = "120d" if higher in ["1h", "4h"] else "5y"
    data = yf.download(symbol, period=period, interval=higher)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data, higher

def calculate_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close_s = df["Close"].squeeze()
    open_s = df["Open"].squeeze()
    high_s = df["High"].squeeze()
    low_s = df["Low"].squeeze()

    df["Body"] = (open_s - close_s).abs()
    df["Wick_Upper"] = high_s - df[["Open", "Close"]].max(axis=1).squeeze()
    df["Wick_Lower"] = df[["Open", "Close"]].min(axis=1).squeeze() - low_s

    df["Is_Doji"] = df["Body"] <= (high_s - low_s) * 0.1
    df["Is_Hammer"] = (df["Wick_Lower"] > df["Body"] * 2) & (df["Wick_Upper"] < df["Body"] * 0.5)
    return df

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).fillna(0).cumsum()

# --- i18n ---
TEXT = {
    "en": {
        "settings": "Settings",
        "select_asset": "Select Asset",
        "gold": "Gold",
        "silver": "Silver",
        "timeframe": "Timeframe",
        "risk_mgmt": "Risk Management",
        "balance": "Balance ($)",
        "risk_pct": "Risk (%)",
        "atr_mult": "ATR Multiplier (SL)",
        "rr": "Risk/Reward",
        "contract_size": "Contract Size",
        "contract_units": "Contract size (units)",
        "live_update": "Live Update",
        "auto_refresh": "Auto-refresh",
        "refresh_interval": "Refresh interval (sec)",
        "title": "AI Trading Terminal",
        "price": "Price",
        "rsi": "RSI",
        "atr": "ATR",
        "dxy": "DXY",
        "corr": "Corr",
        "signal_buy": "Signal: BUY",
        "signal_sell": "Signal: SELL",
        "signal_wait": "Signal: WAIT",
        "risk": "Risk",
        "risk_amount": "Risk amount",
        "lot_size": "Lot size",
        "targets": "Targets",
        "tp": "TP",
        "sl": "SL",
        "rr_fmt": "R:R = 1:{rr}",
        "logic": "Logic Breakdown",
        "higher_tf": "Higher timeframe",
        "trend": "Trend",
        "last_update": "Last update",
        "data_fail": "Data fetch failed! Check your internet connection.",
        "lang": "Language / زبان",
    },
    "fa": {
        "settings": "تنظیمات",
        "select_asset": "انتخاب دارایی",
        "gold": "طلا",
        "silver": "نقره",
        "timeframe": "تایم‌فریم",
        "risk_mgmt": "مدیریت ریسک",
        "balance": "موجودی ($)",
        "risk_pct": "ریسک (%)",
        "atr_mult": "ضریب ATR (حد ضرر)",
        "rr": "ریسک/سود",
        "contract_size": "اندازه قرارداد",
        "contract_units": "اندازه قرارداد (واحد)",
        "live_update": "آپدیت زنده",
        "auto_refresh": "آپدیت خودکار",
        "refresh_interval": "فاصله آپدیت (ثانیه)",
        "title": "ترمینال معامله‌گری هوشمند",
        "price": "قیمت",
        "rsi": "RSI",
        "atr": "ATR",
        "dxy": "DXY",
        "corr": "همبستگی",
        "signal_buy": "سیگنال: خرید",
        "signal_sell": "سیگنال: فروش",
        "signal_wait": "سیگنال: صبر",
        "risk": "ریسک",
        "risk_amount": "میزان ریسک",
        "lot_size": "اندازه لات",
        "targets": "اهداف",
        "tp": "حد سود",
        "sl": "حد ضرر",
        "rr_fmt": "ریسک/سود = 1:{rr}",
        "logic": "جزئیات منطق",
        "higher_tf": "تایم‌فریم بالاتر",
        "trend": "روند",
        "last_update": "آخرین آپدیت",
        "data_fail": "دریافت داده ناموفق بود! اتصال اینترنت را بررسی کنید.",
        "lang": "Language / زبان",
    },
}

lang_choice = st.sidebar.selectbox(TEXT["en"]["lang"], ["فارسی", "English"], index=0)
lang = "fa" if lang_choice == "فارسی" else "en"
T = TEXT[lang]

# --- Sidebar ---
st.sidebar.title(T["settings"])
asset_name = st.sidebar.selectbox(
    T["select_asset"],
    ["GC=F", "SI=F"],
    format_func=lambda x: T["gold"] if x == "GC=F" else T["silver"],
)
timeframe = st.sidebar.selectbox(T["timeframe"], ["1h", "4h", "1d"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader(T["risk_mgmt"])
acc_balance = st.sidebar.number_input(T["balance"], value=1000)
risk_pct = st.sidebar.slider(T["risk_pct"], 0.5, 5.0, 2.0)
atr_mult = st.sidebar.slider(T["atr_mult"], 1.0, 4.0, 2.0, 0.5)
rr_ratio = st.sidebar.slider(T["rr"], 1.0, 5.0, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader(T["contract_size"])
contract_defaults = {"GC=F": 100.0, "SI=F": 5000.0}
contract_size = st.sidebar.number_input(
    T["contract_units"],
    min_value=1.0,
    value=contract_defaults.get(asset_name, 100.0),
)

st.sidebar.markdown("---")
st.sidebar.subheader(T["live_update"])
auto_refresh = st.sidebar.checkbox(T["auto_refresh"])
refresh_sec = st.sidebar.slider(T["refresh_interval"], 5, 120, 30, 5)

# --- Main Logic ---
df, dxy = get_data(asset_name, "120d" if timeframe in ["1h", "4h"] else "5y", timeframe)

if not df.empty:
    df = calculate_patterns(df)

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze() if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    rsi = RSIIndicator(close).rsi()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    atr = AverageTrueRange(high, low, close).average_true_range()
    bb = BollingerBands(close)
    obv = calc_obv(close, volume)

    curr_price = float(close.iloc[-1])
    curr_rsi = float(rsi.iloc[-1])
    curr_atr = float(atr.iloc[-1])

    ema50_slope = float(ema50.iloc[-1] - ema50.iloc[-2]) if len(ema50.dropna()) > 2 else 0.0
    obv_slope = float(obv.iloc[-1] - obv.iloc[-2]) if len(obv.dropna()) > 2 else 0.0

    # Higher timeframe confirmation
    df_higher, higher_tf = get_higher_timeframe(asset_name, timeframe)
    ht_trend = None
    if not df_higher.empty and "Close" in df_higher.columns:
        ht_close = df_higher["Close"].squeeze()
        ht_ema50 = EMAIndicator(ht_close, window=50).ema_indicator()
        if len(ht_ema50.dropna()) > 0:
            ht_trend = "UP" if ht_close.iloc[-1] > ht_ema50.iloc[-1] else "DOWN"

    # Correlation with DXY
    correlation = 0.0
    curr_dxy = 0.0
    if not dxy.empty and "Close" in dxy.columns:
        curr_dxy = float(dxy["Close"].iloc[-1])
        common_idx = df.index.intersection(dxy.index)
        if len(common_idx) > 3:
            correlation = float(df.loc[common_idx]["Close"].corr(dxy.loc[common_idx]["Close"]))

    # --- Signal Logic ---
    score = 0
    reasons = []

    if curr_price > ema50.iloc[-1]:
        score += 1
        reasons.append("Price above EMA50")
    else:
        score -= 1
        reasons.append("Price below EMA50")

    if curr_price > ema200.iloc[-1]:
        score += 1
        reasons.append("Price above EMA200")
    else:
        score -= 1
        reasons.append("Price below EMA200")

    if ema50_slope > 0:
        score += 1
        reasons.append("EMA50 slope positive")
    else:
        score -= 1
        reasons.append("EMA50 slope negative")

    if curr_rsi < 35:
        score += 1
        reasons.append("RSI oversold")
    elif curr_rsi > 65:
        score -= 1
        reasons.append("RSI overbought")

    if obv_slope > 0:
        score += 1
        reasons.append("OBV rising")
    else:
        score -= 1
        reasons.append("OBV falling")

    if ht_trend == "UP":
        score += 1
        reasons.append(f"Higher TF ({higher_tf}) uptrend")
    elif ht_trend == "DOWN":
        score -= 1
        reasons.append(f"Higher TF ({higher_tf}) downtrend")

    signal = "NEUTRAL"
    if score >= 2:
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"

    # --- Risk Management ---
    sl = curr_price - (atr_mult * curr_atr) if signal == "BUY" else curr_price + (atr_mult * curr_atr)
    tp = curr_price + (rr_ratio * atr_mult * curr_atr) if signal == "BUY" else curr_price - (rr_ratio * atr_mult * curr_atr)

    risk_amt = acc_balance * (risk_pct / 100.0)
    risk_per_unit = abs(curr_price - sl) * contract_size
    lot_size = (risk_amt / risk_per_unit) if risk_per_unit != 0 else 0

    # --- UI Layout ---
    st.title(T["title"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(T["price"], f"${curr_price:,.2f}")
    c2.metric(T["rsi"], f"{curr_rsi:.2f}")
    c3.metric(T["atr"], f"{curr_atr:.2f}")
    c4.metric(T["dxy"], f"{curr_dxy:.2f}")
    c5.metric(T["corr"], f"{correlation:.2f}")

    c6, c7, c8 = st.columns([1.2, 1, 1])
    with c6:
        if signal == "BUY":
            st.markdown(f"<div class='signal-buy'><h3>{T['signal_buy']}</h3><p>Score: {score}</p></div>", unsafe_allow_html=True)
        elif signal == "SELL":
            st.markdown(f"<div class='signal-sell'><h3>{T['signal_sell']}</h3><p>Score: {score}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='signal-neutral'><h3>{T['signal_wait']}</h3><p>Score: {score}</p></div>", unsafe_allow_html=True)

    with c7:
        st.subheader(T["risk"])
        st.write(f"{T['risk_amount']}: ${risk_amt:.2f}")
        st.write(f"{T['lot_size']}: {lot_size:.3f}")

    with c8:
        st.subheader(T["targets"])
        st.write(f"{T['tp']}: {tp:,.2f}")
        st.write(f"{T['sl']}: {sl:,.2f}")
        st.write(T["rr_fmt"].format(rr=rr_ratio))

    # --- Chart ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, line=dict(color="orange"), name="EMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema200, line=dict(color="cyan"), name="EMA 200"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_hband(), line=dict(color="gray", width=1), name="BB High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_lband(), line=dict(color="gray", width=1), name="BB Low"), row=1, col=1)

    # TP/SL lines
    fig.add_hline(y=tp, line_dash="dash", line_color="green", row=1, col=1)
    fig.add_hline(y=sl, line_dash="dash", line_color="red", row=1, col=1)

    # Entry marker
    if signal in ["BUY", "SELL"]:
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[curr_price],
                mode="markers",
                marker=dict(size=12, color="lime" if signal == "BUY" else "red", symbol="triangle-up" if signal == "BUY" else "triangle-down"),
                name="Entry",
            ),
            row=1,
            col=1,
        )

    # Patterns
    hammers = df[df["Is_Hammer"]]
    if not hammers.empty:
        fig.add_trace(
            go.Scatter(x=hammers.index, y=hammers["Low"], mode="markers", marker=dict(symbol="triangle-up", size=8, color="yellow"), name="Hammer"),
            row=1,
            col=1,
        )

    dojis = df[df["Is_Doji"]]
    if not dojis.empty:
        fig.add_trace(
            go.Scatter(x=dojis.index, y=dojis["Close"], mode="markers", marker=dict(symbol="circle", size=6, color="white"), name="Doji"),
            row=1,
            col=1,
        )

    # RSI panel
    fig.add_trace(go.Scatter(x=df.index, y=rsi, line=dict(color="purple"), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(T["logic"]):
        for reason in reasons:
            st.write(f"- {reason}")
        st.write(f"- {T['higher_tf']}: {higher_tf} | {T['trend']}: {ht_trend}")

    st.caption(f"{T['last_update']}: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    if auto_refresh:
        try:
            # Streamlit built-in auto-refresh
            st_autorefresh = getattr(st, "autorefresh", None)
            if st_autorefresh is not None:
                st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")
            else:
                import time

                time.sleep(refresh_sec)
                st.rerun()
        except Exception:
            import time

            time.sleep(refresh_sec)
            st.rerun()

else:
    st.error(T["data_fail"])
