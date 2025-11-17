import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Stock Evaluator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Evaluator")
st.markdown("Analyze stocks with real-time data, charts, and key metrics")

# Sidebar
st.sidebar.header("Stock Selection")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()

time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["1M", "3M", "6M", "1Y", "2Y", "5Y", "MAX"],
    index=3
)

period_map = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "MAX": "max"
}

@st.cache_data(ttl=300)
def get_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            return None, None, None, "No historical data available"
        if not info or len(info) == 0:
            return None, None, None, "No stock information available"
            
        return stock, hist, info, None
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

with st.spinner(f"Loading data for {ticker_input}..."):
    stock, hist, info, error = get_stock_data(ticker_input, period_map[time_range])

if error:
    st.error(f"❌ Unable to fetch data for ticker '{ticker_input}'")
    st.warning(f"**Error details:** {error}")
    st.info("""
    **Possible solutions:**
    - Ensure yfinance is installed: `pip install yfinance`
    - Check your internet connection
    - Verify the ticker symbol is correct
    - Try popular tickers: AAPL, MSFT, GOOGL, AMZN, TSLA
    """)
    
elif hist is not None and not hist.empty and info:
    company_name = info.get('longName', ticker_input)
    st.header(f"{company_name} ({ticker_input})")
    
    # Current price
    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
    previous_close = info.get('previousClose')
    
    if not current_price and not hist.empty:
        current_price = float(hist['Close'].iloc[-1])
    
    if current_price and previous_close:
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
        with col3:
            st.metric("Previous Close", f"${previous_close:.2f}")
    
    st.divider()
    
    # Key Statistics
    st.subheader("📊 Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market_cap = info.get('marketCap')
        if market_cap:
            if market_cap >= 1e12:
                mc_display = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                mc_display = f"${market_cap/1e9:.2f}B"
            else:
                mc_display = f"${market_cap/1e6:.2f}M"
        else:
            mc_display = "N/A"
        st.metric("Market Cap", mc_display)
        
        pe = info.get('trailingPE')
        st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
    
    with col2:
        vol = info.get('volume')
        if vol:
            vol_display = f"{vol/1e6:.2f}M" if vol >= 1e6 else f"{vol:,.0f}"
        else:
            vol_display = "N/A"
        st.metric("Volume", vol_display)
        
        avg_vol = info.get('averageVolume')
        if avg_vol:
            avg_display = f"{avg_vol/1e6:.2f}M" if avg_vol >= 1e6 else f"{avg_vol:,.0f}"
        else:
            avg_display = "N/A"
        st.metric("Avg Volume", avg_display)
    
    with col3:
        high_52 = info.get('fiftyTwoWeekHigh')
        st.metric("52W High", f"${high_52:.2f}" if high_52 else "N/A")
        
        low_52 = info.get('fiftyTwoWeekLow')
        st.metric("52W Low", f"${low_52:.2f}" if low_52 else "N/A")
    
    with col4:
        eps = info.get('trailingEps')
        st.metric("EPS (TTM)", f"${eps:.2f}" if eps else "N/A")
        
        div_yield = info.get('dividendYield')
        st.metric("Div Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")
    
    st.divider()
    
    # Moving averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    
    # Charts
    st.subheader("📈 Price Chart & Volume")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist['MA20'],
            name='MA20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hist.index, y=hist['MA50'],
            name='MA50',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    colors = ['red' if row['Open'] > row['Close'] else 'green' 
              for _, row in hist.iterrows()]
    fig.add_trace(
        go.Bar(
            x=hist.index, y=hist['Volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{ticker_input} Stock Price and Volume",
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Company Info
    st.divider()
    st.subheader("🏢 Company Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Country:** {info.get('country', 'N/A')}")
    
    with col2:
        st.write(f"**Website:** {info.get('website', 'N/A')}")
        emp = info.get('fullTimeEmployees')
        st.write(f"**Employees:** {emp:,}" if emp else "**Employees:** N/A")
    
    if 'longBusinessSummary' in info:
        with st.expander("📋 Business Summary"):
            st.write(info['longBusinessSummary'])
    
    # Download
    st.divider()
    csv = hist.to_csv()
    st.download_button(
        label="📥 Download Historical Data (CSV)",
        data=csv,
        file_name=f"{ticker_input}_data.csv",
        mime="text/csv"
    )

else:
    st.error(f"❌ Unable to fetch data for ticker '{ticker_input}'")
    st.info("💡 Try: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA")

st.divider()
st.caption("Data from Yahoo Finance. For educational purposes only.")
