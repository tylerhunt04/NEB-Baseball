
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Stock Evaluator",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Evaluator")
st.markdown("Analyze stocks with real-time data, charts, and key metrics")

# Sidebar for stock input
st.sidebar.header("Stock Selection")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()

# Time range selection
time_range = st.sidebar.selectbox(
    "Select Time Range",
    ["1M", "3M", "6M", "1Y", "2Y", "5Y", "MAX"],
    index=3
)

# Map time range to period
period_map = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "MAX": "max"
}

# Fetch stock data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return stock, hist, info
    except Exception as e:
        return None, None, None

# Load data
with st.spinner(f"Loading data for {ticker_input}..."):
    stock, hist, info = get_stock_data(ticker_input, period_map[time_range])

if hist is not None and not hist.empty and info:
    # Company header
    company_name = info.get('longName', ticker_input)
    st.header(f"{company_name} ({ticker_input})")
    
    # Current price section
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
    previous_close = info.get('previousClose', 'N/A')
    
    if isinstance(current_price, (int, float)) and isinstance(previous_close, (int, float)):
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
        market_cap = info.get('marketCap', 'N/A')
        if isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                market_cap_display = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_display = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_display = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_display = f"${market_cap:,.0f}"
        else:
            market_cap_display = "N/A"
        st.metric("Market Cap", market_cap_display)
        
        pe_ratio = info.get('trailingPE', 'N/A')
        if isinstance(pe_ratio, (int, float)):
            st.metric("P/E Ratio", f"{pe_ratio:.2f}")
        else:
            st.metric("P/E Ratio", "N/A")
    
    with col2:
        volume = info.get('volume', 'N/A')
        if isinstance(volume, (int, float)):
            if volume >= 1e9:
                volume_display = f"{volume/1e9:.2f}B"
            elif volume >= 1e6:
                volume_display = f"{volume/1e6:.2f}M"
            else:
                volume_display = f"{volume:,.0f}"
        else:
            volume_display = "N/A"
        st.metric("Volume", volume_display)
        
        avg_volume = info.get('averageVolume', 'N/A')
        if isinstance(avg_volume, (int, float)):
            if avg_volume >= 1e9:
                avg_volume_display = f"{avg_volume/1e9:.2f}B"
            elif avg_volume >= 1e6:
                avg_volume_display = f"{avg_volume/1e6:.2f}M"
            else:
                avg_volume_display = f"{avg_volume:,.0f}"
        else:
            avg_volume_display = "N/A"
        st.metric("Avg Volume", avg_volume_display)
    
    with col3:
        high_52week = info.get('fiftyTwoWeekHigh', 'N/A')
        if isinstance(high_52week, (int, float)):
            st.metric("52 Week High", f"${high_52week:.2f}")
        else:
            st.metric("52 Week High", "N/A")
        
        low_52week = info.get('fiftyTwoWeekLow', 'N/A')
        if isinstance(low_52week, (int, float)):
            st.metric("52 Week Low", f"${low_52week:.2f}")
        else:
            st.metric("52 Week Low", "N/A")
    
    with col4:
        eps = info.get('trailingEps', 'N/A')
        if isinstance(eps, (int, float)):
            st.metric("EPS (TTM)", f"${eps:.2f}")
        else:
            st.metric("EPS (TTM)", "N/A")
        
        dividend_yield = info.get('dividendYield', 'N/A')
        if isinstance(dividend_yield, (int, float)):
            st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
        else:
            st.metric("Dividend Yield", "N/A")
    
    st.divider()
    
    # Calculate moving averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    
    # Price and Volume Chart
    st.subheader("📈 Price Chart & Volume")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Add price candlestick
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
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist['MA20'],
            name='MA20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist['MA50'],
            name='MA50',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['red' if row['Open'] > row['Close'] else 'green' for index, row in hist.iterrows()]
    fig.add_trace(
        go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Update layout
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
    
    # Company Information
    st.divider()
    st.subheader("🏢 Company Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Country:** {info.get('country', 'N/A')}")
    
    with col2:
        st.write(f"**Website:** {info.get('website', 'N/A')}")
        employees = info.get('fullTimeEmployees', 'N/A')
        if isinstance(employees, int):
            st.write(f"**Employees:** {employees:,}")
        else:
            st.write(f"**Employees:** N/A")
    
    # Business Summary
    if 'longBusinessSummary' in info:
        with st.expander("📋 Business Summary"):
            st.write(info['longBusinessSummary'])
    
    # Download data option
    st.divider()
    csv = hist.to_csv()
    st.download_button(
        label="📥 Download Historical Data (CSV)",
        data=csv,
        file_name=f"{ticker_input}_historical_data.csv",
        mime="text/csv"
    )

else:
    st.error(f"❌ Unable to fetch data for ticker '{ticker_input}'. Please check the ticker symbol and try again.")
    st.info("💡 Try popular tickers like: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA")

# Footer
st.divider()
st.caption("Data provided by Yahoo Finance. This is for educational purposes only and not financial advice.")
EOF
