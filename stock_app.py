import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Evaluator", page_icon="📈", layout="wide")

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
    "1M": "1mo", "3M": "3mo", "6M": "6mo",
    "1Y": "1y", "2Y": "2y", "5Y": "5y", "MAX": "max"
}

@st.cache_data(ttl=300)
def get_stock_data(ticker, period):
    """Fetch stock data with multiple fallback methods"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical data first (most reliable)
        hist = stock.history(period=period)
        if hist.empty:
            return None, None, None, None, "No historical data found. Ticker may be invalid."
        
        # Try to get info - handle cases where it's empty or has minimal data
        info = {}
        try:
            info = stock.info
            # Check if info has actual data (not just empty or minimal dict)
            if not info or len(info) < 5:
                st.warning(f"Limited company info available for {ticker}")
        except Exception as e:
            st.warning(f"Could not fetch detailed company info: {str(e)}")
        
        # Try to get fast_info as backup for price data
        fast_info = None
        try:
            fast_info = stock.fast_info
        except:
            pass
        
        return stock, hist, info, fast_info, None
        
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"

# Load data
with st.spinner(f"Loading data for {ticker_input}..."):
    stock, hist, info, fast_info, error = get_stock_data(ticker_input, period_map[time_range])

if error:
    st.error(f"❌ Unable to fetch data for ticker '{ticker_input}'")
    st.warning(f"**Details:** {error}")
    st.info("""
    **Try these solutions:**
    1. Verify the ticker symbol (e.g., AAPL for Apple)
    2. Make sure yfinance is up to date: `pip install --upgrade yfinance`
    3. Check your internet connection
    4. Try a different ticker: AAPL, MSFT, GOOGL, TSLA
    """)

elif hist is not None and not hist.empty:
    # Get company name
    company_name = info.get('longName') or info.get('shortName') or ticker_input
    st.header(f"{company_name} ({ticker_input})")
    
    # Get current price with multiple fallback methods
    current_price = None
    previous_close = None
    
    # Method 1: From fast_info (most reliable for price)
    if fast_info:
        try:
            current_price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
            previous_close = fast_info.get('previousClose')
        except:
            pass
    
    # Method 2: From info dict
    if not current_price:
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        previous_close = info.get('previousClose')
    
    # Method 3: From latest historical data
    if not current_price and not hist.empty:
        current_price = float(hist['Close'].iloc[-1])
        if len(hist) > 1:
            previous_close = float(hist['Close'].iloc[-2])
    
    # Display price metrics
    if current_price:
        if previous_close:
            price_change = current_price - previous_close
            price_change_pct = (price_change / previous_close) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
            col3.metric("Previous Close", f"${previous_close:.2f}")
        else:
            st.metric("Current Price", f"${current_price:.2f}")
    
    st.divider()
    
    # Key Statistics
    st.subheader("📊 Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    # Helper function to format large numbers
    def format_number(num):
        if not num or not isinstance(num, (int, float)):
            return "N/A"
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        else:
            return f"${num:,.0f}"
    
    with col1:
        # Market Cap
        market_cap = None
        if fast_info:
            try:
                market_cap = fast_info.get('marketCap')
            except:
                pass
        if not market_cap:
            market_cap = info.get('marketCap')
        st.metric("Market Cap", format_number(market_cap))
        
        # P/E Ratio
        pe = info.get('trailingPE') or info.get('forwardPE')
        st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
    
    with col2:
        # Volume - try multiple sources
        volume = None
        if fast_info:
            try:
                volume = fast_info.get('lastVolume')
            except:
                pass
        if not volume:
            volume = info.get('volume') or info.get('regularMarketVolume')
        if not volume and not hist.empty:
            volume = int(hist['Volume'].iloc[-1])
        
        vol_display = f"{volume/1e6:.2f}M" if volume and volume >= 1e6 else (f"{volume:,.0f}" if volume else "N/A")
        st.metric("Volume", vol_display)
        
        # Avg Volume
        avg_vol = info.get('averageVolume') or info.get('averageDailyVolume10Day')
        avg_display = f"{avg_vol/1e6:.2f}M" if avg_vol and avg_vol >= 1e6 else (f"{avg_vol:,.0f}" if avg_vol else "N/A")
        st.metric("Avg Volume", avg_display)
    
    with col3:
        # 52 Week Range
        high_52 = info.get('fiftyTwoWeekHigh')
        low_52 = info.get('fiftyTwoWeekLow')
        
        # Fallback: calculate from history if not in info
        if not high_52 and not hist.empty:
            high_52 = hist['High'].max()
        if not low_52 and not hist.empty:
            low_52 = hist['Low'].min()
        
        st.metric("52W High", f"${high_52:.2f}" if high_52 else "N/A")
        st.metric("52W Low", f"${low_52:.2f}" if low_52 else "N/A")
    
    with col4:
        # EPS
        eps = info.get('trailingEps')
        st.metric("EPS (TTM)", f"${eps:.2f}" if eps else "N/A")
        
        # Dividend Yield
        div_yield = info.get('dividendYield')
        st.metric("Div Yield", f"{div_yield*100:.2f}%" if div_yield else "N/A")
    
    st.divider()
    
    # Calculate moving averages
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    
    # Price Chart
    st.subheader("📈 Price Chart & Volume")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
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
    
    # Moving averages
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['MA20'],
                   name='MA20', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist['MA50'],
                   name='MA50', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['red' if row['Open'] > row['Close'] else 'green' 
              for _, row in hist.iterrows()]
    fig.add_trace(
        go.Bar(x=hist.index, y=hist['Volume'],
               name='Volume', marker_color=colors),
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
        website = info.get('website', 'N/A')
        st.write(f"**Website:** {website}")
        emp = info.get('fullTimeEmployees')
        st.write(f"**Employees:** {emp:,}" if emp else "**Employees:** N/A")
    
    # Business Summary
    summary = info.get('longBusinessSummary')
    if summary:
        with st.expander("📋 Business Summary"):
            st.write(summary)
    
    # Download option
    st.divider()
    csv = hist.to_csv()
    st.download_button(
        label="📥 Download Historical Data (CSV)",
        data=csv,
        file_name=f"{ticker_input}_data.csv",
        mime="text/csv"
    )

else:
    st.error(f"❌ Unable to fetch data")
    st.info("💡 Try: AAPL, MSFT, GOOGL, AMZN, TSLA")

st.divider()
st.caption("Data from Yahoo Finance via yfinance. For educational purposes only.")
