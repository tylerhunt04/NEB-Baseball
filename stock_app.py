import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for watchlist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Default watchlist

# Professional custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(to bottom, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Stock Info Card */
    .stock-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .stock-symbol {
        font-size: 3rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
        line-height: 1;
    }
    
    .stock-company {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    .stock-meta {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-value.positive {
        color: #48bb78;
    }
    
    .metric-value.negative {
        color: #f56565;
    }
    
    .metric-value.neutral {
        color: #cbd5e0;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
    }
    
    .metric-change.positive {
        background: rgba(72, 187, 120, 0.2);
        color: #48bb78;
    }
    
    .metric-change.negative {
        background: rgba(245, 101, 101, 0.2);
        color: #f56565;
    }
    
    /* Signal Badges */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .signal-strong-buy {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #68d391 0%, #48bb78 100%);
        color: white;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
        color: white;
    }
    
    .signal-strong-sell {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-card.success {
        border-left-color: #48bb78;
        background: rgba(72, 187, 120, 0.1);
    }
    
    .info-card.warning {
        border-left-color: #ed8936;
        background: rgba(237, 137, 54, 0.1);
    }
    
    .info-card.error {
        border-left-color: #f56565;
        background: rgba(245, 101, 101, 0.1);
    }
    
    .info-card.info {
        border-left-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(to bottom, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Section Headers */
    .section-header {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Table Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Watchlist Styling */
    .watchlist-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("### Analysis Configuration")
st.sidebar.markdown("<div style='height: 2px; background: linear-gradient(to right, transparent, #667eea, transparent); margin: 1rem 0;'></div>", unsafe_allow_html=True)

# Watchlist Section
st.sidebar.markdown("#### Watchlist")

# Select from watchlist
if st.session_state.watchlist:
    selected_from_watchlist = st.sidebar.selectbox(
        "Select from Watchlist",
        [""] + st.session_state.watchlist,
        key="watchlist_select"
    )
else:
    selected_from_watchlist = ""
    st.sidebar.info("Your watchlist is empty. Add stocks below.")

st.sidebar.markdown("<div style='height: 1px; background: rgba(255,255,255,0.1); margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

# Stock input with icon
st.sidebar.markdown("#### Stock Selection")
ticker_input = st.sidebar.text_input(
    "",
    value=selected_from_watchlist if selected_from_watchlist else "AAPL",
    placeholder="Enter ticker (e.g., AAPL, GOOGL, MSFT)",
    help="Enter a valid stock ticker symbol",
    key="ticker_input"
).upper()

# Use selected ticker from watchlist or manual input
ticker = selected_from_watchlist.upper() if selected_from_watchlist else ticker_input

# Watchlist management
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Add to Watchlist", use_container_width=True):
        if ticker and ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker)
            st.success(f"Added {ticker}")
            st.rerun()
        elif ticker in st.session_state.watchlist:
            st.warning("Already in watchlist")

with col2:
    if st.button("Remove", use_container_width=True):
        if ticker in st.session_state.watchlist:
            st.session_state.watchlist.remove(ticker)
            st.success(f"Removed {ticker}")
            st.rerun()
        else:
            st.warning("Not in watchlist")

# Display current watchlist
if st.session_state.watchlist:
    with st.sidebar.expander("View Watchlist", expanded=False):
        for stock in st.session_state.watchlist:
            st.markdown(f"• {stock}")

# Analyze button right after ticker input
analyze_button = st.sidebar.button("Analyze Stock", type="primary", use_container_width=True)

st.sidebar.markdown("<div style='height: 2px; background: linear-gradient(to right, transparent, #667eea, transparent); margin: 1rem 0;'></div>", unsafe_allow_html=True)

# Period selection with buttons
st.sidebar.markdown("#### Time Period")
period = st.sidebar.radio(
    "",
    ["1M", "3M", "6M", "1Y", "2Y", "5Y", "MAX"],
    index=3,  # Default to 1Y
    horizontal=False
)

# Map period to yfinance format
period_map = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "MAX": "max"
}

# Cache the download function
@st.cache_data(ttl=600)
def download_stock_data(ticker, period):
    """Download stock data with retry logic"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(retry_delay)
            
            df = yf.download(
                ticker, 
                period=period,
                progress=False,
                threads=False
            )
            
            if not df.empty:
                # Fix multi-level columns issue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                # Reset index and ensure Date column is properly named
                df = df.reset_index()
                
                # Make sure we have the right column names
                if 'Date' not in df.columns and 'Datetime' in df.columns:
                    df = df.rename(columns={'Datetime': 'Date'})
                elif 'Date' not in df.columns:
                    # First column is probably the date
                    df = df.rename(columns={df.columns[0]: 'Date'})
                
                return df, None
            else:
                return None, "No data found for this ticker"
                
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "Too Many Requests" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"Rate limited. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    return None, "Rate limited. Please wait a few minutes and try again."
            else:
                return None, f"Error: {error_msg}"
    
    return None, "Failed after multiple retries"

# Main content
if analyze_button or ticker:
    try:
        with st.spinner(f"Loading {period} of data for {ticker}..."):
            df, error = download_stock_data(ticker, period_map[period])
        
        if error:
            st.error(f"Error: {error}")
            
            if "Rate limited" in error:
                st.warning("""
                **You've been rate limited by Yahoo Finance.**
                
                **Solutions:**
                - Wait 2-3 minutes before trying again
                - Press 'C' to clear cache
                - Try a different ticker
                - Use a shorter time period
                """)
            else:
                st.info("Try: AAPL, MSFT, GOOGL, AMZN, TSLA")
            
        elif df is None or df.empty:
            st.error(f"No data found for '{ticker}'")
            st.info("Try: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA")
            
        else:
            st.success(f"Successfully loaded {len(df)} days of data for {ticker} ({period})")
            
            # Get company info
            company_name = ticker
            sector = "N/A"
            market_cap = None
            try:
                stock = yf.Ticker(ticker)
                info = stock.info if hasattr(stock, 'info') else {}
                company_name = info.get('longName', ticker)
                sector = info.get('sector', 'N/A')
                market_cap = info.get('marketCap', None)
            except:
                pass
            
            # Professional Stock Card
            latest_close = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else latest_close
            price_change = latest_close - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            # Determine signal based on price change
            if price_change_pct > 2:
                signal_class = "signal-strong-buy"
                signal_text = "STRONG BUY"
            elif price_change_pct > 0:
                signal_class = "signal-buy"
                signal_text = "BULLISH"
            elif price_change_pct > -2:
                signal_class = "signal-hold"
                signal_text = "HOLD"
            elif price_change_pct > -5:
                signal_class = "signal-sell"
                signal_text = "BEARISH"
            else:
                signal_class = "signal-strong-sell"
                signal_text = "STRONG SELL"
            
            # Format market cap
            mc_display = "N/A"
            if market_cap:
                if market_cap >= 1e12:
                    mc_display = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    mc_display = f"${market_cap/1e9:.2f}B"
                else:
                    mc_display = f"${market_cap/1e6:.2f}M"
            
            st.markdown(f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem;">
                    <div>
                        <h1 class="stock-symbol">{ticker}</h1>
                        <h2 class="stock-company">{company_name}</h2>
                        <p class="stock-meta">{sector} • Market Cap: {mc_display}</p>
                    </div>
                    <div class="signal-badge {signal_class}">
                        {signal_text}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics with professional cards
            col1, col2, col3, col4 = st.columns(4)
            
            change_class = "positive" if price_change >= 0 else "negative"
            arrow = "▲" if price_change >= 0 else "▼"
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value {change_class}">${latest_close:.2f}</div>
                    <div class="metric-change {change_class}">{arrow} {abs(price_change_pct):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Period High</div>
                    <div class="metric-value neutral">${float(df['High'].max()):.2f}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        {period} Range
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Period Low</div>
                    <div class="metric-value neutral">${float(df['Low'].min()):.2f}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        {period} Range
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_vol = float(df['Volume'].mean())
                if avg_vol >= 1e9:
                    vol_display = f"{avg_vol/1e9:.2f}B"
                elif avg_vol >= 1e6:
                    vol_display = f"{avg_vol/1e6:.2f}M"
                else:
                    vol_display = f"{avg_vol/1e3:.2f}K"
                    
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Avg Volume</div>
                    <div class="metric-value neutral">{vol_display}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        Daily Average
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            
            # Smart Correction Banner
            week_52_high = float(df['High'].max())
            drawdown_from_high = ((latest_close - week_52_high) / week_52_high) * 100
            
            if drawdown_from_high <= -10:
                # Get stock info for better messaging
                try:
                    stock_obj = yf.Ticker(ticker)
                    info_data = stock_obj.info if hasattr(stock_obj, 'info') else {}
                    market_cap = info_data.get('marketCap', 0)
                    sector = info_data.get('sector', 'Unknown')
                    
                    # Determine category
                    mega_cap_tech = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
                    defensive_sectors = ['Healthcare', 'Consumer Defensive', 'Utilities']
                    
                    if ticker.upper() in mega_cap_tech:
                        category_msg = "**Mega-Cap Tech** - Quality at a discount. Consider buying on 10-20% pullbacks."
                    elif sector in defensive_sectors:
                        category_msg = f"**Defensive {sector}** - Stable performer. Good time to accumulate."
                    else:
                        category_msg = "Check fundamentals before buying the dip."
                except:
                    category_msg = "Review the Market Insights tab for strategy guidance."
                
                if drawdown_from_high <= -20:
                    st.error(f"""
                    **DEEP CORRECTION ALERT** - {ticker} is down **{abs(drawdown_from_high):.1f}%** from 52-week high (${week_52_high:.2f})
                    
                    {category_msg}
                    
                    **Historical Context**: Deep corrections (20%+) often present the best long-term entry points for quality companies. 
                    See **Market Insights tab** for detailed analysis and opportunity score.
                    """)
                elif drawdown_from_high <= -15:
                    st.warning(f"""
                    **CORRECTION TERRITORY** - {ticker} is down **{abs(drawdown_from_high):.1f}%** from 52-week high (${week_52_high:.2f})
                    
                    {category_msg}
                    
                    Check the **Market Insights tab** for opportunity analysis and buying strategy.
                    """)
                else:
                    st.info(f"""
                    **PULLBACK DETECTED** - {ticker} is down **{abs(drawdown_from_high):.1f}%** from 52-week high (${week_52_high:.2f})
                    
                    {category_msg}
                    
                    Monitor for further weakness. See **Market Insights tab** for analysis.
                    """)
            
            # Create tabs - Fundamentals first, removed Raw Data
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Fundamentals",
                "Price Chart", 
                "Volume", 
                "Moving Averages",
                "Technical Indicators",
                "Analysis",
                "Market Insights"
            ])
            
            # Tab 1: Fundamentals (moved from tab 6)
            with tab1:
                st.subheader("Company Fundamentals")
                
                # Get company info
                try:
                    stock_obj = yf.Ticker(ticker)
                    info_data = stock_obj.info if hasattr(stock_obj, 'info') else {}
                    
                    # Company Overview
                    st.markdown("### Company Overview")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Company Name:** {info_data.get('longName', 'N/A')}")
                        st.markdown(f"**Sector:** {info_data.get('sector', 'N/A')}")
                        st.markdown(f"**Industry:** {info_data.get('industry', 'N/A')}")
                        st.markdown(f"**Country:** {info_data.get('country', 'N/A')}")
                        st.markdown(f"**Website:** {info_data.get('website', 'N/A')}")
                    
                    with col2:
                        employees = info_data.get('fullTimeEmployees', 'N/A')
                        if isinstance(employees, int):
                            st.markdown(f"**Employees:** {employees:,}")
                        else:
                            st.markdown(f"**Employees:** N/A")
                        
                        st.markdown(f"**Exchange:** {info_data.get('exchange', 'N/A')}")
                        st.markdown(f"**Currency:** {info_data.get('currency', 'N/A')}")
                        st.markdown(f"**Quote Type:** {info_data.get('quoteType', 'N/A')}")
                    
                    # Business Summary
                    summary = info_data.get('longBusinessSummary', '')
                    if summary:
                        with st.expander("Business Summary"):
                            st.write(summary)
                    
                    st.divider()
                    
                    # Key Financial Metrics
                    st.markdown("### Key Financial Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Market Cap
                        market_cap = info_data.get('marketCap')
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
                        
                        # Enterprise Value
                        ev = info_data.get('enterpriseValue')
                        if ev:
                            if ev >= 1e12:
                                ev_display = f"${ev/1e12:.2f}T"
                            elif ev >= 1e9:
                                ev_display = f"${ev/1e9:.2f}B"
                            else:
                                ev_display = f"${ev/1e6:.2f}M"
                        else:
                            ev_display = "N/A"
                        st.metric("Enterprise Value", ev_display)
                    
                    with col2:
                        # P/E Ratios
                        pe_trailing = info_data.get('trailingPE')
                        st.metric("P/E Ratio (TTM)", f"{pe_trailing:.2f}" if pe_trailing else "N/A")
                        
                        pe_forward = info_data.get('forwardPE')
                        st.metric("Forward P/E", f"{pe_forward:.2f}" if pe_forward else "N/A")
                    
                    with col3:
                        # Price to Book & Sales
                        pb = info_data.get('priceToBook')
                        st.metric("Price/Book", f"{pb:.2f}" if pb else "N/A")
                        
                        ps = info_data.get('priceToSalesTrailing12Months')
                        st.metric("Price/Sales", f"{ps:.2f}" if ps else "N/A")
                    
                    with col4:
                        # EPS & PEG
                        eps = info_data.get('trailingEps')
                        st.metric("EPS (TTM)", f"${eps:.2f}" if eps else "N/A")
                        
                        peg = info_data.get('pegRatio')
                        st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
                    
                    st.divider()
                    
                    # Profitability & Performance
                    st.markdown("### Profitability & Performance")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Margins
                        profit_margin = info_data.get('profitMargins')
                        if profit_margin:
                            st.metric("Profit Margin", f"{profit_margin*100:.2f}%")
                        else:
                            st.metric("Profit Margin", "N/A")
                        
                        operating_margin = info_data.get('operatingMargins')
                        if operating_margin:
                            st.metric("Operating Margin", f"{operating_margin*100:.2f}%")
                        else:
                            st.metric("Operating Margin", "N/A")
                    
                    with col2:
                        # Returns
                        roe = info_data.get('returnOnEquity')
                        if roe:
                            st.metric("Return on Equity", f"{roe*100:.2f}%")
                        else:
                            st.metric("Return on Equity", "N/A")
                        
                        roa = info_data.get('returnOnAssets')
                        if roa:
                            st.metric("Return on Assets", f"{roa*100:.2f}%")
                        else:
                            st.metric("Return on Assets", "N/A")
                    
                    with col3:
                        # Revenue & Earnings Growth
                        revenue_growth = info_data.get('revenueGrowth')
                        if revenue_growth:
                            st.metric("Revenue Growth", f"{revenue_growth*100:.2f}%")
                        else:
                            st.metric("Revenue Growth", "N/A")
                        
                        earnings_growth = info_data.get('earningsGrowth')
                        if earnings_growth:
                            st.metric("Earnings Growth", f"{earnings_growth*100:.2f}%")
                        else:
                            st.metric("Earnings Growth", "N/A")
                    
                    with col4:
                        # Beta & 52 Week Change
                        beta = info_data.get('beta')
                        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
                        
                        week52_change = info_data.get('52WeekChange')
                        if week52_change:
                            st.metric("52 Week Change", f"{week52_change*100:.2f}%")
                        else:
                            st.metric("52 Week Change", "N/A")
                    
                    st.divider()
                    
                    # Dividends & Financial Health
                    st.markdown("### Dividends & Financial Health")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Dividend Info
                        div_rate = info_data.get('dividendRate')
                        st.metric("Dividend Rate", f"${div_rate:.2f}" if div_rate else "N/A")
                        
                        div_yield = info_data.get('dividendYield')
                        if div_yield:
                            st.metric("Dividend Yield", f"{div_yield*100:.2f}%")
                        else:
                            st.metric("Dividend Yield", "N/A")
                    
                    with col2:
                        # Payout Ratio
                        payout = info_data.get('payoutRatio')
                        if payout:
                            st.metric("Payout Ratio", f"{payout*100:.2f}%")
                        else:
                            st.metric("Payout Ratio", "N/A")
                        
                        # Ex-Dividend Date
                        ex_div = info_data.get('exDividendDate')
                        if ex_div:
                            from datetime import datetime
                            ex_div_date = datetime.fromtimestamp(ex_div).strftime('%Y-%m-%d')
                            st.metric("Ex-Dividend Date", ex_div_date)
                        else:
                            st.metric("Ex-Dividend Date", "N/A")
                    
                    with col3:
                        # Debt & Cash
                        debt_to_equity = info_data.get('debtToEquity')
                        st.metric("Debt/Equity", f"{debt_to_equity:.2f}" if debt_to_equity else "N/A")
                        
                        current_ratio = info_data.get('currentRatio')
                        st.metric("Current Ratio", f"{current_ratio:.2f}" if current_ratio else "N/A")
                    
                    with col4:
                        # Cash & Book Value
                        total_cash = info_data.get('totalCash')
                        if total_cash:
                            if total_cash >= 1e9:
                                cash_display = f"${total_cash/1e9:.2f}B"
                            else:
                                cash_display = f"${total_cash/1e6:.2f}M"
                        else:
                            cash_display = "N/A"
                        st.metric("Total Cash", cash_display)
                        
                        book_value = info_data.get('bookValue')
                        st.metric("Book Value", f"${book_value:.2f}" if book_value else "N/A")
                    
                    st.divider()
                    
                    # Analyst Recommendations
                    st.markdown("### Analyst Recommendations")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        target_high = info_data.get('targetHighPrice')
                        st.metric("Target High", f"${target_high:.2f}" if target_high else "N/A")
                    
                    with col2:
                        target_mean = info_data.get('targetMeanPrice')
                        st.metric("Target Mean", f"${target_mean:.2f}" if target_mean else "N/A")
                    
                    with col3:
                        target_low = info_data.get('targetLowPrice')
                        st.metric("Target Low", f"${target_low:.2f}" if target_low else "N/A")
                    
                    recommendation = info_data.get('recommendationKey', 'N/A')
                    if recommendation != 'N/A':
                        if recommendation in ['buy', 'strong_buy']:
                            st.success(f"Analyst Recommendation: **{recommendation.upper().replace('_', ' ')}**")
                        elif recommendation == 'hold':
                            st.info(f"Analyst Recommendation: **{recommendation.upper()}**")
                        else:
                            st.warning(f"Analyst Recommendation: **{recommendation.upper().replace('_', ' ')}**")
                    
                except Exception as e:
                    st.error("Unable to load fundamental data. This may be due to API limitations.")
                    st.info(f"Error details: {str(e)}")
                
                # Educational section
                with st.expander("Understanding Fundamental Metrics"):
                    st.markdown("""
                    ### Key Fundamental Metrics Explained
                    
                    #### Valuation Metrics
                    
                    **P/E Ratio (Price-to-Earnings)**
                    - Price per share ÷ Earnings per share
                    - **Low P/E** (< 15) = Potentially undervalued or slow growth
                    - **High P/E** (> 25) = Potentially overvalued or high growth
                    - Compare to industry average for context
                    
                    **PEG Ratio**
                    - P/E Ratio ÷ Earnings Growth Rate
                    - **< 1.0** = Potentially undervalued
                    - **> 2.0** = Potentially overvalued
                    - Better than P/E for growth stocks
                    
                    **Price/Book (P/B)**
                    - Market value ÷ Book value
                    - **< 1.0** = Trading below book value (value opportunity)
                    - **> 3.0** = Premium valuation
                    
                    **Price/Sales (P/S)**
                    - Useful for unprofitable companies
                    - Compare within same industry
                    
                    ---
                    
                    #### Profitability Metrics
                    
                    **Profit Margin**
                    - Net income ÷ Revenue
                    - **> 10%** = Good profitability
                    - **> 20%** = Excellent profitability
                    
                    **ROE (Return on Equity)**
                    - Net income ÷ Shareholder equity
                    - **> 15%** = Good
                    - **> 20%** = Excellent
                    - Shows how efficiently company uses investments
                    
                    **ROA (Return on Assets)**
                    - Net income ÷ Total assets
                    - **> 5%** = Good
                    - **> 10%** = Excellent
                    
                    ---
                    
                    #### Growth Metrics
                    
                    **Revenue Growth**
                    - Year-over-year revenue increase
                    - **> 10%** = Strong growth
                    - Consistency matters more than one-time spikes
                    
                    **Earnings Growth**
                    - Year-over-year earnings increase
                    - **> 15%** = Strong growth
                    - Should match or exceed revenue growth
                    
                    ---
                    
                    #### Financial Health
                    
                    **Debt/Equity Ratio**
                    - Total debt ÷ Total equity
                    - **< 0.5** = Very safe
                    - **0.5-1.0** = Reasonable
                    - **> 2.0** = High debt, risky
                    
                    **Current Ratio**
                    - Current assets ÷ Current liabilities
                    - **> 1.5** = Healthy liquidity
                    - **< 1.0** = May struggle with short-term obligations
                    
                    **Beta**
                    - Measures volatility vs market
                    - **< 1.0** = Less volatile than market
                    - **= 1.0** = Same as market
                    - **> 1.0** = More volatile than market
                    
                    ---
                    
                    #### Dividend Metrics
                    
                    **Dividend Yield**
                    - Annual dividend ÷ Stock price
                    - **2-6%** = Typical for dividend stocks
                    - **> 8%** = Very high (check if sustainable)
                    
                    **Payout Ratio**
                    - Dividends ÷ Earnings
                    - **< 60%** = Sustainable
                    - **> 80%** = May be at risk if earnings drop
                    
                    ---
                    
                    #### How to Use This Data
                    
                    **Value Investing (Buy undervalued stocks)**
                    - Low P/E, PEG < 1.0, P/B < 1.5
                    - High ROE, good margins
                    - Low debt
                    
                    **Growth Investing (Buy high-growth stocks)**
                    - High revenue & earnings growth
                    - May have high P/E (paying for future growth)
                    - Strong margins improving over time
                    
                    **Income Investing (Buy dividend stocks)**
                    - High dividend yield (but sustainable)
                    - Low payout ratio (room to maintain/grow)
                    - Stable, profitable company
                    
                    **Quality Investing (Buy best companies)**
                    - High ROE & ROA
                    - Consistent growth
                    - Low debt
                    - Strong competitive position
                    """)
                    
                    st.info("""
                    **Pro Tip**: Don't look at metrics in isolation! A "high" P/E might be 
                    justified for a fast-growing company. Always compare to:
                    - Company's historical averages
                    - Industry peers
                    - Market average
                    - Growth rate (use PEG ratio)
                    """)
            
            # Tab 2: Price Chart
            with tab2:
                st.subheader(f"Closing Price Chart ({period})")
                
                # Chart type selector
                chart_type = st.radio("Chart Type:", ["Line Chart", "Candlestick Chart"], horizontal=True)
                
                # Ensure we have the right data
                chart_df = df.copy()
                
                fig = go.Figure()
                
                if chart_type == "Candlestick Chart":
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=chart_df['Date'],
                        open=chart_df['Open'],
                        high=chart_df['High'],
                        low=chart_df['Low'],
                        close=chart_df['Close'],
                        name='Price'
                    ))
                else:
                    # Line chart
                    fig.add_trace(go.Scatter(
                        x=chart_df['Date'],
                        y=chart_df['Close'],
                        mode='lines+markers',
                        name='Close Price',
                        line=dict(color='#00ff00', width=3),
                        marker=dict(size=4, color='#00ff00')
                    ))
                
                fig.update_layout(
                    title=f"{ticker} Stock Prices - {period}",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    height=500,
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white'),
                    xaxis=dict(
                        gridcolor='#444',
                        showgrid=True,
                        rangeslider=dict(visible=False)
                    ),
                    yaxis=dict(
                        gridcolor='#444',
                        showgrid=True
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Period High", f"${float(df['High'].max()):.2f}")
                col2.metric("Period Low", f"${float(df['Low'].min()):.2f}")
                
                # Educational section
                with st.expander("How to Read This Chart"):
                    st.markdown("""
                    ### Understanding Price Charts
                    
                    **What am I looking at?**
                    - The line/candles show the stock's price over time
                    - **Line Chart**: Simple closing prices connected
                    - **Candlestick Chart**: Shows Open, High, Low, Close for each day
                    
                    #### What to Look For:
                    
                    **Bullish Patterns (Good Signs)**
                    - **Uptrend**: Price steadily going up over time
                    - **Higher Highs**: Each peak is higher than the last
                    - **Higher Lows**: Each dip is higher than the last
                    - **Green Candles**: More green (up) days than red (down) days
                    
                    **Bearish Patterns (Warning Signs)**
                    - **Downtrend**: Price steadily going down
                    - **Lower Highs**: Each peak is lower than the last
                    - **Lower Lows**: Each dip is lower than the last
                    - **Red Candles**: More red (down) days than green (up) days
                    
                    **Sideways/Consolidation**
                    - Price moving horizontally
                    - No clear direction
                    - Often happens before a big move
                    
                    #### Candlestick Colors:
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("""
                        **Green Candle = Bullish**
                        - Close > Open
                        - Buyers won that day
                        - Price went up
                        """)
                    with col2:
                        st.error("""
                        **Red Candle = Bearish**
                        - Close < Open
                        - Sellers won that day
                        - Price went down
                        """)
                    
                    st.info("""
                    **Quick Tip**: Look at the overall pattern, not individual days. 
                    Is the general direction up, down, or sideways?
                    """)
            
            # Tab 3: Volume
            with tab3:
                st.subheader(f"Trading Volume ({period})")
                
                # Create volume chart with better colors - use vectorized operation
                vol_df = df.copy()
                
                # Check if we have the necessary columns
                if 'Close' in vol_df.columns and 'Open' in vol_df.columns:
                    vol_df['color'] = 'green'
                    vol_df.loc[vol_df['Close'] < vol_df['Open'], 'color'] = 'red'
                    colors = vol_df['color'].tolist()
                else:
                    # Default to blue if we don't have Open/Close
                    colors = 'lightblue'
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=vol_df['Date'],
                    y=vol_df['Volume'],
                    name='Volume',
                    marker_color=colors
                ))
                
                fig.update_layout(
                    title=f"{ticker} Trading Volume - {period}",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    hovermode='x unified',
                    height=500,
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444', showgrid=True),
                    yaxis=dict(gridcolor='#444', showgrid=True)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Educational section
                with st.expander("How to Read Volume"):
                    st.markdown("""
                    ### Understanding Trading Volume
                    
                    **What is Volume?**
                    - Number of shares traded each day
                    - Shows how much interest there is in the stock
                    - **Tall bars** = High volume (lots of activity)
                    - **Short bars** = Low volume (little activity)
                    
                    #### Volume Bar Colors:
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("""
                        **Green Bar**
                        - Price went UP that day
                        - Close > Open
                        - Buyers were stronger
                        """)
                    with col2:
                        st.error("""
                        **Red Bar**
                        - Price went DOWN that day
                        - Close < Open
                        - Sellers were stronger
                        """)
                    
                    st.markdown("""
                    #### What to Look For:
                    
                    **Best Signals (Strong Buy)**
                    - **Tall GREEN bars** when price is rising
                    - Means: Strong buying pressure, rally is real
                    - More greens than reds = Bullish sentiment
                    
                    **Warning Signals (Strong Sell)**
                    - **Tall RED bars** when price is falling
                    - Means: Strong selling pressure, decline is serious
                    - More reds than greens = Bearish sentiment
                    
                    **Caution Signals**
                    - **Short bars** on price increases = Weak rally, might reverse
                    - **Short bars** on price decreases = Minor dip, not concerning
                    
                    #### Volume Patterns:
                    
                    **Volume Spike**
                    - Sudden very tall bar (2-3x normal)
                    - Usually triggered by news or events
                    - Often signals trend change or acceleration
                    
                    **Increasing Volume**
                    - Bars getting taller over time
                    - Shows growing interest
                    - Confirms the current trend
                    
                    **Decreasing Volume**
                    - Bars getting shorter
                    - Shows declining interest
                    - May signal consolidation before next move
                    """)
                    
                    st.info("""
                    **Quick Tip**: Volume should **confirm** price movement. 
                    Price up + High volume = Good | Price up + Low volume = Suspicious
                    """)
            
            # Tab 4: Moving Averages
            with tab4:
                st.subheader("Moving Averages Analysis")
                
                ma_days = st.slider("Select Moving Average Period (days)", 
                                   min_value=5, max_value=200, value=20, step=5)
                
                # Calculate moving average
                df_copy = df.copy()
                df_copy['MA'] = df_copy['Close'].rolling(window=ma_days).mean()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_copy['Date'],
                    y=df_copy['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#00aaff', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_copy['Date'],
                    y=df_copy['MA'],
                    mode='lines',
                    name=f'{ma_days}-Day MA',
                    line=dict(color='#ff6600', width=3, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{ticker} Price with {ma_days}-Day Moving Average ({period})",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    height=500,
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444', showgrid=True),
                    yaxis=dict(gridcolor='#444', showgrid=True),
                    legend=dict(
                        bgcolor='#2e2e2e',
                        bordercolor='#555'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Educational section
                with st.expander("How to Read Moving Averages"):
                    st.markdown("""
                    ### Understanding Moving Averages (MA)
                    
                    **What is a Moving Average?**
                    - Average price over the last X days
                    - **Blue line** = Actual stock price
                    - **Orange dashed line** = Moving average
                    - Smooths out daily noise to show the trend
                    
                    #### What the Lines Mean:
                    
                    **20-Day MA** = Average of last 20 days (short-term trend)  
                    **50-Day MA** = Average of last 50 days (medium-term trend)  
                    **200-Day MA** = Average of last 200 days (long-term trend)
                    
                    #### What to Look For:
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("""
                        **Bullish Signals (Good)**
                        - Price ABOVE the MA line
                        - MA line sloping UPWARD
                        - Price bouncing off MA (support)
                        - Short MA crosses above long MA
                        """)
                    with col2:
                        st.error("""
                        **Bearish Signals (Bad)**
                        - Price BELOW the MA line
                        - MA line sloping DOWNWARD
                        - Price rejected by MA (resistance)
                        - Short MA crosses below long MA
                        """)
                    
                    st.markdown("""
                    #### Key Patterns:
                    
                    **Golden Cross (Very Bullish)**
                    - 20-day MA crosses ABOVE 50-day MA
                    - Strong buy signal
                    - Often starts a major uptrend
                    
                    **Death Cross (Very Bearish)**
                    - 20-day MA crosses BELOW 50-day MA
                    - Strong sell signal
                    - Often starts a major downtrend
                    
                    **Support**
                    - Price drops to MA and bounces back up
                    - MA acts like a floor
                    - Shows buyers defend that level
                    
                    **Resistance**
                    - Price rises to MA and gets pushed down
                    - MA acts like a ceiling
                    - Shows sellers defend that level
                    
                    #### How to Use the Slider:
                    
                    **5-20 days**: Very short-term, responds quickly to changes  
                    **20-50 days**: Good for swing trading (weeks to months)  
                    **50-200 days**: Long-term investing (months to years)
                    
                    Try different periods and see how the MA changes!
                    """)
                    
                    st.info("""
                    **Quick Analysis**: 
                    - Price **above** MA + MA going **up** = Buy signal
                    - Price **below** MA + MA going **down** = Sell signal
                    - Price **at** MA + MA **flat** = Wait for direction
                    """)
                    
                    st.warning("""
                    **Important**: Moving averages are **lagging indicators** - they show what 
                    already happened, not what will happen. Use them to confirm trends, not predict them.
                    """)
            
            # Tab 5: Technical Indicators
            with tab5:
                st.subheader("Technical Indicators")
                
                # Calculate technical indicators
                
                # RSI (Relative Strength Index)
                def calculate_rsi(data, period=14):
                    delta = data.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi
                
                df_tech = df.copy()
                df_tech['RSI'] = calculate_rsi(df_tech['Close'])
                
                # MACD
                exp1 = df_tech['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df_tech['Close'].ewm(span=26, adjust=False).mean()
                df_tech['MACD'] = exp1 - exp2
                df_tech['Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
                df_tech['MACD_Histogram'] = df_tech['MACD'] - df_tech['Signal']
                
                # Bollinger Bands
                df_tech['BB_Middle'] = df_tech['Close'].rolling(window=20).mean()
                df_tech['BB_Std'] = df_tech['Close'].rolling(window=20).std()
                df_tech['BB_Upper'] = df_tech['BB_Middle'] + (df_tech['BB_Std'] * 2)
                df_tech['BB_Lower'] = df_tech['BB_Middle'] - (df_tech['BB_Std'] * 2)
                
                # RSI Chart
                st.markdown("### RSI (Relative Strength Index)")
                fig_rsi = go.Figure()
                
                fig_rsi.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                
                # Add overbought/oversold lines
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                  annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                  annotation_text="Oversold (30)")
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", 
                                  annotation_text="Neutral (50)")
                
                # Add background colors
                fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
                fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
                
                fig_rsi.update_layout(
                    title="RSI Indicator",
                    yaxis_title="RSI Value",
                    xaxis_title="Date",
                    height=300,
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444', range=[0, 100])
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # Current RSI value
                current_rsi = df_tech['RSI'].iloc[-1]
                if current_rsi > 70:
                    st.error(f"Current RSI: {current_rsi:.2f} - **OVERBOUGHT** (Potential sell signal)")
                elif current_rsi < 30:
                    st.success(f"Current RSI: {current_rsi:.2f} - **OVERSOLD** (Potential buy signal)")
                else:
                    st.info(f"Current RSI: {current_rsi:.2f} - **NEUTRAL** (No strong signal)")
                
                st.divider()
                
                # MACD Chart
                st.markdown("### MACD (Moving Average Convergence Divergence)")
                fig_macd = go.Figure()
                
                # MACD line
                fig_macd.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ))
                
                # Signal line
                fig_macd.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='orange', width=2)
                ))
                
                # Histogram
                colors = ['green' if val >= 0 else 'red' for val in df_tech['MACD_Histogram']]
                fig_macd.add_trace(go.Bar(
                    x=df_tech['Date'],
                    y=df_tech['MACD_Histogram'],
                    name='Histogram',
                    marker_color=colors,
                    opacity=0.3
                ))
                
                fig_macd.update_layout(
                    title="MACD Indicator",
                    yaxis_title="MACD Value",
                    xaxis_title="Date",
                    height=300,
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444')
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
                
                # MACD Signal
                current_macd = df_tech['MACD'].iloc[-1]
                current_signal = df_tech['Signal'].iloc[-1]
                if current_macd > current_signal:
                    st.success(f"MACD above Signal - **BULLISH** (Upward momentum)")
                else:
                    st.error(f"MACD below Signal - **BEARISH** (Downward momentum)")
                
                st.divider()
                
                # Bollinger Bands Chart
                st.markdown("### Bollinger Bands")
                fig_bb = go.Figure()
                
                # Price
                fig_bb.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='white', width=2)
                ))
                
                # Upper band
                fig_bb.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['BB_Upper'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='red', width=1, dash='dash')
                ))
                
                # Middle band
                fig_bb.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['BB_Middle'],
                    mode='lines',
                    name='Middle Band (20 MA)',
                    line=dict(color='gray', width=1)
                ))
                
                # Lower band
                fig_bb.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['BB_Lower'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='green', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(100, 100, 100, 0.2)'
                ))
                
                fig_bb.update_layout(
                    title="Bollinger Bands",
                    yaxis_title="Price (USD)",
                    xaxis_title="Date",
                    height=400,
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444')
                )
                
                st.plotly_chart(fig_bb, use_container_width=True)
                
                # Bollinger Bands Signal
                latest_close = float(df_tech['Close'].iloc[-1])
                latest_upper = float(df_tech['BB_Upper'].iloc[-1])
                latest_lower = float(df_tech['BB_Lower'].iloc[-1])
                
                if latest_close > latest_upper:
                    st.error(f"Price above upper band (${latest_upper:.2f}) - **OVERBOUGHT**")
                elif latest_close < latest_lower:
                    st.success(f"Price below lower band (${latest_lower:.2f}) - **OVERSOLD**")
                else:
                    st.info(f"Price within bands - **NORMAL RANGE**")
                
                # Educational section
                with st.expander("How to Read Technical Indicators"):
                    st.markdown("""
                    ### Understanding Technical Indicators
                    
                    #### RSI (Relative Strength Index)
                    
                    **What is RSI?**
                    - Measures momentum on a scale of 0-100
                    - Shows if stock is overbought or oversold
                    - Based on recent price changes
                    
                    **How to Read RSI:**
                    - **Above 70** = **OVERBOUGHT** (might drop soon, consider selling)
                    - **Below 30** = **OVERSOLD** (might rise soon, consider buying)
                    - **Around 50** = **NEUTRAL** (no strong signal)
                    
                    **RSI Divergence:**
                    - Price makes new high, but RSI doesn't = Bearish signal
                    - Price makes new low, but RSI doesn't = Bullish signal
                    
                    ---
                    
                    #### MACD (Moving Average Convergence Divergence)
                    
                    **What is MACD?**
                    - Shows relationship between two moving averages
                    - Three components: MACD line, Signal line, Histogram
                    - Identifies momentum and trend changes
                    
                    **How to Read MACD:**
                    - **MACD crosses above Signal** = **BUY SIGNAL** (bullish)
                    - **MACD crosses below Signal** = **SELL SIGNAL** (bearish)
                    - **Histogram growing** = Momentum increasing
                    - **Histogram shrinking** = Momentum decreasing
                    
                    **MACD Colors:**
                    - Green histogram = Positive momentum
                    - Red histogram = Negative momentum
                    
                    ---
                    
                    #### Bollinger Bands
                    
                    **What are Bollinger Bands?**
                    - Three lines: Upper, Middle (20-day MA), Lower
                    - Bands widen when volatility increases
                    - Bands narrow when volatility decreases
                    
                    **How to Read Bollinger Bands:**
                    - **Price touches upper band** = Overbought (might reverse down)
                    - **Price touches lower band** = Oversold (might reverse up)
                    - **Price at middle band** = Fair value
                    - **Bands squeezing** = Big move coming soon
                    - **Bands widening** = High volatility, big moves happening
                    
                    **The "Squeeze":**
                    - When bands get very narrow
                    - Stock is consolidating
                    - Often followed by big breakout (up or down)
                    
                    ---
                    
                    #### Combining Indicators
                    
                    **Strong Buy Signal:**
                    - RSI < 30 (oversold)
                    - MACD crosses above Signal
                    - Price near lower Bollinger Band
                    
                    **Strong Sell Signal:**
                    - RSI > 70 (overbought)
                    - MACD crosses below Signal
                    - Price near upper Bollinger Band
                    
                    **Wait Signal:**
                    - Mixed signals from different indicators
                    - Better to wait for confirmation
                    """)
                    
                    st.info("""
                    **Pro Tip**: Don't rely on just one indicator! Use multiple indicators 
                    together for confirmation. When 2-3 indicators agree, the signal is stronger!
                    """)
            
            # Tab 6: Analysis
            with tab6:
                st.subheader(f"Stock Analysis Summary ({period})")
                
                # Calculate statistics
                first_close = float(df['Close'].iloc[0])
                last_close = float(df['Close'].iloc[-1])
                total_return = ((last_close - first_close) / first_close) * 100
                
                daily_returns = df['Close'].pct_change()
                volatility = float(daily_returns.std()) * 100
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Total Return", f"{total_return:.2f}%")
                col2.metric("Daily Volatility", f"{volatility:.2f}%")
                col3.metric("Data Points", len(df))
                
                st.divider()
                
                # Daily returns histogram
                st.subheader("Daily Returns Distribution")
                daily_returns_pct = daily_returns.dropna() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=daily_returns_pct,
                    nbinsx=50,
                    name='Daily Returns',
                    marker_color='#00ff88',
                    opacity=0.8
                ))
                
                fig.update_layout(
                    title=f"{ticker} Daily Returns Distribution",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=400,
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444', showgrid=True),
                    yaxis=dict(gridcolor='#444', showgrid=True)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent performance
                st.subheader("Recent Performance (Last 10 Days)")
                recent_data = df.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                st.dataframe(recent_data, use_container_width=True, hide_index=True)
                
                # Educational section
                with st.expander("Understanding the Analysis Metrics"):
                    st.markdown("""
                    ### What Do These Numbers Mean?
                    
                    **Total Return**
                    - Percentage change from start to end of period
                    - **Positive %** = Stock went up
                    - **Negative %** = Stock went down
                    - Example: +25% means $100 became $125
                    
                    **Daily Volatility**
                    - How much the stock bounces around each day
                    - **Low volatility** (0-2%) = Stable, steady stock
                    - **Medium volatility** (2-5%) = Normal stock
                    - **High volatility** (5%+) = Risky, jumps a lot
                    
                    **Daily Returns Distribution**
                    - Histogram shows how often stock goes up/down
                    - **Right side** (positive) = Up days
                    - **Left side** (negative) = Down days
                    - **Peak in middle** = Most days are small moves
                    - **Spread out** = Lots of big swings (volatile)
                    
                    #### Good vs Bad:
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("""
                        **Positive Signs**
                        - Total Return > 10%
                        - More green on histogram (right side)
                        - Steady climb in recent performance
                        - Reasonable volatility for your risk level
                        """)
                    with col2:
                        st.error("""
                        **Warning Signs**
                        - Total Return negative
                        - More red on histogram (left side)
                        - Declining recent performance
                        - Extreme volatility (unless you like risk)
                        """)
                    
                    st.info("""
                    **Investment Tip**: 
                    - **Conservative**: Look for positive returns + low volatility
                    - **Moderate**: Accept medium volatility for better returns
                    - **Aggressive**: High volatility = high risk + high reward potential
                    """)
            
            # Tab 7: Market Insights
            with tab7:
                st.markdown("### Market Insights & Investment Strategy")
                
                # Get additional stock info for classification
                try:
                    stock_obj = yf.Ticker(ticker)
                    info_data = stock_obj.info if hasattr(stock_obj, 'info') else {}
                    market_cap = info_data.get('marketCap', 0)
                    sector = info_data.get('sector', 'Unknown')
                    pe_ratio = info_data.get('trailingPE', None)
                    industry = info_data.get('industry', 'Unknown')
                except:
                    market_cap = 0
                    sector = 'Unknown'
                    pe_ratio = None
                    industry = 'Unknown'
                
                # Calculate correction metrics
                latest_price = float(df['Close'].iloc[-1])
                week_52_high = float(df['High'].max())
                week_52_low = float(df['Low'].min())
                drawdown_from_high = ((latest_price - week_52_high) / week_52_high) * 100
                distance_from_low = ((latest_price - week_52_low) / week_52_low) * 100
                
                # Classify stock category
                def classify_stock(market_cap, sector, ticker):
                    """Classify stock into investment categories"""
                    # Mega-cap tech stocks
                    mega_cap_tech = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
                    
                    # Defensive sectors
                    defensive_sectors = ['Healthcare', 'Consumer Defensive', 'Utilities']
                    
                    # Value sectors
                    value_sectors = ['Energy', 'Financial Services', 'Industrials', 'Basic Materials']
                    
                    if ticker.upper() in mega_cap_tech or (market_cap >= 500e9 and sector == 'Technology'):
                        return "Mega-Cap Technology", "mega_cap"
                    elif sector in defensive_sectors:
                        return f"Defensive ({sector})", "defensive"
                    elif sector in value_sectors:
                        return f"Value Play ({sector})", "value"
                    elif market_cap >= 200e9:
                        return "Large-Cap", "large_cap"
                    elif market_cap >= 10e9:
                        return "Mid-Cap", "mid_cap"
                    else:
                        return "Small-Cap / Growth", "small_cap"
                
                category_name, category_type = classify_stock(market_cap, sector, ticker)
                
                # Correction status
                if drawdown_from_high <= -20:
                    correction_status = "DEEP CORRECTION"
                    correction_color = "error"
                    correction_desc = "Down 20%+ from highs - Strong buy signal for quality stocks"
                elif drawdown_from_high <= -10:
                    correction_status = "CORRECTION TERRITORY"
                    correction_color = "warning"
                    correction_desc = "Down 10-20% from highs - Buy zone for long-term investors"
                elif drawdown_from_high <= -5:
                    correction_status = "MILD PULLBACK"
                    correction_color = "info"
                    correction_desc = "Down 5-10% from highs - Watch for entry opportunity"
                else:
                    correction_status = "NEAR HIGHS"
                    correction_color = "success"
                    correction_desc = "Within 5% of 52-week high - Exercise caution"
                
                # Opportunity Score Calculation
                score = 0
                score_factors = []
                
                # Factor 1: Correction depth (0-3 points)
                if drawdown_from_high <= -20:
                    score += 3
                    score_factors.append("Deep correction (3 pts)")
                elif drawdown_from_high <= -10:
                    score += 2
                    score_factors.append("In correction (2 pts)")
                elif drawdown_from_high <= -5:
                    score += 1
                    score_factors.append("Mild pullback (1 pt)")
                else:
                    score_factors.append("Near highs (0 pts)")
                
                # Factor 2: Fundamentals (0-3 points)
                if pe_ratio and pe_ratio < 25:
                    score += 2
                    score_factors.append("Reasonable P/E (2 pts)")
                elif pe_ratio and pe_ratio < 35:
                    score += 1
                    score_factors.append("Moderate P/E (1 pt)")
                else:
                    score_factors.append("High/No P/E (0 pts)")
                
                # Factor 3: Category strength (0-2 points)
                if category_type in ['mega_cap', 'defensive']:
                    score += 2
                    score_factors.append("Strong category (2 pts)")
                elif category_type in ['large_cap', 'value']:
                    score += 1
                    score_factors.append("Good category (1 pt)")
                else:
                    score_factors.append("Higher risk (0 pts)")
                
                # Factor 4: Distance from low (0-2 points)
                if distance_from_low > 50:
                    score += 1
                    score_factors.append("Well above lows (1 pt)")
                elif distance_from_low > 100:
                    score += 2
                    score_factors.append("Far from lows (2 pts)")
                else:
                    score_factors.append("Near lows (0 pts)")
                
                max_score = 10
                opportunity_score = (score / max_score) * 10
                
                # Strategy recommendations by category
                strategy_guide = {
                    "mega_cap": {
                        "strategy": "Buy on 10-20% pullbacks",
                        "reasoning": "These titans don't disappear in downturns; they just go on sale. A 10-20% drop is a long-term gift.",
                        "hold_period": "Long-term (3-5+ years)",
                        "risk_level": "Low to Moderate"
                    },
                    "defensive": {
                        "strategy": "Accumulate during volatility",
                        "reasoning": "When economic uncertainty rises, consistency becomes king. Essential services with steady cash flow.",
                        "hold_period": "Long-term (5+ years)",
                        "risk_level": "Low"
                    },
                    "value": {
                        "strategy": "Buy when P/E is below historical average",
                        "reasoning": "Quality value stocks overlooked by growth obsession. Trading below historical averages presents opportunity.",
                        "hold_period": "Medium to Long-term (2-5 years)",
                        "risk_level": "Moderate"
                    },
                    "large_cap": {
                        "strategy": "Buy on weakness, verify fundamentals",
                        "reasoning": "Established companies with proven business models. Verify financial health before buying dips.",
                        "hold_period": "Long-term (3-5 years)",
                        "risk_level": "Moderate"
                    },
                    "mid_cap": {
                        "strategy": "Selective buying on deep corrections",
                        "reasoning": "More volatile but growth potential. Buy only on significant pullbacks with strong fundamentals.",
                        "hold_period": "Medium-term (2-3 years)",
                        "risk_level": "Moderate to High"
                    },
                    "small_cap": {
                        "strategy": "High risk - Buy only with strong conviction",
                        "reasoning": "Highest volatility. Requires deep research and strong conviction. Small position sizes recommended.",
                        "hold_period": "Variable (1-3 years)",
                        "risk_level": "High"
                    }
                }
                
                strategy = strategy_guide.get(category_type, strategy_guide["large_cap"])
                
                # Display Correction Analysis
                st.markdown("### Correction Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="info-card {correction_color}">
                        <h4>{correction_status}</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                            {drawdown_from_high:.1f}%
                        </p>
                        <p style="font-size: 0.85rem; opacity: 0.8;">from 52-week high</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="info-card info">
                        <h4>52-Week Range</h4>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0;">
                            ${week_52_low:.2f} - ${week_52_high:.2f}
                        </p>
                        <p style="font-size: 0.85rem; opacity: 0.8;">Current: ${latest_price:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    score_color = "success" if opportunity_score >= 7 else ("warning" if opportunity_score >= 5 else "error")
                    st.markdown(f"""
                    <div class="info-card {score_color}">
                        <h4>Opportunity Score</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                            {opportunity_score:.1f}/10
                        </p>
                        <p style="font-size: 0.85rem; opacity: 0.8;">
                            {"Strong Buy" if opportunity_score >= 7 else ("Moderate" if opportunity_score >= 5 else "Weak")}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-card info" style="margin-top: 1rem;">
                    <strong>Assessment:</strong> {correction_desc}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                
                # Stock Classification & Strategy
                st.markdown("### Investment Category & Strategy")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Stock Category</div>
                        <div class="metric-value neutral" style="font-size: 1.5rem;">
                            {category_name}
                        </div>
                        <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 0.5rem;">
                            Sector: {sector}<br>
                            Industry: {industry}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Recommended Strategy</div>
                        <div style="font-size: 1.2rem; margin: 0.5rem 0; color: #667eea;">
                            {strategy['strategy']}
                        </div>
                        <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                            Risk Level: {strategy['risk_level']}<br>
                            Hold Period: {strategy['hold_period']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-card info" style="margin-top: 1rem;">
                    <strong>Strategy Reasoning:</strong><br>
                    {strategy['reasoning']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                
                # Opportunity Score Breakdown
                st.markdown("### Opportunity Score Breakdown")
                
                st.markdown("""
                <div class="metric-container" style="margin-bottom: 1rem;">
                """, unsafe_allow_html=True)
                
                for factor in score_factors:
                    st.markdown(f"- {factor}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Buying conditions
                if opportunity_score >= 7:
                    st.success("""
                    **STRONG BUYING OPPORTUNITY**
                    
                    Multiple factors align for a potential entry:
                    - Stock is significantly below recent highs
                    - Fundamentals appear reasonable
                    - Category fits long-term strategy
                    
                    Consider starting or adding to position with proper position sizing.
                    """)
                elif opportunity_score >= 5:
                    st.warning("""
                    **MODERATE OPPORTUNITY**
                    
                    Some positive factors present:
                    - May be worth watching for further weakness
                    - Verify fundamentals before entry
                    - Consider waiting for better entry point
                    
                    Not a clear buy signal yet - patience recommended.
                    """)
                else:
                    st.error("""
                    **WEAK OPPORTUNITY**
                    
                    Limited positive factors:
                    - Stock may be overvalued or near highs
                    - Fundamentals may not support current price
                    - Higher risk category or weak technicals
                    
                    Better opportunities likely available elsewhere.
                    """)
                
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                
                # Market Context & Philosophy
                st.markdown("### Understanding Market Corrections")
                
                with st.expander("Market Correction Philosophy"):
                    st.markdown("""
                    ### The Nature of Market Corrections
                    
                    A deep red flush across market tickers, rising panic in financial headlines, palpable anxiety—these are the unmistakable signs of a market correction. In these moments, the impulse to react emotionally is strong. Yet, what if we chose to see this not as a crisis, but as a **predictable cycle**?
                    
                    **The market, after a period of exuberant growth, is simply taking a breath.**
                    
                    ---
                    
                    ### Why Corrections Happen
                    
                    This pullback is not the financial apocalypse. It is the **cost of doing business** for anyone committed to long-term wealth creation. Current factors include:
                    
                    **1. Persistent Inflation & Rate Cut Delays**
                    - Market priced in swift interest rate cuts
                    - Inflation proving stickier than anticipated
                    - Central banks signaling patience
                    - Era of cheap money isn't returning quickly
                    
                    **2. Tech & AI Valuation Reset**
                    - Meteoric rise stretched valuations
                    - Even revolutionary companies answer to fundamentals
                    - Healthy correction bringing prices to reality
                    
                    **3. "Risk-Off" Sentiment Shift**
                    - After high-risk speculation period
                    - Capital rotating toward safety
                    - Traders unwinding leveraged bets
                    - Moving to less volatile sectors
                    
                    ---
                    
                    ### The Opportunity in Volatility
                    
                    **For the strategic investor, volatility is not a threat; it is a signal of opportunity.**
                    
                    This is the window where:
                    - Assets return to rational prices
                    - Careful planning pays dividends
                    - Quality companies go "on sale"
                    - Long-term wealth is built
                    
                    Think of corrections not as a collapse, but as a **cleansing**. They purge the system of:
                    - Excess speculation
                    - Fragile business models
                    - Undisciplined leverage
                    
                    It's during these resets that the foundation for the next wave of growth is quietly laid.
                    
                    ---
                    
                    ### Investment Categories During Corrections
                    
                    #### Mega-Cap Technology
                    **Examples:** Microsoft, Alphabet, Amazon, Apple, Nvidia
                    
                    **Strategy:** Buy on 10-20% pullbacks
                    - Don't disappear in downturns
                    - Global reach and fortified balance sheets
                    - Undeniable market power
                    - Fundamental building blocks
                    
                    #### Defensive Stalwarts
                    **Sectors:** Healthcare, Consumer Staples, Utilities, Insurance
                    **Examples:** UnitedHealth, Johnson & Johnson, PepsiCo, Procter & Gamble
                    
                    **Strategy:** Accumulate during volatility
                    - Essential services
                    - Steady cash flow
                    - Reliable dividends
                    - Defensive buffer against fear
                    
                    #### Undervalued Value Plays
                    **Sectors:** Energy, Financials, Industrials
                    
                    **Strategy:** Buy when P/E below historical average
                    - Overlooked by growth obsession
                    - Capital now rotating here
                    - Trading below historical P/E ratios
                    - Compelling for value investors
                    
                    ---
                    
                    ### The Strategic Mindset
                    
                    **What separates successful investors from emotional traders:**
                    
                    - See corrections as opportunities, not crises
                    - Focus on fundamentals, not fear
                    - Buy quality assets at discount prices
                    - Think in years, not days
                    - Position sizing protects capital
                    - Patience beats panic
                    
                    ---
                    
                    ### Key Principles
                    
                    1. **Volatility Creates Discounts on Quality**
                       - Best companies become affordable
                       - 10-20% drops are gifts for patient investors
                    
                    2. **Consistency Beats Speculation**
                       - Steady accumulation during fear
                       - Dollar-cost averaging into weakness
                    
                    3. **Fundamentals Always Matter**
                       - Strong balance sheets survive downturns
                       - Quality businesses recover faster
                    
                    4. **Time in Market > Timing Market**
                       - Long-term holdings outperform trading
                       - Corrections are temporary, growth is permanent
                    
                    ---
                    
                    ### Final Thought
                    
                    > "Be fearful when others are greedy, and greedy when others are fearful." 
                    > — Warren Buffett
                    
                    This is not about reckless buying or catching falling knives. It's about **strategic accumulation of quality assets when they return to rational prices**.
                    
                    The best investors are made during corrections, not bull markets.
                    """)
                
                with st.expander("How to Use This Analysis"):
                    st.markdown("""
                    ### Using the Opportunity Score
                    
                    **Score 8-10: Strong Buy Signal**
                    - Multiple positive factors aligned
                    - Consider starting or adding to position
                    - Use proper position sizing (never go all-in)
                    - Set a long-term horizon
                    
                    **Score 5-7: Moderate Opportunity**
                    - Some positive factors present
                    - Add to watchlist
                    - Wait for more confirmation
                    - May want to see further weakness
                    
                    **Score 0-4: Weak Signal**
                    - Limited positive factors
                    - Better opportunities exist elsewhere
                    - May be overvalued or risky
                    - Pass for now
                    
                    ---
                    
                    ### Position Sizing Guidelines
                    
                    **Never invest more than you can afford to lose:**
                    
                    - **Conservative:** 2-3% per position
                    - **Moderate:** 3-5% per position
                    - **Aggressive:** 5-10% per position (max)
                    
                    **Diversification is key:**
                    - 10-20 positions minimum
                    - Multiple sectors
                    - Mix of categories (Mega-cap, Defensive, Value)
                    
                    ---
                    
                    ### When to Buy
                    
                    **Do Buy When:**
                    - Opportunity score ≥ 7
                    - Stock down 10-20%+ from highs
                    - Fundamentals remain strong
                    - Category fits your strategy
                    - You have long-term horizon (3+ years)
                    
                    **Don't Buy When:**
                    - Opportunity score < 5
                    - Stock near all-time highs
                    - Fundamentals deteriorating
                    - Just because price is falling
                    - You need the money short-term
                    
                    ---
                    
                    ### Risk Management
                    
                    **Always:**
                    - Have an emergency fund (6-12 months expenses)
                    - Never use leverage or margin
                    - Don't invest money you need soon
                    - Diversify across categories
                    - Review positions regularly
                    
                    **Remember:**
                    - This tool provides analysis, not financial advice
                    - Do your own research
                    - Consider your risk tolerance
                    - Consult a financial advisor if needed
                    """)
                
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                
                # Watchlist by Category
                st.markdown("### Example Watchlist by Category")
                
                watchlist_data = {
                    "Mega-Cap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
                    "Defensive Healthcare": ["UNH", "JNJ", "PFE", "ABBV"],
                    "Consumer Staples": ["PEP", "PG", "KO", "WMT", "COST"],
                    "Energy Value": ["XOM", "CVX", "COP", "SLB"],
                    "Financial Value": ["JPM", "BAC", "WFC", "GS"],
                    "Industrial Value": ["CAT", "BA", "HON", "UNP"]
                }
                
                for category, tickers in watchlist_data.items():
                    with st.expander(f"{category} ({len(tickers)} stocks)"):
                        ticker_str = " • ".join([f"`{t}`" for t in tickers])
                        st.markdown(ticker_str)
                        st.caption(f"Click on any ticker in the sidebar to analyze")
                                
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Wait a few minutes and try again
        2. Press 'C' to clear cache
        3. Try a different ticker
        4. Use a shorter time period
        """)

else:
    # Welcome message
    st.info("Select a time period and click 'Analyze Stock' to get started!")
    
    st.markdown("""
    ### How to Use:
    1. **Enter a stock ticker** (e.g., AAPL, MSFT, GOOGL)
    2. **Select a time period** (1M, 3M, 6M, 1Y, 2Y, 5Y, or MAX)
    3. **Click "Analyze Stock"**
    
    ### Popular Stock Tickers:
    - **AAPL** - Apple Inc.
    - **MSFT** - Microsoft Corporation
    - **GOOGL** - Alphabet Inc.
    - **AMZN** - Amazon.com Inc.
    - **TSLA** - Tesla Inc.
    - **META** - Meta Platforms Inc.
    - **NVDA** - NVIDIA Corporation
    
    ### Time Periods:
    - **1M** - Last month
    - **3M** - Last 3 months
    - **6M** - Last 6 months
    - **1Y** - Last year (recommended)
    - **2Y** - Last 2 years
    - **5Y** - Last 5 years
    - **MAX** - All available history
    
    ### Tips:
    - Data is cached for 10 minutes
    - Wait 30-60 seconds between different stocks
    - Shorter periods load faster
    """)

st.divider()
st.caption("Data from Yahoo Finance | Cached for 10 minutes")
