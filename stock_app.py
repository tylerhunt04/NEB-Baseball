import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import re

# Page config
st.set_page_config(
    page_title="Professional Stock Analyzer • Long-Term Investing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Stock Analysis Tool for Long-Term Investors"
    }
)

# Professional custom CSS - Premium Design
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container - Professional Dark Theme */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    /* Stock Info Card - Premium Header */
    .stock-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 100px rgba(99, 102, 241, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .stock-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    }
    
    .stock-header-container {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 0;
        flex: 1;
    }
    
    .company-logo {
        width: 80px;
        height: 80px;
        min-width: 80px;
        min-height: 80px;
        border-radius: 16px;
        background: white;
        padding: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        object-fit: contain;
        border: 2px solid rgba(99, 102, 241, 0.2);
        flex-shrink: 0;
    }
    
    .stock-info {
        flex: 1;
        min-width: 0;
    }
    
    .stock-symbol {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1;
        letter-spacing: -0.02em;
    }
    
    .stock-company {
        font-size: 1.5rem;
        color: rgba(255, 255, 255, 0.95);
        margin-top: 0.5rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    .stock-meta {
        color: rgba(148, 163, 184, 0.8);
        font-size: 0.875rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Metric Cards - Premium Design */
    .metric-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
        border-radius: 16px;
        padding: 1.75rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-container:hover::before {
        opacity: 1;
    }
    
    .metric-label {
        color: rgba(148, 163, 184, 0.8);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        font-weight: 700;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.02em;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-value.positive {
        color: #10b981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
    }
    
    .metric-value.negative {
        color: #ef4444;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    
    .metric-value.neutral {
        color: #e2e8f0;
        text-shadow: 0 0 20px rgba(226, 232, 240, 0.2);
    }
    
    .metric-change {
        font-size: 0.875rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.875rem;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-change.positive {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .metric-change.negative {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Signal Badges - Premium Design */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.875rem 1.75rem;
        border-radius: 12px;
        font-weight: 800;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .signal-strong-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4);
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(52, 211, 153, 0.4);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(245, 158, 11, 0.4);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(248, 113, 113, 0.4);
    }
    
    .signal-strong-sell {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.4);
    }
    
    /* Info Cards - Premium Design */
    .info-card {
        background: rgba(30, 41, 59, 0.4);
        border-left: 3px solid;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .info-card.success {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.08);
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2);
    }
    
    .info-card.warning {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.08);
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.2);
    }
    
    .info-card.error {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.08);
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.2);
    }
    
    .info-card.info {
        border-left-color: #6366f1;
        background: rgba(99, 102, 241, 0.08);
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(to bottom, #0f172a 0%, #1e293b 100%);
    }
    
    /* Section Headers */
    .section-header {
        color: #6366f1;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        letter-spacing: -0.01em;
    }
    
    /* Divider - Professional */
    .custom-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(99, 102, 241, 0.5), transparent);
        margin: 2rem 0;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(30, 41, 59, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Table Styling */
    .dataframe {
        background: rgba(30, 41, 59, 0.4) !important;
        border-radius: 12px;
    }
    
    /* Tabs Styling - Premium */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.4);
        padding: 0.5rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: rgba(148, 163, 184, 0.8);
        font-weight: 700;
        padding: 0.875rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 0.875rem;
        letter-spacing: 0.02em;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Button Styling - Premium */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 700;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        font-weight: 700;
        color: #a5b4fc;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
    
    /* Success/Warning/Error Messages - Premium */
    .stAlert {
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Metric component styling */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }
    
    /* Input fields */
    .stTextInput input {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        color: white;
        padding: 0.75rem 1rem;
        font-weight: 500;
    }
    
    .stTextInput input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* Radio buttons */
    .stRadio > label {
        background: rgba(30, 41, 59, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stRadio > label:hover {
        background: rgba(99, 102, 241, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0 1.5rem 0;">
    <h1 style="
        font-size: 1.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
    ">📊 Long-Term Analyzer</h1>
    <p style="color: rgba(148, 163, 184, 0.8); font-size: 0.875rem; margin-top: 0.5rem; font-weight: 500;">
        Professional Stock Analysis
    </p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("<div style='height: 1px; background: linear-gradient(to right, transparent, rgba(99, 102, 241, 0.5), transparent); margin: 0 0 1.5rem 0;'></div>", unsafe_allow_html=True)

# Stock input
st.sidebar.markdown("""
<div style="
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
">Stock Selection</div>
""", unsafe_allow_html=True)
ticker = st.sidebar.text_input(
    "",
    value="AAPL",
    placeholder="Enter ticker (e.g., AAPL, GOOGL, MSFT)",
    help="Enter a valid stock ticker symbol",
    label_visibility="collapsed"
).upper()

# Analyze button
analyze_button = st.sidebar.button("🔍 Analyze Stock", type="primary", use_container_width=True)

st.sidebar.markdown("<div style='height: 1px; background: linear-gradient(to right, transparent, rgba(99, 102, 241, 0.3), transparent); margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

# Period selection
st.sidebar.markdown("""
<div style="
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
">Time Period</div>
""", unsafe_allow_html=True)
period = st.sidebar.radio(
    "",
    ["1Y", "2Y", "3Y", "5Y", "MAX"],
    index=1,  # Default to 2Y
    horizontal=False
)

# Map period to yfinance format
period_map = {
    "1Y": "1y",
    "2Y": "2y",
    "3Y": "3y",
    "5Y": "5y",
    "MAX": "max"
}

st.sidebar.markdown("<div style='height: 1px; background: linear-gradient(to right, transparent, rgba(99, 102, 241, 0.3), transparent); margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

# Investment Philosophy
with st.sidebar.expander("💡 Investment Philosophy", expanded=False):
    st.markdown("""
    <div style="font-size: 0.875rem; line-height: 1.6;">
        <strong style="color: #a5b4fc;">This tool focuses on:</strong>
        <ul style="margin: 0.5rem 0; padding-left: 1.25rem;">
            <li style="margin: 0.25rem 0;">Buying quality at reasonable prices</li>
            <li style="margin: 0.25rem 0;">Long-term value creation (3-5+ years)</li>
            <li style="margin: 0.25rem 0;">Fundamental analysis over speculation</li>
            <li style="margin: 0.25rem 0;">Patience over timing the market</li>
        </ul>
        
        <strong style="color: #a5b4fc; display: block; margin-top: 1rem;">Best for:</strong>
        <ul style="margin: 0.5rem 0; padding-left: 1.25rem;">
            <li style="margin: 0.25rem 0;">Buy-and-hold investors</li>
            <li style="margin: 0.25rem 0;">Building wealth over time</li>
            <li style="margin: 0.25rem 0;">Retirement accounts</li>
            <li style="margin: 0.25rem 0;">Patient capital</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

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
                """)
            else:
                st.info("Try: AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, PG")
            
        elif df is None or df.empty:
            st.error(f"No data found for '{ticker}'")
            st.info("Try: AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, PG, KO, PEP")
            
        else:
            st.success(f"Successfully loaded {len(df)} days of data for {ticker} ({period})")
            
            # Get company info and logo
            company_name = ticker
            sector = "N/A"
            market_cap = None
            logo_url = None
            try:
                stock = yf.Ticker(ticker)
                info = stock.info if hasattr(stock, 'info') else {}
                company_name = info.get('longName', ticker)
                sector = info.get('sector', 'N/A')
                market_cap = info.get('marketCap', None)
                logo_url = info.get('logo_url', None)
                
                # Fallback: try to construct logo URL from website domain
                if not logo_url:
                    website = info.get('website', '')
                    if website:
                        # Extract domain from website URL
                        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', website)
                        if domain_match:
                            domain = domain_match.group(1)
                            # Use Clearbit logo API as fallback
                            logo_url = f"https://logo.clearbit.com/{domain}"
            except Exception as e:
                # If logo fetch fails, continue without logo
                pass
            
            # Professional Stock Card
            latest_close = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else latest_close
            price_change = latest_close - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            # Calculate long-term trend
            period_start = float(df['Close'].iloc[0])
            total_return = ((latest_close - period_start) / period_start) * 100
            
            # Determine signal based on long-term trend
            if total_return > 20:
                signal_class = "signal-strong-buy"
                signal_text = "STRONG GROWTH"
            elif total_return > 10:
                signal_class = "signal-buy"
                signal_text = "GROWING"
            elif total_return > -10:
                signal_class = "signal-hold"
                signal_text = "STABLE"
            elif total_return > -20:
                signal_class = "signal-sell"
                signal_text = "DECLINING"
            else:
                signal_class = "signal-strong-sell"
                signal_text = "WEAK"
            
            # Format market cap
            mc_display = "N/A"
            if market_cap:
                if market_cap >= 1e12:
                    mc_display = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    mc_display = f"${market_cap/1e9:.2f}B"
                else:
                    mc_display = f"${market_cap/1e6:.2f}M"
            
            # Build logo HTML separately with error handling
            logo_html = ""
            if logo_url:
                logo_html = f'<img src="{logo_url}" class="company-logo" alt="{ticker} logo" onerror="this.style.display=\'none\'">'
            
            stock_card_html = f"""
            <div class="stock-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="stock-header-container">
                        {logo_html}
                        <div class="stock-info">
                            <h1 class="stock-symbol">{ticker}</h1>
                            <h2 class="stock-company">{company_name}</h2>
                            <p class="stock-meta">{sector} • Market Cap: {mc_display}</p>
                        </div>
                    </div>
                    <div class="signal-badge {signal_class}">
                        {signal_text}
                    </div>
                </div>
            </div>
            """
            
            st.markdown(stock_card_html, unsafe_allow_html=True)
            
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
                total_return_class = "positive" if total_return >= 0 else "negative"
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">{period} Return</div>
                    <div class="metric-value {total_return_class}">{total_return:+.2f}%</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        Long-term Trend
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">52-Week High</div>
                    <div class="metric-value neutral">${float(df['High'].max()):.2f}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        Period High
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">52-Week Low</div>
                    <div class="metric-value neutral">${float(df['Low'].min()):.2f}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        Period Low
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
                        category_msg = "**Mega-Cap Tech** - Quality at a discount. Consider accumulating on 10-20% pullbacks."
                    elif sector in defensive_sectors:
                        category_msg = f"**Defensive {sector}** - Stable performer. Good time to accumulate for long-term."
                    else:
                        category_msg = "Check fundamentals before buying the dip. Review Market Insights tab."
                except:
                    category_msg = "Review the Market Insights tab for detailed strategy guidance."
                
                if drawdown_from_high <= -20:
                    st.error(f"""
                    **DEEP CORRECTION ALERT** - {ticker} is down **{abs(drawdown_from_high):.1f}%** from recent high (${week_52_high:.2f})
                    
                    {category_msg}
                    
                    **For Long-Term Investors**: Deep corrections (20%+) often present the best entry points for quality companies. 
                    See **Market Insights** tab for detailed analysis and opportunity score.
                    """)
                elif drawdown_from_high <= -15:
                    st.warning(f"""
                    **CORRECTION TERRITORY** - {ticker} is down **{abs(drawdown_from_high):.1f}%** from recent high (${week_52_high:.2f})
                    
                    {category_msg}
                    
                    Check the **Market Insights** tab for opportunity analysis and buying strategy.
                    """)
                else:
                    st.info(f"""
                    **PULLBACK DETECTED** - {ticker} is down **{abs(drawdown_from_high):.1f}%** from recent high (${week_52_high:.2f})
                    
                    {category_msg}
                    
                    Monitor for further weakness. See **Market Insights** tab for analysis.
                    """)
            
            # Create tabs - focused on long-term investing
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Fundamentals",
                "📈 Long-Term Trend",
                "🎯 Investment Strategy",
                "📋 Performance Metrics"
            ])
            
            # Tab 1: Fundamentals
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
                    
                    # Business Summary
                    summary = info_data.get('longBusinessSummary', '')
                    if summary:
                        with st.expander("Business Summary"):
                            st.write(summary)
                    
                    st.divider()
                    
                    # Key Financial Metrics - LONG-TERM FOCUS
                    st.markdown("### 🎯 Key Metrics for Long-Term Investors")
                    st.info("These metrics help assess company quality and fair value - critical for buy-and-hold investing.")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("**📊 Valuation**")
                        # P/E Ratios
                        pe_trailing = info_data.get('trailingPE')
                        pe_forward = info_data.get('forwardPE')
                        peg = info_data.get('pegRatio')
                        
                        st.metric("P/E Ratio (TTM)", f"{pe_trailing:.2f}" if pe_trailing else "N/A")
                        st.metric("Forward P/E", f"{pe_forward:.2f}" if pe_forward else "N/A")
                        st.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
                        
                        # Valuation assessment
                        if peg and peg < 1.0:
                            st.success("Potentially undervalued")
                        elif peg and peg < 2.0:
                            st.info("Fairly valued")
                        elif peg:
                            st.warning("Potentially overvalued")
                    
                    with col2:
                        st.markdown("**💰 Profitability**")
                        # Margins
                        profit_margin = info_data.get('profitMargins')
                        roe = info_data.get('returnOnEquity')
                        roa = info_data.get('returnOnAssets')
                        
                        if profit_margin:
                            st.metric("Profit Margin", f"{profit_margin*100:.2f}%")
                            if profit_margin > 0.20:
                                st.success("Excellent margins")
                            elif profit_margin > 0.10:
                                st.info("Good margins")
                            else:
                                st.warning("Low margins")
                        else:
                            st.metric("Profit Margin", "N/A")
                        
                        if roe:
                            st.metric("Return on Equity", f"{roe*100:.2f}%")
                        else:
                            st.metric("Return on Equity", "N/A")
                        
                        if roa:
                            st.metric("Return on Assets", f"{roa*100:.2f}%")
                        else:
                            st.metric("Return on Assets", "N/A")
                    
                    with col3:
                        st.markdown("**📈 Growth**")
                        # Growth metrics
                        revenue_growth = info_data.get('revenueGrowth')
                        earnings_growth = info_data.get('earningsGrowth')
                        
                        if revenue_growth:
                            st.metric("Revenue Growth", f"{revenue_growth*100:.2f}%")
                            if revenue_growth > 0.15:
                                st.success("Strong growth")
                            elif revenue_growth > 0.05:
                                st.info("Moderate growth")
                            else:
                                st.warning("Slow growth")
                        else:
                            st.metric("Revenue Growth", "N/A")
                        
                        if earnings_growth:
                            st.metric("Earnings Growth", f"{earnings_growth*100:.2f}%")
                        else:
                            st.metric("Earnings Growth", "N/A")
                        
                        # Book Value
                        book_value = info_data.get('bookValue')
                        st.metric("Book Value", f"${book_value:.2f}" if book_value else "N/A")
                    
                    with col4:
                        st.markdown("**🛡️ Financial Health**")
                        # Debt metrics
                        debt_to_equity = info_data.get('debtToEquity')
                        current_ratio = info_data.get('currentRatio')
                        
                        if debt_to_equity:
                            st.metric("Debt/Equity", f"{debt_to_equity:.2f}")
                            if debt_to_equity < 0.5:
                                st.success("Very safe")
                            elif debt_to_equity < 1.0:
                                st.info("Reasonable")
                            else:
                                st.warning("High debt")
                        else:
                            st.metric("Debt/Equity", "N/A")
                        
                        if current_ratio:
                            st.metric("Current Ratio", f"{current_ratio:.2f}")
                            if current_ratio > 1.5:
                                st.success("Healthy liquidity")
                            elif current_ratio > 1.0:
                                st.info("Adequate liquidity")
                            else:
                                st.warning("Liquidity concern")
                        else:
                            st.metric("Current Ratio", "N/A")
                        
                        # Total Cash
                        total_cash = info_data.get('totalCash')
                        if total_cash:
                            if total_cash >= 1e9:
                                cash_display = f"${total_cash/1e9:.2f}B"
                            else:
                                cash_display = f"${total_cash/1e6:.2f}M"
                        else:
                            cash_display = "N/A"
                        st.metric("Total Cash", cash_display)
                    
                    st.divider()
                    
                    # Dividends (Important for long-term investors)
                    st.markdown("### 💵 Dividend Information")
                    st.info("Dividends provide passive income and signal financial stability.")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        div_yield = info_data.get('dividendYield')
                        if div_yield:
                            st.metric("Dividend Yield", f"{div_yield*100:.2f}%")
                            if div_yield > 0.04:
                                st.success("Attractive yield")
                            elif div_yield > 0.02:
                                st.info("Moderate yield")
                            else:
                                st.info("Low yield")
                        else:
                            st.metric("Dividend Yield", "N/A")
                            st.warning("No dividend")
                    
                    with col2:
                        div_rate = info_data.get('dividendRate')
                        st.metric("Annual Dividend", f"${div_rate:.2f}" if div_rate else "N/A")
                    
                    with col3:
                        payout = info_data.get('payoutRatio')
                        if payout:
                            st.metric("Payout Ratio", f"{payout*100:.2f}%")
                            if payout < 0.60:
                                st.success("Sustainable")
                            elif payout < 0.80:
                                st.warning("Moderate risk")
                            else:
                                st.error("High risk")
                        else:
                            st.metric("Payout Ratio", "N/A")
                    
                    with col4:
                        ex_div = info_data.get('exDividendDate')
                        if ex_div:
                            from datetime import datetime
                            ex_div_date = datetime.fromtimestamp(ex_div).strftime('%Y-%m-%d')
                            st.metric("Ex-Dividend Date", ex_div_date)
                        else:
                            st.metric("Ex-Dividend Date", "N/A")
                    
                    st.divider()
                    
                    # Price Targets vs Opportunity Score
                    st.markdown("### 🎯 Analyst Price Targets vs Your Opportunity Score")
                    
                    st.info("""
                    **Important Note:** Analyst recommendations and the opportunity score serve different purposes:
                    
                    **Analyst Recommendations** (Wall Street):
                    - Based on 12-month price targets
                    - Often recommend buying near highs if they expect continued growth
                    - Focus on momentum and forward earnings
                    - Good for: Understanding market sentiment
                    
                    **Your Opportunity Score** (Value-Based):
                    - Focuses on buying quality at a discount (10-20% pullbacks)
                    - Contrarian approach: "Be greedy when others are fearful"
                    - Based on current valuation + correction depth
                    - Good for: Long-term value investing
                    
                    **Why they can disagree:**
                    - Analysts: "Buy at $500 because it's going to $600" (momentum)
                    - Opportunity Score: "Wait for it to drop to $400" (value)
                    
                    For long-term investors, the opportunity score approach is generally better - you want to 
                    buy quality companies at reasonable prices, not chase momentum.
                    """)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        target_high = info_data.get('targetHighPrice')
                        st.metric("Analyst Target High", f"${target_high:.2f}" if target_high else "N/A")
                    
                    with col2:
                        target_mean = info_data.get('targetMeanPrice')
                        st.metric("Analyst Target Mean", f"${target_mean:.2f}" if target_mean else "N/A")
                        
                        if target_mean:
                            upside = ((target_mean - latest_close) / latest_close) * 100
                            if upside > 0:
                                st.success(f"+{upside:.1f}% upside potential")
                            else:
                                st.error(f"{upside:.1f}% downside risk")
                    
                    with col3:
                        target_low = info_data.get('targetLowPrice')
                        st.metric("Analyst Target Low", f"${target_low:.2f}" if target_low else "N/A")
                    
                    recommendation = info_data.get('recommendationKey', 'N/A')
                    if recommendation != 'N/A':
                        if recommendation in ['buy', 'strong_buy']:
                            st.success(f"**Analyst Consensus:** {recommendation.upper().replace('_', ' ')}")
                        elif recommendation == 'hold':
                            st.info(f"**Analyst Consensus:** {recommendation.upper()}")
                        else:
                            st.warning(f"**Analyst Consensus:** {recommendation.upper().replace('_', ' ')}")
                    
                    st.caption("""
                    💡 **Investment Tip**: Use analyst targets for context, but don't follow them blindly. 
                    Focus on buying quality companies when they're on sale (see Market Insights tab for your opportunity score).
                    """)
                    
                except Exception as e:
                    st.error("Unable to load fundamental data. This may be due to API limitations.")
                    st.info(f"Error details: {str(e)}")
                
                # Educational section
                with st.expander("📚 Understanding These Metrics"):
                    st.markdown("""
                    ### Quick Guide to Fundamental Analysis
                    
                    #### ✅ What Makes a Good Long-Term Investment?
                    
                    **Valuation (Is it fairly priced?)**
                    - **PEG Ratio < 1.0** = Potentially undervalued
                    - **PEG Ratio 1.0-2.0** = Fairly valued
                    - **PEG Ratio > 2.0** = Potentially overvalued
                    
                    **Profitability (Is the business healthy?)**
                    - **Profit Margin > 15%** = Strong profitability
                    - **ROE > 15%** = Efficient use of shareholder money
                    - **Consistent margins** = Business stability
                    
                    **Growth (Is the business expanding?)**
                    - **Revenue Growth > 10%** = Strong expansion
                    - **Earnings Growth > Revenue Growth** = Improving efficiency
                    - **Sustainable growth** = Not too fast (risky), not too slow
                    
                    **Financial Health (Can they weather storms?)**
                    - **Debt/Equity < 1.0** = Manageable debt
                    - **Current Ratio > 1.5** = Can pay short-term bills
                    - **Strong cash position** = Financial flexibility
                    
                    **Dividends (Bonus income)**
                    - **Yield 2-6%** = Attractive income
                    - **Payout Ratio < 60%** = Sustainable and room to grow
                    - **Consistent/growing dividends** = Financial stability
                    
                    #### 🎯 The Ideal Long-Term Stock:
                    - PEG Ratio < 1.5 (reasonable valuation)
                    - Profit Margin > 15% (strong profitability)
                    - Revenue & Earnings Growth > 10% (expanding business)
                    - Debt/Equity < 1.0 (financial stability)
                    - Optional: Dividend yield 2-4% (income + growth)
                    
                    **Remember**: No stock is perfect. Look for 3-4 strong metrics, not perfection!
                    """)
            
            # Tab 2: Long-Term Trend
            with tab2:
                st.subheader(f"Long-Term Price Trend ({period})")
                
                st.info("For long-term investors, focus on the overall trend, not daily fluctuations. Zoom out to see the big picture.")
                
                # Simple line chart - no clutter
                chart_df = df.copy()
                
                fig = go.Figure()
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#00ff00', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.1)'
                ))
                
                # Add 50-day and 200-day moving averages
                if len(df) >= 200:
                    chart_df['MA50'] = chart_df['Close'].rolling(window=50).mean()
                    chart_df['MA200'] = chart_df['Close'].rolling(window=200).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=chart_df['Date'],
                        y=chart_df['MA50'],
                        mode='lines',
                        name='50-Day MA',
                        line=dict(color='#ff6600', width=2, dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=chart_df['Date'],
                        y=chart_df['MA200'],
                        mode='lines',
                        name='200-Day MA',
                        line=dict(color='#6666ff', width=2, dash='dot')
                    ))
                
                fig.update_layout(
                    title=f"{ticker} Long-Term Trend - {period}",
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
                    ),
                    legend=dict(
                        bgcolor='#2e2e2e',
                        bordercolor='#555'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend Analysis
                st.markdown("### Trend Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(f"{period} Return", f"{total_return:+.2f}%")
                    if total_return > 0:
                        st.success("Positive long-term trend")
                    else:
                        st.error("Negative long-term trend")
                
                with col2:
                    # Calculate annualized return
                    years = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "MAX": 10}[period]
                    if period == "MAX":
                        years = len(df) / 252  # Trading days
                    
                    annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
                    st.metric("Annualized Return", f"{annualized_return:+.2f}%")
                    if annualized_return > 10:
                        st.success("Strong long-term performance")
                    elif annualized_return > 5:
                        st.info("Moderate long-term performance")
                    else:
                        st.warning("Below-average performance")
                
                with col3:
                    # Volatility
                    daily_returns = df['Close'].pct_change()
                    volatility = float(daily_returns.std() * (252 ** 0.5)) * 100  # Annualized
                    st.metric("Volatility (Annual)", f"{volatility:.2f}%")
                    if volatility < 20:
                        st.success("Low volatility - stable stock")
                    elif volatility < 40:
                        st.info("Moderate volatility - typical stock")
                    else:
                        st.warning("High volatility - risky stock")
                
                # Moving Average Analysis
                if len(df) >= 200:
                    st.markdown("### Moving Average Analysis")
                    
                    latest_price = float(chart_df['Close'].iloc[-1])
                    ma50_latest = float(chart_df['MA50'].iloc[-1]) if not pd.isna(chart_df['MA50'].iloc[-1]) else None
                    ma200_latest = float(chart_df['MA200'].iloc[-1]) if not pd.isna(chart_df['MA200'].iloc[-1]) else None
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if ma50_latest:
                            st.markdown("**50-Day Moving Average**")
                            st.metric("50-Day MA", f"${ma50_latest:.2f}")
                            
                            if latest_price > ma50_latest:
                                diff_pct = ((latest_price - ma50_latest) / ma50_latest) * 100
                                st.success(f"Price is {diff_pct:.1f}% above 50-day MA - Short-term bullish")
                            else:
                                diff_pct = ((ma50_latest - latest_price) / ma50_latest) * 100
                                st.error(f"Price is {diff_pct:.1f}% below 50-day MA - Short-term bearish")
                    
                    with col2:
                        if ma200_latest:
                            st.markdown("**200-Day Moving Average**")
                            st.metric("200-Day MA", f"${ma200_latest:.2f}")
                            
                            if latest_price > ma200_latest:
                                diff_pct = ((latest_price - ma200_latest) / ma200_latest) * 100
                                st.success(f"Price is {diff_pct:.1f}% above 200-day MA - Long-term bullish")
                            else:
                                diff_pct = ((ma200_latest - latest_price) / ma200_latest) * 100
                                st.error(f"Price is {diff_pct:.1f}% below 200-day MA - Long-term bearish")
                    
                    # Golden/Death Cross
                    if ma50_latest and ma200_latest:
                        if ma50_latest > ma200_latest:
                            st.success("✅ **Golden Cross** - 50-day MA above 200-day MA: Long-term uptrend confirmed")
                        else:
                            st.error("⚠️ **Death Cross** - 50-day MA below 200-day MA: Long-term downtrend confirmed")
                
                with st.expander("📚 How to Interpret This Chart"):
                    st.markdown("""
                    ### Understanding Long-Term Trends
                    
                    **What You're Looking At:**
                    - **Green Area**: Stock price over time
                    - **Orange Dashed Line**: 50-day moving average (short-term trend)
                    - **Blue Dotted Line**: 200-day moving average (long-term trend)
                    
                    **Key Patterns for Long-Term Investors:**
                    
                    ✅ **Strong Buy Signals:**
                    - Stock price consistently above both 50-day and 200-day MA
                    - 50-day MA above 200-day MA (Golden Cross)
                    - Steady upward trend over multiple years
                    - Pullbacks that bounce off the 200-day MA (support)
                    
                    ⚠️ **Warning Signals:**
                    - Stock price consistently below both moving averages
                    - 50-day MA below 200-day MA (Death Cross)
                    - Downward trend over multiple years
                    - Failing to recover to moving averages
                    
                    **The Best Entry Points:**
                    - When price pulls back to the 50-day or 200-day MA in an uptrend
                    - After a correction of 10-20% in a strong company
                    - When Golden Cross just occurred
                    
                    **Long-Term Investor Mindset:**
                    - Ignore daily/weekly noise - focus on the multi-year trend
                    - Use pullbacks as buying opportunities (if fundamentals are strong)
                    - Don't panic sell on temporary dips
                    - Time in the market > timing the market
                    """)
            
            # Tab 3: Investment Strategy (Replaces old Market Insights)
            with tab3:
                st.markdown("### Investment Strategy & Opportunity Analysis")
                
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
                    mega_cap_tech = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
                    defensive_sectors = ['Healthcare', 'Consumer Defensive', 'Utilities']
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
                    correction_desc = "Within 5% of recent high - Exercise caution on new purchases"
                
                # Opportunity Score Calculation
                score = 0
                score_factors = []
                
                # Factor 1: Correction depth (0-3 points)
                if drawdown_from_high <= -20:
                    score += 3
                    score_factors.append("✅ Deep correction (3 pts)")
                elif drawdown_from_high <= -10:
                    score += 2
                    score_factors.append("✅ In correction (2 pts)")
                elif drawdown_from_high <= -5:
                    score += 1
                    score_factors.append("⚠️ Mild pullback (1 pt)")
                else:
                    score_factors.append("❌ Near highs (0 pts)")
                
                # Factor 2: Fundamentals (0-3 points)
                if pe_ratio and pe_ratio < 25:
                    score += 2
                    score_factors.append("✅ Reasonable P/E (2 pts)")
                elif pe_ratio and pe_ratio < 35:
                    score += 1
                    score_factors.append("⚠️ Moderate P/E (1 pt)")
                else:
                    score_factors.append("❌ High/No P/E (0 pts)")
                
                # Factor 3: Category strength (0-2 points)
                if category_type in ['mega_cap', 'defensive']:
                    score += 2
                    score_factors.append("✅ Strong category (2 pts)")
                elif category_type in ['large_cap', 'value']:
                    score += 1
                    score_factors.append("⚠️ Good category (1 pt)")
                else:
                    score_factors.append("❌ Higher risk (0 pts)")
                
                # Factor 4: Distance from low (0-2 points)
                if distance_from_low > 50:
                    score += 1
                    score_factors.append("✅ Well above lows (1 pt)")
                elif distance_from_low > 100:
                    score += 2
                    score_factors.append("✅ Far from lows (2 pts)")
                else:
                    score_factors.append("❌ Near lows (0 pts)")
                
                max_score = 10
                opportunity_score = (score / max_score) * 10
                
                # Strategy recommendations by category
                strategy_guide = {
                    "mega_cap": {
                        "strategy": "Accumulate on 10-20% pullbacks",
                        "reasoning": "These market leaders don't disappear in downturns; they just go on sale. A 10-20% drop in a mega-cap tech stock is often a gift for long-term investors.",
                        "hold_period": "Long-term (3-5+ years)",
                        "risk_level": "Low to Moderate"
                    },
                    "defensive": {
                        "strategy": "Steady accumulation during market volatility",
                        "reasoning": "Defensive sectors provide stability during economic uncertainty. Essential services with steady cash flow and reliable dividends make excellent long-term holdings.",
                        "hold_period": "Long-term (5+ years)",
                        "risk_level": "Low"
                    },
                    "value": {
                        "strategy": "Buy when P/E is below historical average",
                        "reasoning": "Quality value stocks often overlooked by growth obsession. Trading below historical averages presents opportunity for patient investors.",
                        "hold_period": "Medium to Long-term (2-5 years)",
                        "risk_level": "Moderate"
                    },
                    "large_cap": {
                        "strategy": "Buy on weakness, verify fundamentals first",
                        "reasoning": "Established companies with proven business models. Verify financial health and competitive position before buying dips.",
                        "hold_period": "Long-term (3-5 years)",
                        "risk_level": "Moderate"
                    },
                    "mid_cap": {
                        "strategy": "Selective buying on deep corrections only",
                        "reasoning": "More volatile but with growth potential. Buy only on significant pullbacks (15%+) and with strong fundamentals.",
                        "hold_period": "Medium-term (2-3 years)",
                        "risk_level": "Moderate to High"
                    },
                    "small_cap": {
                        "strategy": "High risk - Requires deep conviction",
                        "reasoning": "Highest volatility and risk. Requires thorough research and strong conviction. Use small position sizes and longer holding periods.",
                        "hold_period": "Variable (1-3 years)",
                        "risk_level": "High"
                    }
                }
                
                strategy = strategy_guide.get(category_type, strategy_guide["large_cap"])
                
                # Display Correction Analysis
                st.markdown("### 📊 Current Valuation Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="info-card {correction_color}">
                        <h4>{correction_status}</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                            {drawdown_from_high:.1f}%
                        </p>
                        <p style="font-size: 0.85rem; opacity: 0.8;">from recent high</p>
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
                    score_label = "Strong Buy" if opportunity_score >= 7 else ("Moderate" if opportunity_score >= 5 else "Weak/Wait")
                    st.markdown(f"""
                    <div class="info-card {score_color}">
                        <h4>Opportunity Score</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                            {opportunity_score:.1f}/10
                        </p>
                        <p style="font-size: 0.85rem; opacity: 0.8;">
                            {score_label}
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
                st.markdown("### 🎯 Investment Category & Recommended Strategy")
                
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
                st.markdown("### 📋 Opportunity Score Breakdown")
                
                st.markdown("""
                <div class="metric-container" style="margin-bottom: 1rem;">
                """, unsafe_allow_html=True)
                
                for factor in score_factors:
                    st.markdown(f"**{factor}**")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Investment Decision
                st.markdown("### 💡 Investment Decision")
                
                if opportunity_score >= 7:
                    st.success("""
                    **STRONG BUYING OPPORTUNITY**
                    
                    Multiple positive factors align for a potential entry:
                    - ✅ Stock is significantly below recent highs
                    - ✅ Fundamentals appear reasonable
                    - ✅ Category fits long-term strategy
                    
                    **Action:** Consider starting or adding to position with proper position sizing (2-5% of portfolio).
                    Set a 3-5 year hold period and use pullbacks to add more shares.
                    """)
                elif opportunity_score >= 5:
                    st.warning("""
                    **MODERATE OPPORTUNITY**
                    
                    Some positive factors present:
                    - ⚠️ May be worth watching for further weakness
                    - ⚠️ Verify fundamentals thoroughly before entry
                    - ⚠️ Consider waiting for a better entry point (10-15% lower)
                    
                    **Action:** Add to watchlist. Set price alerts for 10-15% below current level.
                    Not a clear buy signal yet - patience recommended.
                    """)
                else:
                    st.error("""
                    **WEAK OPPORTUNITY**
                    
                    Limited positive factors:
                    - ❌ Stock may be overvalued or near highs
                    - ❌ Fundamentals may not support current price
                    - ❌ Higher risk category or weak technicals
                    
                    **Action:** Pass for now. Better opportunities likely available elsewhere.
                    If you believe in the long-term story, wait for a 15-20% correction before considering entry.
                    """)
                
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                
                # Position Sizing Guide
                st.markdown("### 💼 Position Sizing for Long-Term Investors")
                
                st.info("""
                **Golden Rules of Position Sizing:**
                
                Never put all your eggs in one basket! Even the best companies can stumble.
                """)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    **Conservative**
                    - 2-3% per position
                    - 15-25 total holdings
                    - Lower risk tolerance
                    - Nearing retirement
                    """)
                
                with col2:
                    st.markdown("""
                    **Moderate** (Recommended)
                    - 3-5% per position
                    - 10-20 total holdings
                    - Balanced approach
                    - Long time horizon
                    """)
                
                with col3:
                    st.markdown("""
                    **Aggressive**
                    - 5-10% per position
                    - 8-15 total holdings
                    - Higher risk tolerance
                    - High conviction + long horizon
                    """)
                
                st.warning("""
                **Never exceed 10% in a single position**, even for your highest conviction ideas. 
                Diversification protects you from single-stock disasters.
                """)
                
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                
                # Market Philosophy
                st.markdown("### 📖 Long-Term Investment Philosophy")
                
                with st.expander("Understanding Market Corrections"):
                    st.markdown("""
                    ### The Nature of Market Corrections
                    
                    Market corrections are not crises—they are **predictable cycles** and **opportunities**.
                    
                    **Why Corrections Happen:**
                    - Stocks run up too fast → natural pullback
                    - Economic concerns → temporary fear
                    - Profit-taking → healthy reset
                    - Interest rate changes → valuation adjustments
                    
                    **The Opportunity in Volatility:**
                    
                    For strategic investors, volatility is not a threat; it's a **signal of opportunity**.
                    
                    This is when:
                    - Quality companies go "on sale"
                    - Assets return to rational prices
                    - Long-term wealth is built
                    - Careful planning pays off
                    
                    **Historical Truth:**
                    - **Every major correction** has been followed by new all-time highs
                    - **Best buying opportunities** come during corrections
                    - **Patient investors** who bought in 2008, 2020 were rewarded handsomely
                    
                    ---
                    
                    ### Investment Categories
                    
                    **Mega-Cap Technology** (AAPL, MSFT, GOOGL, AMZN, NVDA)
                    - Strategy: Buy on 10-20% pullbacks
                    - Why: Market leaders, fortress balance sheets, durable moats
                    - Don't disappear in downturns—they just go on sale
                    
                    **Defensive Stalwarts** (Healthcare, Consumer Staples, Utilities)
                    - Strategy: Accumulate during market volatility
                    - Why: Essential services, steady cash flow, reliable dividends
                    - Your defensive buffer against fear
                    
                    **Undervalued Value** (Energy, Financials, Industrials)
                    - Strategy: Buy when P/E below historical average
                    - Why: Overlooked by growth obsession, capital rotating here
                    - Compelling for patient value investors
                    
                    ---
                    
                    ### Key Principles
                    
                    1. **Volatility Creates Discounts**
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
                    
                    > "Be fearful when others are greedy, and greedy when others are fearful."  
                    > — Warren Buffett
                    
                    The best investors are made during corrections, not bull markets.
                    """)
                
                with st.expander("When to Buy vs When to Wait"):
                    st.markdown("""
                    ### Making Smart Buy Decisions
                    
                    **✅ DO BUY WHEN:**
                    - Opportunity score ≥ 7
                    - Stock down 10-20%+ from highs
                    - Fundamentals remain strong (check Fundamentals tab)
                    - Category fits your strategy
                    - You have 3-5+ year time horizon
                    - You're using proper position sizing (2-5%)
                    - You have emergency fund in place
                    
                    **❌ DON'T BUY WHEN:**
                    - Opportunity score < 5
                    - Stock near all-time highs
                    - Fundamentals deteriorating
                    - Just because price is falling (check why!)
                    - You need the money within 2 years
                    - You're at max position size
                    - Buying on margin/leverage
                    
                    **⏸️ WAIT WHEN:**
                    - Score is 5-7 (moderate opportunity)
                    - Mixed signals from fundamentals
                    - Major news pending (earnings, FDA decision, etc.)
                    - Already at target position size
                    - Better opportunities elsewhere
                    
                    ---
                    
                    ### The Patient Investor's Advantage
                    
                    **Why Patience Wins:**
                    - Better entry prices mean higher returns
                    - Less stress during market swings
                    - More dry powder for opportunities
                    - Avoid FOMO mistakes
                    
                    **How to Practice Patience:**
                    1. Set price alerts at your target buy level
                    2. Keep a watchlist of quality stocks
                    3. Have cash ready (10-20% of portfolio)
                    4. Don't chase momentum
                    5. Remember: there's always another opportunity
                    """)
            
            # Tab 4: Performance Metrics
            with tab4:
                st.subheader(f"Long-Term Performance Analysis ({period})")
                
                # Calculate statistics
                first_close = float(df['Close'].iloc[0])
                last_close = float(df['Close'].iloc[-1])
                total_return = ((last_close - first_close) / first_close) * 100
                
                # Annualized metrics
                years = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "MAX": 10}[period]
                if period == "MAX":
                    years = len(df) / 252
                
                annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
                
                daily_returns = df['Close'].pct_change()
                annual_volatility = float(daily_returns.std() * (252 ** 0.5)) * 100
                
                # Sharpe Ratio (assuming 4% risk-free rate)
                risk_free_rate = 4.0
                excess_return = annualized_return - risk_free_rate
                sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                return_class = "positive" if total_return >= 0 else "negative"
                col1.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {return_class}">{total_return:+.2f}%</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        Over {period}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                annual_class = "positive" if annualized_return >= 10 else ("neutral" if annualized_return >= 0 else "negative")
                col2.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Annualized Return</div>
                    <div class="metric-value {annual_class}">{annualized_return:+.2f}%</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        Per Year Average
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                vol_class = "positive" if annual_volatility < 25 else ("neutral" if annual_volatility < 40 else "negative")
                col3.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Annual Volatility</div>
                    <div class="metric-value {vol_class}">{annual_volatility:.2f}%</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        Risk Measure
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                sharpe_class = "positive" if sharpe_ratio > 1.0 else ("neutral" if sharpe_ratio > 0.5 else "negative")
                col4.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {sharpe_class}">{sharpe_ratio:.2f}</div>
                    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                        Risk-Adj Return
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                # Performance interpretation
                st.markdown("### 📊 Performance Interpretation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Annualized Return**")
                    if annualized_return > 15:
                        st.success(f"✅ Excellent performance ({annualized_return:.1f}%) - Beating most benchmarks")
                    elif annualized_return > 10:
                        st.success(f"✅ Strong performance ({annualized_return:.1f}%) - Above S&P 500 historical average")
                    elif annualized_return > 5:
                        st.info(f"⚠️ Moderate performance ({annualized_return:.1f}%) - Below market average")
                    elif annualized_return > 0:
                        st.warning(f"⚠️ Weak performance ({annualized_return:.1f}%) - Underperforming")
                    else:
                        st.error(f"❌ Negative performance ({annualized_return:.1f}%) - Losing money")
                
                with col2:
                    st.markdown("**Risk Level (Volatility)**")
                    if annual_volatility < 20:
                        st.success(f"✅ Low volatility ({annual_volatility:.1f}%) - Stable, low-risk stock")
                    elif annual_volatility < 30:
                        st.info(f"⚠️ Moderate volatility ({annual_volatility:.1f}%) - Typical stock")
                    elif annual_volatility < 50:
                        st.warning(f"⚠️ High volatility ({annual_volatility:.1f}%) - Risky, requires conviction")
                    else:
                        st.error(f"❌ Very high volatility ({annual_volatility:.1f}%) - Extreme risk")
                
                st.markdown("**Sharpe Ratio (Risk-Adjusted Returns)**")
                if sharpe_ratio > 2.0:
                    st.success(f"✅ Excellent ({sharpe_ratio:.2f}) - Great returns relative to risk")
                elif sharpe_ratio > 1.0:
                    st.success(f"✅ Good ({sharpe_ratio:.2f}) - Positive risk-adjusted returns")
                elif sharpe_ratio > 0:
                    st.info(f"⚠️ Acceptable ({sharpe_ratio:.2f}) - Returns barely compensate for risk")
                else:
                    st.error(f"❌ Poor ({sharpe_ratio:.2f}) - Not compensating for risk taken")
                
                st.divider()
                
                # Historical comparison
                st.markdown("### 📊 Benchmark Comparison")
                
                st.info("""
                **S&P 500 Historical Performance:**
                - Long-term average: ~10% annual return
                - Low volatility: ~15-20% annually
                - Sharpe Ratio: ~0.5-0.7
                
                Compare {ticker}'s metrics above to these benchmarks.
                """.format(ticker=ticker))
                
                # Best/worst periods
                st.markdown("### 📈 Best & Worst Periods")
                
                # Calculate rolling returns
                if len(df) >= 90:
                    df_analysis = df.copy()
                    df_analysis['90d_return'] = df_analysis['Close'].pct_change(90) * 100
                    
                    best_90d = df_analysis['90d_return'].max()
                    worst_90d = df_analysis['90d_return'].min()
                    
                    best_date = df_analysis[df_analysis['90d_return'] == best_90d]['Date'].iloc[0]
                    worst_date = df_analysis[df_analysis['90d_return'] == worst_90d]['Date'].iloc[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success(f"""
                        **Best 90-Day Period**
                        - Return: +{best_90d:.1f}%
                        - Ending: {best_date.strftime('%Y-%m-%d') if hasattr(best_date, 'strftime') else str(best_date)[:10]}
                        """)
                    
                    with col2:
                        st.error(f"""
                        **Worst 90-Day Period**
                        - Return: {worst_90d:.1f}%
                        - Ending: {worst_date.strftime('%Y-%m-%d') if hasattr(worst_date, 'strftime') else str(worst_date)[:10]}
                        """)
                    
                    st.caption("90-day periods show how volatile the stock can be in the short-term. Long-term investors should focus on multi-year trends.")
                
                st.divider()
                
                # What-if calculator
                st.markdown("### 💰 Investment Calculator")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    investment_amount = st.number_input(
                        "Initial Investment ($)",
                        min_value=100,
                        max_value=1000000,
                        value=10000,
                        step=100
                    )
                
                with col2:
                    monthly_contribution = st.number_input(
                        "Monthly Contribution ($)",
                        min_value=0,
                        max_value=10000,
                        value=0,
                        step=50
                    )
                
                if annualized_return > -50:  # Reasonable bounds
                    # Calculate future value
                    months_in_period = years * 12
                    monthly_return = (1 + annualized_return/100) ** (1/12) - 1
                    
                    # Future value with monthly contributions
                    if monthly_contribution > 0:
                        fv = investment_amount * (1 + annualized_return/100) ** years
                        fv += monthly_contribution * (((1 + monthly_return) ** months_in_period - 1) / monthly_return) * (1 + monthly_return)
                    else:
                        fv = investment_amount * (1 + annualized_return/100) ** years
                    
                    total_invested = investment_amount + (monthly_contribution * months_in_period)
                    profit = fv - total_invested
                    profit_pct = (profit / total_invested) * 100 if total_invested > 0 else 0
                    
                    st.markdown("### 📊 Projected Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric("Total Invested", f"${total_invested:,.2f}")
                    col2.metric("Future Value", f"${fv:,.2f}")
                    
                    profit_class = "positive" if profit >= 0 else "negative"
                    col3.metric("Profit/Loss", f"${profit:,.2f} ({profit_pct:+.1f}%)")
                    
                    st.caption(f"""
                    *Based on historical {annualized_return:.1f}% annualized return over {years:.1f} years. 
                    Past performance does not guarantee future results.*
                    """)
                
                st.divider()
                
                # Educational section
                with st.expander("📚 Understanding These Metrics"):
                    st.markdown("""
                    ### Performance Metrics Explained
                    
                    **Total Return**
                    - Simple percentage gain or loss over the period
                    - Example: $100 → $125 = 25% total return
                    - Doesn't account for time or volatility
                    
                    **Annualized Return** ⭐ Most Important
                    - Average return per year
                    - Makes it easy to compare different time periods
                    - **>15%** = Excellent
                    - **10-15%** = Strong (beats S&P 500 average)
                    - **5-10%** = Moderate (below market average)
                    - **<5%** = Weak (could be doing better)
                    
                    **Volatility (Risk)**
                    - How much the stock bounces around
                    - Higher volatility = higher risk
                    - **<20%** = Stable (like utilities)
                    - **20-40%** = Normal (most stocks)
                    - **>40%** = Risky (tech growth, small caps)
                    
                    **Sharpe Ratio** (Risk-Adjusted Returns)
                    - Return per unit of risk taken
                    - Accounts for volatility
                    - **>2.0** = Excellent risk-adjusted returns
                    - **1.0-2.0** = Good risk-adjusted returns
                    - **0.5-1.0** = Acceptable
                    - **<0.5** = Poor (not worth the risk)
                    
                    ---
                    
                    ### What Matters for Long-Term Investors?
                    
                    1. **Annualized Return** - Your compound growth rate
                    2. **Sharpe Ratio** - Making sure you're compensated for risk
                    3. **Consistency** - Steady growth better than wild swings
                    4. **Time Horizon** - The longer you hold, the more volatility smooths out
                    
                    **Key Insight**: A stock with 12% annual return and low volatility can beat 
                    a stock with 15% return and extreme volatility over the long term, because 
                    you're less likely to panic sell during crashes!
                    """)
                
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Wait a few minutes and try again
        2. Press 'C' to clear cache
        3. Try a different ticker
        """)

else:
    # Welcome message
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem;">
        <h1 style="
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            letter-spacing: -0.02em;
        ">👋 Welcome to Long-Term Stock Analyzer</h1>
        <p style="
            font-size: 1.25rem;
            color: rgba(148, 163, 184, 0.9);
            margin-bottom: 2rem;
            font-weight: 500;
        ">Professional analysis tool for buy-and-hold investors</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("This tool is designed specifically for buy-and-hold investors who focus on fundamentals and long-term value creation.")
    
    st.markdown("""
    ### 🎯 How to Use:
    1. **Enter a stock ticker** (e.g., AAPL, MSFT, JPM, JNJ, PG)
    2. **Select a time period** (1Y, 2Y, 3Y, 5Y, or MAX)
    3. **Click "Analyze Stock"**
    
    ### 📊 What This Tool Provides:
    
    **📈 Fundamentals Tab**
    - Company overview and business summary
    - Valuation metrics (P/E, PEG, Price/Book)
    - Profitability metrics (Margins, ROE, ROA)
    - Growth metrics (Revenue, Earnings)
    - Financial health (Debt, Cash, Liquidity)
    - Dividend information
    - Analyst vs Opportunity Score comparison
    
    **📊 Long-Term Trend Tab**
    - Multi-year price chart
    - 50-day and 200-day moving averages
    - Golden/Death cross identification
    - Annualized returns
    - Volatility analysis
    
    **🎯 Investment Strategy Tab**
    - Current valuation analysis
    - Opportunity score (0-10)
    - Stock category classification
    - Recommended strategy for your category
    - Position sizing guidance
    - Buy vs Wait decision framework
    
    **📋 Performance Metrics Tab**
    - Total and annualized returns
    - Risk-adjusted returns (Sharpe Ratio)
    - Volatility assessment
    - Best/worst periods
    - Investment calculator
    
    ### 💡 Philosophy:
    - Buy quality companies at reasonable prices
    - Focus on 3-5+ year time horizons
    - Use corrections as buying opportunities
    - Diversify across 10-20 positions
    - Position size: 2-5% per stock
    - Patience over timing
    
    ### 🏆 Example Stocks to Try:
    
    **Mega-Cap Tech:**
    - AAPL (Apple), MSFT (Microsoft), GOOGL (Alphabet)
    - AMZN (Amazon), NVDA (Nvidia), META (Meta)
    
    **Defensive Stocks:**
    - JNJ (Johnson & Johnson), PG (Procter & Gamble)
    - KO (Coca-Cola), PEP (PepsiCo)
    
    **Financial Value:**
    - JPM (JPMorgan), BAC (Bank of America)
    - V (Visa), MA (Mastercard)
    
    **Healthcare:**
    - UNH (UnitedHealth), ABBV (AbbVie)
    - PFE (Pfizer), LLY (Eli Lilly)
    """)
    
    st.success("👆 Enter a ticker in the sidebar to get started!")

st.divider()
st.markdown("""
<div style="text-align: center; padding: 1rem; color: rgba(148, 163, 184, 0.6); font-size: 0.875rem;">
    <p style="margin: 0;">Data from Yahoo Finance • Cached for 10 minutes</p>
    <p style="margin: 0.5rem 0 0 0;">For educational purposes only • Not financial advice</p>
</div>
""", unsafe_allow_html=True)
