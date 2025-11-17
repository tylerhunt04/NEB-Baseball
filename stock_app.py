import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Analyzer")
st.markdown("Real-time stock analysis with Yahoo Finance data")

# Sidebar inputs
st.sidebar.header("Settings")

# Ticker input
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now()
    )

# Download button
analyze_button = st.sidebar.button("📊 Analyze Stock", type="primary")

# Cache the download function with longer TTL to avoid rate limits
@st.cache_data(ttl=600)
def download_stock_data(ticker, start, end):
    """Download stock data with retry logic"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(retry_delay)
            
            df = yf.download(
                ticker, 
                start=start, 
                end=end, 
                progress=False,
                threads=False
            )
            
            if not df.empty:
                # Reset index to make date a column for easier handling
                df = df.reset_index()
                return df, None
            else:
                return None, "No data found for this ticker"
                
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "Too Many Requests" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"⏳ Rate limited. Retrying in {retry_delay} seconds...")
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
        with st.spinner(f"Loading data for {ticker}..."):
            df, error = download_stock_data(ticker, start_date, end_date)
        
        if error:
            st.error(f"❌ {error}")
            
            if "Rate limited" in error:
                st.warning("""
                **You've been rate limited by Yahoo Finance.**
                
                **Solutions:**
                - ⏰ Wait 2-3 minutes before trying again
                - 🔄 Press 'C' to clear cache
                - 🎯 Try a different ticker
                - 📉 Use a shorter date range
                """)
            else:
                st.info("💡 Try: AAPL, MSFT, GOOGL, AMZN, TSLA")
            
        elif df is None or df.empty:
            st.error(f"❌ No data found for '{ticker}'")
            st.info("💡 Try: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA")
            
        else:
            st.success(f"✅ Successfully loaded {len(df)} days of data for {ticker}")
            
            # Get company name
            company_name = ticker
            try:
                stock = yf.Ticker(ticker)
                info = stock.info if hasattr(stock, 'info') else {}
                company_name = info.get('longName', ticker)
            except:
                pass
            
            st.header(f"{company_name} ({ticker})")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            latest_close = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else latest_close
            price_change = latest_close - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            col1.metric("Latest Close", f"${latest_close:.2f}", 
                       f"{price_change:.2f} ({price_change_pct:.2f}%)")
            
            col2.metric("Period High", f"${float(df['High'].max()):.2f}")
            col3.metric("Period Low", f"${float(df['Low'].min()):.2f}")
            col4.metric("Avg Volume", f"{float(df['Volume'].mean())/1e6:.2f}M")
            
            st.divider()
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Raw Data", 
                "📈 Price Chart", 
                "📊 Volume", 
                "📉 Moving Averages",
                "💰 Analysis"
            ])
            
            # Tab 1: Raw Data
            with tab1:
                st.subheader("Historical Stock Data")
                
                # Display without special formatting
                display_df = df.copy()
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"{ticker}_data.csv",
                    mime="text/csv"
                )
            
            # Tab 2: Price Chart
            with tab2:
                st.subheader("Closing Price Chart")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title=f"{ticker} Closing Prices",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Period High", f"${float(df['High'].max()):.2f}")
                col2.metric("Period Low", f"${float(df['Low'].min()):.2f}")
            
            # Tab 3: Volume
            with tab3:
                st.subheader("Trading Volume")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df['Date'],
                    y=df['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title=f"{ticker} Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
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
                    line=dict(color='blue', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_copy['Date'],
                    y=df_copy['MA'],
                    mode='lines',
                    name=f'{ma_days}-Day MA',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{ticker} Price with {ma_days}-Day Moving Average",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 5: Analysis
            with tab5:
                st.subheader("Stock Analysis Summary")
                
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
                    marker_color='lightgreen'
                ))
                
                fig.update_layout(
                    title=f"{ticker} Daily Returns Distribution",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent performance
                st.subheader("Recent Performance (Last 10 Days)")
                recent_data = df.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                st.dataframe(recent_data, use_container_width=True, hide_index=True)
                
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Wait a few minutes and try again
        2. Press 'C' to clear cache
        3. Try a different ticker
        4. Check your internet connection
        """)

else:
    # Welcome message
    st.info("👈 Enter a stock ticker in the sidebar and click 'Analyze Stock' to get started!")
    
    st.markdown("""
    ### Popular Stock Tickers:
    - **AAPL** - Apple Inc.
    - **MSFT** - Microsoft Corporation
    - **GOOGL** - Alphabet Inc.
    - **AMZN** - Amazon.com Inc.
    - **TSLA** - Tesla Inc.
    - **META** - Meta Platforms Inc.
    - **NVDA** - NVIDIA Corporation
    
    ### 💡 Tips:
    - Data is cached for 10 minutes
    - Wait 30-60 seconds between different stocks
    - Use shorter date ranges to load faster
    """)

st.divider()
st.caption("📊 Data from Yahoo Finance | Cached for 10 minutes")
