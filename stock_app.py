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

# Main content
if analyze_button or ticker:
    try:
        # Download data using yf.download (most reliable method)
        with st.spinner(f"Loading data for {ticker}..."):
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            st.error(f"❌ No data found for ticker '{ticker}'. Please check the symbol and try again.")
            st.info("💡 Try these tickers: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA")
        else:
            # Get additional info
            stock = yf.Ticker(ticker)
            info = stock.info if hasattr(stock, 'info') else {}
            
            # Company name
            company_name = info.get('longName', ticker)
            st.header(f"{company_name} ({ticker})")
            
            # Key metrics at top
            col1, col2, col3, col4 = st.columns(4)
            
            latest_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
            price_change = latest_close - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            col1.metric("Latest Close", f"${latest_close:.2f}", 
                       f"{price_change:.2f} ({price_change_pct:.2f}%)")
            
            col2.metric("Period High", f"${df['High'].max():.2f}")
            col3.metric("Period Low", f"${df['Low'].min():.2f}")
            col4.metric("Avg Volume", f"{df['Volume'].mean()/1e6:.2f}M")
            
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
                st.dataframe(df, use_container_width=True)
                
                # Download CSV
                csv = df.to_csv()
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
                    x=df.index,
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
                
                # Show high and low
                col1, col2 = st.columns(2)
                col1.metric("Period High", f"${df['High'].max():.2f}")
                col2.metric("Period Low", f"${df['Low'].min():.2f}")
            
            # Tab 3: Volume
            with tab3:
                st.subheader("Trading Volume")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df.index,
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
                
                # MA selector
                ma_days = st.slider("Select Moving Average Period (days)", 
                                   min_value=5, max_value=200, value=20, step=5)
                
                # Calculate moving average
                df['MA'] = df['Close'].rolling(window=ma_days).mean()
                
                fig = go.Figure()
                
                # Close price
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=1)
                ))
                
                # Moving average
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['MA'],
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
                total_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                volatility = df['Close'].pct_change().std() * 100
                
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Total Return", f"{total_return:.2f}%")
                col2.metric("Daily Volatility", f"{volatility:.2f}%")
                col3.metric("Data Points", len(df))
                
                st.divider()
                
                # Show daily returns
                st.subheader("Daily Returns Distribution")
                daily_returns = df['Close'].pct_change().dropna() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=daily_returns,
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
                st.subheader("Recent Performance")
                recent_data = df.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']]
                st.dataframe(recent_data, use_container_width=True)
                
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. Make sure yfinance is installed: `pip install yfinance`
        2. Check your internet connection
        3. Verify the ticker symbol is correct
        4. Try a different date range
        """)

else:
    # Welcome message
    st.info("👈 Enter a stock ticker in the sidebar and click 'Analyze Stock' to get started!")
    
    st.markdown("""
    ### Popular Stock Tickers:
    - **AAPL** - Apple Inc.
    - **MSFT** - Microsoft Corporation
    - **GOOGL** - Alphabet Inc. (Google)
    - **AMZN** - Amazon.com Inc.
    - **TSLA** - Tesla Inc.
    - **META** - Meta Platforms Inc.
    - **NVDA** - NVIDIA Corporation
    - **JPM** - JPMorgan Chase & Co.
    - **V** - Visa Inc.
    - **WMT** - Walmart Inc.
    """)

# Footer
st.divider()
st.caption("📊 Data provided by Yahoo Finance via yfinance library | For educational purposes only")
