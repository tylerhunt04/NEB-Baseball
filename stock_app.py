import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Professional Color Scheme (Theme 1: Professional Blue)
COLORS = {
    'primary': '#1E88E5',      # Professional Blue
    'success': '#43A047',      # Green for gains
    'danger': '#E53935',       # Red for losses
    'warning': '#FB8C00',      # Orange for warnings
    'info': '#00ACC1',         # Cyan for info
    'neutral': '#757575',      # Gray for neutral
    'background': '#0E1117',   # Dark background
    'surface': '#1E1E1E',      # Card background
    'text': '#FAFAFA',         # White text
}

# Page config
st.set_page_config(
    page_title="Stock Market Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown(f"""
<style>
    /* Main container styling */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    
    /* Professional metric cards */
    div[data-testid="metric-container"] {{
        background: {COLORS['surface']};
        border: 1px solid #333;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    
    div[data-testid="metric-container"]:hover {{
        border-color: {COLORS['primary']};
        box-shadow: 0 4px 8px rgba(30,136,229,0.2);
        transition: all 0.3s ease;
    }}
    
    /* Performance badges */
    .performance-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.25rem;
    }}
    
    .badge-excellent {{
        background: {COLORS['success']};
        color: white;
    }}
    
    .badge-good {{
        background: #66BB6A;
        color: white;
    }}
    
    .badge-neutral {{
        background: {COLORS['neutral']};
        color: white;
    }}
    
    .badge-poor {{
        background: {COLORS['danger']};
        color: white;
    }}
    
    /* Section headers */
    .section-header {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLORS['primary']};
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {COLORS['primary']};
    }}
    
    /* Info cards */
    .info-card {{
        background: {COLORS['surface']};
        border-left: 4px solid {COLORS['primary']};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Success/Warning cards */
    .success-card {{
        background: rgba(67, 160, 71, 0.1);
        border-left: 4px solid {COLORS['success']};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    .warning-card {{
        background: rgba(229, 57, 53, 0.1);
        border-left: 4px solid {COLORS['danger']};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {COLORS['surface']};
        padding: 0.5rem;
        border-radius: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        border-radius: 6px;
        padding: 0 1.5rem;
        background: transparent;
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']};
    }}
    
    /* Button styling */
    .stButton > button {{
        background: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background: #1565C0;
        box-shadow: 0 4px 12px rgba(30,136,229,0.3);
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: {COLORS['surface']};
    }}
    
    /* Data table styling */
    .dataframe {{
        border: 1px solid #333 !important;
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# Header with professional styling
st.markdown(f"""
<div style="background: linear-gradient(135deg, {COLORS['primary']} 0%, #1565C0 100%); 
            padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
        Stock Market Analyzer
    </h1>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">
        Professional Real-Time Stock Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown('<p class="section-header">Configuration</p>', unsafe_allow_html=True)
    
    # Ticker input with cleaner styling
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter a valid stock symbol (e.g., AAPL, MSFT)").upper()
    
    # Period selection
    st.markdown("**Time Period**")
    period = st.radio(
        "Select analysis range:",
        ["1M", "3M", "6M", "1Y", "2Y", "5Y", "MAX"],
        index=3,
        label_visibility="collapsed"
    )
    
    # Map period to yfinance format
    period_map = {
        "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y",
        "2Y": "2y", "5Y": "5y", "MAX": "max"
    }
    
    st.markdown("---")
    analyze_button = st.button("Analyze Stock", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Quick Links**")
    st.markdown("• [Market News](https://finance.yahoo.com)")
    st.markdown("• [SEC Filings](https://www.sec.gov)")
    st.markdown("• [Investor Relations](#)")

# Helper function for performance badges
def get_performance_badge(value, thresholds, metric_type="return"):
    """Generate professional performance badge"""
    if metric_type == "return":
        if value > thresholds[0]:
            return f'<span class="performance-badge badge-excellent">Excellent ({value:+.2f}%)</span>'
        elif value > thresholds[1]:
            return f'<span class="performance-badge badge-good">Good ({value:+.2f}%)</span>'
        elif value > thresholds[2]:
            return f'<span class="performance-badge badge-neutral">Neutral ({value:+.2f}%)</span>'
        else:
            return f'<span class="performance-badge badge-poor">Poor ({value:+.2f}%)</span>'
    elif metric_type == "volatility":
        if value < thresholds[0]:
            return f'<span class="performance-badge badge-excellent">Low Risk ({value:.2f}%)</span>'
        elif value < thresholds[1]:
            return f'<span class="performance-badge badge-good">Moderate ({value:.2f}%)</span>'
        else:
            return f'<span class="performance-badge badge-poor">High Risk ({value:.2f}%)</span>'

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
            
            df = yf.download(ticker, period=period, progress=False, threads=False)
            
            if not df.empty:
                # Fix multi-level columns issue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                df = df.reset_index()
                
                # Ensure proper column naming
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

# Main analysis section
if analyze_button or ticker:
    try:
        with st.spinner(f"Loading {period} data for {ticker}..."):
            df, error = download_stock_data(ticker, period_map[period])
        
        if error:
            st.markdown(f'<div class="warning-card"><strong>Error:</strong> {error}</div>', unsafe_allow_html=True)
            
            if "Rate limited" in error:
                st.info("**Solutions:** Wait 2-3 minutes • Clear cache (press 'C') • Try different ticker")
            else:
                st.info("**Try:** AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA")
            
        elif df is None or df.empty:
            st.markdown(f'<div class="warning-card">No data found for \'{ticker}\'</div>', unsafe_allow_html=True)
            st.info("**Suggested tickers:** AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA")
            
        else:
            # Success message
            st.markdown(f'<div class="success-card">Successfully loaded {len(df)} days of data for {ticker}</div>', 
                       unsafe_allow_html=True)
            
            # Get company information
            company_name = ticker
            try:
                stock = yf.Ticker(ticker)
                info = stock.info if hasattr(stock, 'info') else {}
                company_name = info.get('longName', ticker)
            except:
                pass
            
            # Company header
            st.markdown(f'<h2 style="color: {COLORS["primary"]}; margin: 2rem 0 1rem 0;">{company_name} ({ticker})</h2>', 
                       unsafe_allow_html=True)
            
            # Calculate key metrics
            latest_close = float(df['Close'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else latest_close
            price_change = latest_close - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            period_high = float(df['High'].max())
            period_low = float(df['Low'].min())
            avg_volume = float(df['Volume'].mean())
            
            # Professional metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"${latest_close:.2f}", 
                    f"{price_change:.2f} ({price_change_pct:+.2f}%)",
                    delta_color="normal"
                )
            
            with col2:
                period_range_pct = ((period_high - period_low) / period_low) * 100
                st.metric("Period High", f"${period_high:.2f}", f"{period_range_pct:.1f}% range")
            
            with col3:
                distance_from_low = ((latest_close - period_low) / period_low) * 100
                st.metric("Period Low", f"${period_low:.2f}", f"+{distance_from_low:.1f}% from low")
            
            with col4:
                volume_display = f"{avg_volume/1e6:.1f}M" if avg_volume >= 1e6 else f"{avg_volume/1e3:.1f}K"
                st.metric("Avg Volume", volume_display)
            
            # Performance summary
            first_close = float(df['Close'].iloc[0])
            last_close = float(df['Close'].iloc[-1])
            total_return = ((last_close - first_close) / first_close) * 100
            
            daily_returns = df['Close'].pct_change()
            volatility = float(daily_returns.std()) * 100
            
            st.markdown('<p class="section-header">Performance Summary</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                badge = get_performance_badge(total_return, [15, 5, -5], "return")
                st.markdown(f"**Period Return:** {badge}", unsafe_allow_html=True)
            
            with col2:
                vol_badge = get_performance_badge(volatility, [2, 5, 10], "volatility")
                st.markdown(f"**Volatility:** {vol_badge}", unsafe_allow_html=True)
            
            with col3:
                win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
                st.markdown(f"**Win Rate:** <span style='color: {COLORS["primary"]}; font-weight: 600;'>{win_rate:.1f}%</span> "
                          f"({(daily_returns > 0).sum()}/{len(daily_returns)} days)", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Create tabs with cleaner names
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "Price Chart", 
                "Volume Analysis", 
                "Moving Averages",
                "Technical Indicators",
                "Fundamentals",
                "Performance",
                "Data Export"
            ])
            
            # Tab 1: Price Chart
            with tab1:
                st.markdown('<p class="section-header">Price Chart</p>', unsafe_allow_html=True)
                
                chart_type = st.radio("Chart Type:", ["Candlestick", "Line"], horizontal=True)
                
                chart_df = df.copy()
                fig = go.Figure()
                
                if chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=chart_df['Date'],
                        open=chart_df['Open'],
                        high=chart_df['High'],
                        low=chart_df['Low'],
                        close=chart_df['Close'],
                        name='Price',
                        increasing_line_color=COLORS['success'],
                        decreasing_line_color=COLORS['danger']
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=chart_df['Date'],
                        y=chart_df['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color=COLORS['primary'], width=2.5),
                        fill='tozeroy',
                        fillcolor=f"rgba(30, 136, 229, 0.1)"
                    ))
                
                fig.update_layout(
                    title=f"{ticker} - {period} Performance",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    height=500,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['surface'],
                    font=dict(color=COLORS['text'], size=12),
                    xaxis=dict(gridcolor='#2a2a2a', showgrid=True, rangeslider=dict(visible=False)),
                    yaxis=dict(gridcolor='#2a2a2a', showgrid=True),
                    margin=dict(l=60, r=60, t=60, b=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Quick stats
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("High", f"${period_high:.2f}")
                col2.metric("Low", f"${period_low:.2f}")
                col3.metric("Change", f"{total_return:+.2f}%")
                col4.metric("Days", len(df))
            
            # Tab 2: Volume Analysis
            with tab2:
                st.markdown('<p class="section-header">Volume Analysis</p>', unsafe_allow_html=True)
                
                vol_df = df.copy()
                
                if 'Close' in vol_df.columns and 'Open' in vol_df.columns:
                    vol_df['color'] = vol_df.apply(
                        lambda row: COLORS['success'] if row['Close'] >= row['Open'] else COLORS['danger'],
                        axis=1
                    )
                    colors = vol_df['color'].tolist()
                else:
                    colors = COLORS['info']
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=vol_df['Date'],
                    y=vol_df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.8
                ))
                
                # Add average volume line
                avg_vol = vol_df['Volume'].mean()
                fig.add_hline(
                    y=avg_vol,
                    line_dash="dash",
                    line_color=COLORS['neutral'],
                    annotation_text=f"Avg: {avg_vol/1e6:.1f}M",
                    annotation_position="right"
                )
                
                fig.update_layout(
                    title=f"{ticker} Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    hovermode='x unified',
                    height=500,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['surface'],
                    font=dict(color=COLORS['text'], size=12),
                    xaxis=dict(gridcolor='#2a2a2a', showgrid=True),
                    yaxis=dict(gridcolor='#2a2a2a', showgrid=True),
                    margin=dict(l=60, r=60, t=60, b=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume insights
                col1, col2, col3 = st.columns(3)
                max_vol = vol_df['Volume'].max()
                min_vol = vol_df['Volume'].min()
                recent_avg = vol_df.tail(10)['Volume'].mean()
                
                col1.metric("Max Volume", f"{max_vol/1e6:.1f}M")
                col2.metric("Min Volume", f"{min_vol/1e6:.1f}M")
                col3.metric("Recent Avg (10D)", f"{recent_avg/1e6:.1f}M")
            
            # Tab 3: Moving Averages
            with tab3:
                st.markdown('<p class="section-header">Moving Average Analysis</p>', unsafe_allow_html=True)
                
                # Add multiple MA options
                col1, col2 = st.columns([3, 1])
                with col1:
                    ma_days = st.slider("Moving Average Period", 5, 200, 20, 5)
                with col2:
                    show_multiple = st.checkbox("Show 20/50/200 MA", value=False)
                
                df_ma = df.copy()
                
                fig = go.Figure()
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=df_ma['Date'],
                    y=df_ma['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color=COLORS['text'], width=2)
                ))
                
                if show_multiple:
                    # Show standard MAs
                    for ma_period, color, name in [(20, COLORS['success'], '20-Day'), 
                                                     (50, COLORS['warning'], '50-Day'),
                                                     (200, COLORS['danger'], '200-Day')]:
                        if len(df_ma) >= ma_period:
                            df_ma[f'MA{ma_period}'] = df_ma['Close'].rolling(window=ma_period).mean()
                            fig.add_trace(go.Scatter(
                                x=df_ma['Date'],
                                y=df_ma[f'MA{ma_period}'],
                                mode='lines',
                                name=f'{name} MA',
                                line=dict(color=color, width=2, dash='dash')
                            ))
                else:
                    # Single MA
                    df_ma['MA'] = df_ma['Close'].rolling(window=ma_days).mean()
                    fig.add_trace(go.Scatter(
                        x=df_ma['Date'],
                        y=df_ma['MA'],
                        mode='lines',
                        name=f'{ma_days}-Day MA',
                        line=dict(color=COLORS['primary'], width=2.5, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"{ticker} with Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    hovermode='x unified',
                    height=500,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['surface'],
                    font=dict(color=COLORS['text'], size=12),
                    xaxis=dict(gridcolor='#2a2a2a', showgrid=True),
                    yaxis=dict(gridcolor='#2a2a2a', showgrid=True),
                    legend=dict(bgcolor=COLORS['surface'], bordercolor='#333'),
                    margin=dict(l=60, r=60, t=60, b=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # MA signals
                if not show_multiple:
                    current_price = float(df_ma['Close'].iloc[-1])
                    current_ma = float(df_ma['MA'].iloc[-1])
                    
                    if pd.notna(current_ma):
                        diff_pct = ((current_price - current_ma) / current_ma) * 100
                        
                        if diff_pct > 2:
                            st.markdown(f'<div class="success-card"><strong>Signal:</strong> Price {diff_pct:.2f}% above MA - Bullish trend</div>', 
                                      unsafe_allow_html=True)
                        elif diff_pct < -2:
                            st.markdown(f'<div class="warning-card"><strong>Signal:</strong> Price {diff_pct:.2f}% below MA - Bearish trend</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="info-card"><strong>Signal:</strong> Price near MA ({diff_pct:+.2f}%) - Consolidating</div>', 
                                      unsafe_allow_html=True)
            
            # Tab 4: Technical Indicators
            with tab4:
                st.markdown('<p class="section-header">Technical Indicators</p>', unsafe_allow_html=True)
                
                # Calculate indicators
                def calculate_rsi(data, period=14):
                    delta = data.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    return 100 - (100 / (1 + rs))
                
                df_tech = df.copy()
                df_tech['RSI'] = calculate_rsi(df_tech['Close'])
                
                # MACD
                exp1 = df_tech['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df_tech['Close'].ewm(span=26, adjust=False).mean()
                df_tech['MACD'] = exp1 - exp2
                df_tech['Signal'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
                df_tech['Histogram'] = df_tech['MACD'] - df_tech['Signal']
                
                # Bollinger Bands
                df_tech['BB_Middle'] = df_tech['Close'].rolling(window=20).mean()
                df_tech['BB_Std'] = df_tech['Close'].rolling(window=20).std()
                df_tech['BB_Upper'] = df_tech['BB_Middle'] + (df_tech['BB_Std'] * 2)
                df_tech['BB_Lower'] = df_tech['BB_Middle'] - (df_tech['BB_Std'] * 2)
                
                # RSI Chart
                st.markdown("**RSI (Relative Strength Index)**")
                fig_rsi = go.Figure()
                
                fig_rsi.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=COLORS['primary'], width=2.5),
                    fill='tozeroy',
                    fillcolor=f"rgba(30, 136, 229, 0.1)"
                ))
                
                # Reference lines
                fig_rsi.add_hline(y=70, line_dash="dash", line_color=COLORS['danger'], 
                                  annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color=COLORS['success'], 
                                  annotation_text="Oversold")
                fig_rsi.add_hline(y=50, line_dash="dot", line_color=COLORS['neutral'], 
                                  annotation_text="Neutral")
                
                fig_rsi.add_hrect(y0=70, y1=100, fillcolor=COLORS['danger'], opacity=0.1, line_width=0)
                fig_rsi.add_hrect(y0=0, y1=30, fillcolor=COLORS['success'], opacity=0.1, line_width=0)
                
                fig_rsi.update_layout(
                    height=300,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['surface'],
                    font=dict(color=COLORS['text']),
                    xaxis=dict(gridcolor='#2a2a2a'),
                    yaxis=dict(gridcolor='#2a2a2a', range=[0, 100]),
                    margin=dict(l=60, r=60, t=40, b=40)
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # RSI interpretation
                current_rsi = df_tech['RSI'].iloc[-1]
                if current_rsi > 70:
                    st.markdown(f'<div class="warning-card">RSI: {current_rsi:.1f} - <strong>Overbought</strong> (Potential sell signal)</div>', 
                              unsafe_allow_html=True)
                elif current_rsi < 30:
                    st.markdown(f'<div class="success-card">RSI: {current_rsi:.1f} - <strong>Oversold</strong> (Potential buy signal)</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="info-card">RSI: {current_rsi:.1f} - <strong>Neutral</strong></div>', 
                              unsafe_allow_html=True)
                
                st.markdown("---")
                
                # MACD Chart
                st.markdown("**MACD**")
                fig_macd = go.Figure()
                
                fig_macd.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color=COLORS['primary'], width=2)
                ))
                
                fig_macd.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color=COLORS['warning'], width=2)
                ))
                
                colors_hist = [COLORS['success'] if val >= 0 else COLORS['danger'] 
                              for val in df_tech['Histogram']]
                fig_macd.add_trace(go.Bar(
                    x=df_tech['Date'],
                    y=df_tech['Histogram'],
                    name='Histogram',
                    marker_color=colors_hist,
                    opacity=0.4
                ))
                
                fig_macd.update_layout(
                    height=300,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['surface'],
                    font=dict(color=COLORS['text']),
                    xaxis=dict(gridcolor='#2a2a2a'),
                    yaxis=dict(gridcolor='#2a2a2a'),
                    margin=dict(l=60, r=60, t=40, b=40)
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
                
                # MACD interpretation
                current_macd = df_tech['MACD'].iloc[-1]
                current_signal = df_tech['Signal'].iloc[-1]
                
                if current_macd > current_signal:
                    crossover_diff = current_macd - current_signal
                    st.markdown(f'<div class="success-card">MACD above Signal (+{crossover_diff:.3f}) - <strong>Bullish momentum</strong></div>', 
                              unsafe_allow_html=True)
                else:
                    crossover_diff = current_signal - current_macd
                    st.markdown(f'<div class="warning-card">MACD below Signal (-{crossover_diff:.3f}) - <strong>Bearish momentum</strong></div>', 
                              unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Bollinger Bands
                st.markdown("**Bollinger Bands**")
                fig_bb = go.Figure()
                
                fig_bb.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color=COLORS['text'], width=2.5)
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['BB_Upper'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color=COLORS['danger'], width=1, dash='dash')
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['BB_Middle'],
                    mode='lines',
                    name='Middle (20 MA)',
                    line=dict(color=COLORS['neutral'], width=1)
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=df_tech['Date'],
                    y=df_tech['BB_Lower'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color=COLORS['success'], width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(100, 100, 100, 0.1)'
                ))
                
                fig_bb.update_layout(
                    height=400,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['surface'],
                    font=dict(color=COLORS['text']),
                    xaxis=dict(gridcolor='#2a2a2a'),
                    yaxis=dict(gridcolor='#2a2a2a'),
                    legend=dict(bgcolor=COLORS['surface'], bordercolor='#333'),
                    margin=dict(l=60, r=60, t=40, b=40)
                )
                
                st.plotly_chart(fig_bb, use_container_width=True)
                
                # Bollinger Bands interpretation
                latest_close = float(df_tech['Close'].iloc[-1])
                latest_upper = float(df_tech['BB_Upper'].iloc[-1])
                latest_lower = float(df_tech['BB_Lower'].iloc[-1])
                latest_middle = float(df_tech['BB_Middle'].iloc[-1])
                
                bb_position = ((latest_close - latest_lower) / (latest_upper - latest_lower)) * 100
                
                if latest_close > latest_upper:
                    st.markdown(f'<div class="warning-card">Price above upper band - <strong>Overbought</strong> (Position: {bb_position:.0f}%)</div>', 
                              unsafe_allow_html=True)
                elif latest_close < latest_lower:
                    st.markdown(f'<div class="success-card">Price below lower band - <strong>Oversold</strong> (Position: {bb_position:.0f}%)</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="info-card">Price within bands - <strong>Normal range</strong> (Position: {bb_position:.0f}%)</div>', 
                              unsafe_allow_html=True)
            
            # Tab 5: Fundamentals
            with tab5:
                st.markdown('<p class="section-header">Company Fundamentals</p>', unsafe_allow_html=True)
                
                try:
                    stock_obj = yf.Ticker(ticker)
                    info_data = stock_obj.info if hasattr(stock_obj, 'info') else {}
                    
                    # Company Overview
                    st.markdown("**Company Overview**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Company:** {info_data.get('longName', 'N/A')}")
                        st.markdown(f"**Sector:** {info_data.get('sector', 'N/A')}")
                        st.markdown(f"**Industry:** {info_data.get('industry', 'N/A')}")
                        st.markdown(f"**Country:** {info_data.get('country', 'N/A')}")
                    
                    with col2:
                        employees = info_data.get('fullTimeEmployees', 'N/A')
                        if isinstance(employees, int):
                            st.markdown(f"**Employees:** {employees:,}")
                        else:
                            st.markdown(f"**Employees:** N/A")
                        
                        st.markdown(f"**Exchange:** {info_data.get('exchange', 'N/A')}")
                        st.markdown(f"**Currency:** {info_data.get('currency', 'N/A')}")
                    
                    summary = info_data.get('longBusinessSummary', '')
                    if summary:
                        with st.expander("Business Summary"):
                            st.write(summary)
                    
                    st.markdown("---")
                    
                    # Key Metrics
                    st.markdown("**Key Financial Metrics**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        market_cap = info_data.get('marketCap')
                        if market_cap:
                            mc_display = f"${market_cap/1e12:.2f}T" if market_cap >= 1e12 else f"${market_cap/1e9:.2f}B"
                        else:
                            mc_display = "N/A"
                        st.metric("Market Cap", mc_display)
                        
                        pe = info_data.get('trailingPE')
                        st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
                    
                    with col2:
                        eps = info_data.get('trailingEps')
                        st.metric("EPS", f"${eps:.2f}" if eps else "N/A")
                        
                        pb = info_data.get('priceToBook')
                        st.metric("P/B Ratio", f"{pb:.2f}" if pb else "N/A")
                    
                    with col3:
                        profit_margin = info_data.get('profitMargins')
                        if profit_margin:
                            st.metric("Profit Margin", f"{profit_margin*100:.2f}%")
                        else:
                            st.metric("Profit Margin", "N/A")
                        
                        roe = info_data.get('returnOnEquity')
                        if roe:
                            st.metric("ROE", f"{roe*100:.2f}%")
                        else:
                            st.metric("ROE", "N/A")
                    
                    with col4:
                        div_yield = info_data.get('dividendYield')
                        if div_yield:
                            st.metric("Dividend Yield", f"{div_yield*100:.2f}%")
                        else:
                            st.metric("Dividend Yield", "N/A")
                        
                        beta = info_data.get('beta')
                        st.metric("Beta", f"{beta:.2f}" if beta else "N/A")
                    
                    st.markdown("---")
                    
                    # Analyst Targets
                    st.markdown("**Analyst Price Targets**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        target_high = info_data.get('targetHighPrice')
                        st.metric("High", f"${target_high:.2f}" if target_high else "N/A")
                    
                    with col2:
                        target_mean = info_data.get('targetMeanPrice')
                        st.metric("Mean", f"${target_mean:.2f}" if target_mean else "N/A")
                    
                    with col3:
                        target_low = info_data.get('targetLowPrice')
                        st.metric("Low", f"${target_low:.2f}" if target_low else "N/A")
                    
                    with col4:
                        if target_mean:
                            upside = ((target_mean - latest_close) / latest_close) * 100
                            st.metric("Upside", f"{upside:+.1f}%")
                        else:
                            st.metric("Upside", "N/A")
                    
                    recommendation = info_data.get('recommendationKey', 'N/A')
                    if recommendation != 'N/A':
                        rec_display = recommendation.upper().replace('_', ' ')
                        if recommendation in ['buy', 'strong_buy']:
                            st.markdown(f'<div class="success-card"><strong>Analyst Recommendation:</strong> {rec_display}</div>', 
                                      unsafe_allow_html=True)
                        elif recommendation == 'hold':
                            st.markdown(f'<div class="info-card"><strong>Analyst Recommendation:</strong> {rec_display}</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="warning-card"><strong>Analyst Recommendation:</strong> {rec_display}</div>', 
                                      unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error("Unable to load fundamental data")
                    st.info(f"Error: {str(e)}")
            
            # Tab 6: Performance Analysis
            with tab6:
                st.markdown('<p class="section-header">Performance Analysis</p>', unsafe_allow_html=True)
                
                # Return metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Return", f"{total_return:+.2f}%")
                
                with col2:
                    annualized_return = total_return * (252 / len(df))
                    st.metric("Annualized Return", f"{annualized_return:+.2f}%")
                
                with col3:
                    sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                st.markdown("---")
                
                # Returns distribution
                st.markdown("**Daily Returns Distribution**")
                
                daily_returns_pct = daily_returns.dropna() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=daily_returns_pct,
                    nbinsx=50,
                    name='Daily Returns',
                    marker_color=COLORS['primary'],
                    opacity=0.8
                ))
                
                fig.update_layout(
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    height=400,
                    plot_bgcolor=COLORS['background'],
                    paper_bgcolor=COLORS['surface'],
                    font=dict(color=COLORS['text']),
                    xaxis=dict(gridcolor='#2a2a2a'),
                    yaxis=dict(gridcolor='#2a2a2a'),
                    margin=dict(l=60, r=60, t=40, b=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance statistics
                st.markdown("**Statistics**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Daily Return", f"{daily_returns.mean()*100:.3f}%")
                    st.metric("Std Deviation", f"{daily_returns.std()*100:.3f}%")
                
                with col2:
                    st.metric("Best Day", f"{daily_returns.max()*100:+.2f}%")
                    st.metric("Worst Day", f"{daily_returns.min()*100:+.2f}%")
                
                with col3:
                    positive_days = (daily_returns > 0).sum()
                    st.metric("Positive Days", f"{positive_days} ({positive_days/len(daily_returns)*100:.1f}%)")
                    
                    win_avg = daily_returns[daily_returns > 0].mean() * 100
                    st.metric("Avg Win", f"+{win_avg:.2f}%")
                
                with col4:
                    negative_days = (daily_returns < 0).sum()
                    st.metric("Negative Days", f"{negative_days} ({negative_days/len(daily_returns)*100:.1f}%)")
                    
                    loss_avg = daily_returns[daily_returns < 0].mean() * 100
                    st.metric("Avg Loss", f"{loss_avg:.2f}%")
                
                st.markdown("---")
                
                # Recent performance table
                st.markdown("**Recent Trading Days (Last 10)**")
                recent_data = df.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                recent_data['Change %'] = recent_data['Close'].pct_change() * 100
                
                st.dataframe(
                    recent_data.style.format({
                        'Open': '${:.2f}',
                        'High': '${:.2f}',
                        'Low': '${:.2f}',
                        'Close': '${:.2f}',
                        'Volume': '{:,.0f}',
                        'Change %': '{:+.2f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Tab 7: Data Export
            with tab7:
                st.markdown('<p class="section-header">Data Export</p>', unsafe_allow_html=True)
                
                st.markdown("**Download Historical Data**")
                
                # Prepare export data
                export_df = df.copy()
                
                # Add technical indicators to export
                export_df['Daily_Return_%'] = daily_returns * 100
                export_df['RSI'] = calculate_rsi(export_df['Close'])
                
                # Format for display
                display_export = export_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return_%', 'RSI']].copy()
                
                st.dataframe(
                    display_export.style.format({
                        'Open': '${:.2f}',
                        'High': '${:.2f}',
                        'Low': '${:.2f}',
                        'Close': '${:.2f}',
                        'Volume': '{:,.0f}',
                        'Daily_Return_%': '{:+.2f}%',
                        'RSI': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download Full Dataset (CSV)",
                        data=csv,
                        file_name=f"{ticker}_{period}_full_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    summary_stats = {
                        'Metric': ['Period', 'Total Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 
                                  'Win Rate', 'Avg Daily Return', 'Data Points'],
                        'Value': [
                            period,
                            f"{total_return:.2f}%",
                            f"{volatility:.2f}%",
                            f"{sharpe:.2f}",
                            f"{(df['Close'] / df['Close'].cummax() - 1).min() * 100:.2f}%",
                            f"{(daily_returns > 0).sum() / len(daily_returns) * 100:.1f}%",
                            f"{daily_returns.mean() * 100:.3f}%",
                            len(df)
                        ]
                    }
                    summary_df = pd.DataFrame(summary_stats)
                    summary_csv = summary_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Summary Report (CSV)",
                        data=summary_csv,
                        file_name=f"{ticker}_{period}_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                st.markdown("---")
                
                # Data info
                st.markdown("**Dataset Information**")
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Rows", len(export_df))
                col2.metric("Columns", len(export_df.columns))
                col3.metric("Period", f"{export_df['Date'].min().date()} to {export_df['Date'].max().date()}")
                
    except Exception as e:
        st.markdown(f'<div class="warning-card"><strong>Error:</strong> {str(e)}</div>', unsafe_allow_html=True)
        st.info("Try: Wait a few minutes • Clear cache (press 'C') • Different ticker • Shorter period")

else:
    # Welcome screen
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### Getting Started")
    st.markdown("""
    1. Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    2. Select your preferred time period
    3. Click **Analyze Stock** to generate comprehensive analysis
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Popular tickers
    st.markdown('<p class="section-header">Popular Stocks</p>', unsafe_allow_html=True)
    
    popular = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "JPM": "JPMorgan Chase"
    }
    
    cols = st.columns(4)
    for idx, (symbol, name) in enumerate(popular.items()):
        with cols[idx % 4]:
            st.markdown(f"**{symbol}**")
            st.caption(name)

# Footer
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; color: {COLORS["neutral"]}; padding: 1rem;">'
    'Data provided by Yahoo Finance • Updated every 10 minutes • For educational purposes only'
    '</div>',
    unsafe_allow_html=True
)
