import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os

# Page config
st.set_page_config(
    page_title="",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    
    /* Main styling - Light Mode */
    .main {
        background: #fafafa;
        color: #1a1a1a;
    }
    
    /* Headers */
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        color: #1a1a1a;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    h2 {
        font-family: 'Space Grotesk', sans-serif;
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    h3 {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    p, div, span, label {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
    }
    
    /* Section dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
        margin: 3rem 0;
    }
    
    /* Cards with modern shadow */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
        transform: translateY(-2px);
    }
    
    .budget-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #FFD700;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border: 1px solid #3d3d3d;
    }
    
    .spending-card {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.25);
        border: 1px solid #bd2130;
    }
    
    .income-card {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.25);
        border: 1px solid #1e7e34;
    }
    
    .remaining-card-green {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.25);
        border: 1px solid #1e7e34;
    }
    
    .remaining-card-red {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.25);
        border: 1px solid #bd2130;
    }
    
    .poetry-card {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin: 2rem 0;
        border-left: 4px solid #FFD700;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    }
    
    .poetry-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-style: italic;
        color: #2d2d2d;
        line-height: 1.8;
        text-align: center;
        font-weight: 400;
    }
    
    .poetry-author {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #6b6b6b;
        text-align: right;
        margin-top: 1.5rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%);
        border-right: 1px solid #2d2d2d;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFD700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Force selectbox text to be black */
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] input {
        color: #1a1a1a !important;
    }
    
    /* Button styling - FORCE ALL BUTTON TEXT TO BLACK */
    button {
        color: #000000 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    button * {
        color: #000000 !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFC700 100%);
        color: #000000 !important;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(255, 215, 0, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 215, 0, 0.4);
        color: #000000 !important;
    }
    
    .stButton>button p {
        color: #000000 !important;
    }
    
    .stButton>button span {
        color: #000000 !important;
    }
    
    .stButton>button div {
        color: #000000 !important;
    }
    
    button[kind="primary"] {
        color: #000000 !important;
    }
    
    button[kind="primary"] * {
        color: #000000 !important;
    }
    
    button[type="submit"] {
        color: #000000 !important;
    }
    
    button[type="submit"] * {
        color: #000000 !important;
    }
    
    .stFormSubmitButton button {
        color: #000000 !important;
    }
    
    .stFormSubmitButton button * {
        color: #000000 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #FFD700 0%, #FFC700 100%);
    }
    
    /* Input fields */
    input, textarea, select {
        background: white !important;
        color: #1a1a1a !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.1) !important;
    }
    
    /* Dropdown options */
    option {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Streamlit selectbox */
    [data-baseweb="select"] {
        background: white !important;
        border-radius: 10px !important;
    }
    
    [data-baseweb="select"] > div {
        background: white !important;
        color: #1a1a1a !important;
        border-color: #e0e0e0 !important;
        border-radius: 10px !important;
    }
    
    [data-baseweb="select"] span {
        color: #1a1a1a !important;
    }
    
    [data-baseweb="popover"] {
        background: #ffffff !important;
    }
    
    [role="option"] {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    [role="option"]:hover {
        background: #f5f5f5 !important;
        color: #1a1a1a !important;
    }
    
    /* Dataframe/table styling */
    [data-testid="stDataFrame"] {
        background: white;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    /* Make sure text is readable */
    .stMarkdown, .stText {
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left-width: 4px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Ensure Streamlit's default theme doesn't override */
    .stApp {
        background: #fafafa;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# File paths - store in a persistent location
import os
from pathlib import Path

# Create data directory in user's home folder
DATA_DIR = Path.home() / ".solana_finance_tracker"
DATA_DIR.mkdir(exist_ok=True)

TRANSACTIONS_FILE = str(DATA_DIR / "transactions.csv")
BUDGETS_FILE = str(DATA_DIR / "budgets.csv")

# Default categories
INCOME_CATEGORIES = ["Work"]

EXPENSE_CATEGORIES = [
    "Groceries", "Rent/Mortgage", "Utilities", "Transportation", 
    "Entertainment", "Dining Out", "Shopping", "Healthcare", 
    "Savings", "Other"
]

# Color wheel palette for categories
CATEGORY_COLORS = {
    "Groceries": "#FF6B6B",       # Red
    "Rent/Mortgage": "#4ECDC4",   # Teal
    "Utilities": "#45B7D1",       # Blue
    "Transportation": "#FFA07A",  # Light Salmon
    "Entertainment": "#98D8C8",   # Mint
    "Dining Out": "#F7DC6F",      # Yellow
    "Shopping": "#BB8FCE",        # Purple
    "Healthcare": "#85C1E2",      # Sky Blue
    "Savings": "#52C41A",         # Green
    "Other": "#95A5A6"            # Gray
}

def get_category_color(category):
    """Get consistent color for a category"""
    return CATEGORY_COLORS.get(category, "#95A5A6")

# Initialize data files
def initialize_files():
    if not os.path.exists(TRANSACTIONS_FILE):
        pd.DataFrame(columns=['date', 'amount', 'category', 'type', 'description']).to_csv(TRANSACTIONS_FILE, index=False)
    if not os.path.exists(BUDGETS_FILE):
        pd.DataFrame(columns=['category', 'budget']).to_csv(BUDGETS_FILE, index=False)

# Load data
def load_transactions():
    if os.path.exists(TRANSACTIONS_FILE):
        df = pd.read_csv(TRANSACTIONS_FILE)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Remove any rows with invalid dates
            df = df.dropna(subset=['date'])
        return df
    return pd.DataFrame(columns=['date', 'amount', 'category', 'type', 'description'])

def load_budgets():
    if os.path.exists(BUDGETS_FILE):
        return pd.read_csv(BUDGETS_FILE)
    return pd.DataFrame(columns=['category', 'budget'])

# Save data
def save_transaction(date_val, amount, category, trans_type, description):
    df = load_transactions()
    # Convert date to string in consistent format
    date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
    new_row = pd.DataFrame([{
        'date': date_str,
        'amount': amount,
        'category': category,
        'type': trans_type,
        'description': description
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    # Convert all dates to strings before saving
    if not df.empty:
        # Handle both datetime and string dates
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df.to_csv(TRANSACTIONS_FILE, index=False)

def delete_transaction(index):
    df = load_transactions()
    df = df.drop(index)
    df = df.reset_index(drop=True)
    # Convert all dates to strings before saving
    if not df.empty:
        # Handle both datetime and string dates
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df.to_csv(TRANSACTIONS_FILE, index=False)

def save_budgets(budgets_df):
    budgets_df.to_csv(BUDGETS_FILE, index=False)

# Initialize
initialize_files()

# Sidebar - Add Transaction
with st.sidebar:
    st.subheader("Income and Expenses")
    st.markdown('<div style="height: 2px; background: linear-gradient(90deg, #FFD700 0%, #FFC700 100%); margin: 0.5rem 0 1rem 0; border-radius: 2px;"></div>', unsafe_allow_html=True)
    
    # Add Income Section
    st.subheader("💵 Add Income")
    
    with st.form("income_form", clear_on_submit=True):
        income_date = st.date_input("Date", value=date.today(), key="income_date")
        income_amount = st.number_input("Amount ($)", min_value=0.01, step=5.00, key="income_amount")
        income_category = st.selectbox("Category", INCOME_CATEGORIES, key="income_category")
        income_description = st.text_input("Description (optional)", key="income_description")
        
        income_submitted = st.form_submit_button("Add Income")
        
        if income_submitted:
            save_transaction(income_date, income_amount, income_category, "Income", income_description)
            st.success("Income added!")
            st.rerun()
    
    st.markdown('<div style="height: 2px; background: linear-gradient(90deg, #FFD700 0%, #FFC700 100%); margin: 1rem 0; border-radius: 2px;"></div>', unsafe_allow_html=True)
    
    # Add Expense Section
    st.subheader("💳 Add Expense")
    
    with st.form("expense_form", clear_on_submit=True):
        expense_date = st.date_input("Date", value=date.today(), key="expense_date")
        expense_amount = st.number_input("Amount ($)", min_value=0.01, step=5.00, key="expense_amount")
        expense_category = st.selectbox("Category", EXPENSE_CATEGORIES, key="expense_category")
        expense_description = st.text_input("Description (optional)", key="expense_description")
        
        expense_submitted = st.form_submit_button("Add Expense")
        
        if expense_submitted:
            save_transaction(expense_date, expense_amount, expense_category, "Expense", expense_description)
            st.success("Expense added!")
            st.rerun()

# Load current data
transactions_df = load_transactions()
budgets_df = load_budgets()

# Main content
st.title("✨ Solana's Finances")

# Create tabs for different sections
tab1, tab2 = st.tabs(["📊 Finance Dashboard", "💰 Budget Creator"])

# ============================================================================
# FINANCE DASHBOARD TAB
# ============================================================================
with tab1:
    # Poetry rotation
    import random
    
    poems = [
        {
            "text": "A penny saved is not a penny earned,\nBut wisdom gained and lessons learned.\nFor wealth is more than coins in hand—\nIt's knowing how to wisely plan.",
            "author": "— On Financial Wisdom"
        },
        {
            "text": "Count your blessings, count your change,\nBoth require care to arrange.\nThe former fills the soul with light,\nThe latter keeps your future bright.",
            "author": "— On Gratitude & Growth"
        },
        {
            "text": "Small streams make mighty rivers flow,\nSmall savings help your wealth to grow.\nPatience is the golden key,\nTo financial serenity.",
            "author": "— On Patience & Prosperity"
        },
        {
            "text": "Not all that glitters must be bought,\nSome treasures can't be sold or sought.\nTrue wealth lies in mindful choice,\nNot silencing your inner voice.",
            "author": "— On Mindful Spending"
        },
        {
            "text": "Track the moments, track the spending,\nEvery journey has a beginning.\nWith each choice you write your story,\nOf financial health and glory.",
            "author": "— On Your Journey"
        }
    ]
    
    selected_poem = random.choice(poems)
    
    st.markdown(f"""
<div class="poetry-card">
    <div class="poetry-text">{selected_poem['text']}</div>
    <div class="poetry-author">{selected_poem['author']}</div>
</div>
""", unsafe_allow_html=True)

# Month Selector for Budget and Spending Overview
st.markdown("---")

# Get available months from transaction data
if not transactions_df.empty:
    transactions_df['year_month'] = transactions_df['date'].dt.to_period('M')
    available_months = sorted(transactions_df['year_month'].unique(), reverse=True)
    
    if len(available_months) > 0:
        # Create readable month options
        month_options = {str(month): month.strftime('%B %Y') for month in available_months}
        
        # Default to current month if available, otherwise most recent
        current_period = pd.Period(datetime.now(), freq='M')
        if current_period in available_months:
            default_month = str(current_period)
        else:
            default_month = str(available_months[0])
        
        # Month selector
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            selected_month_str = st.selectbox(
                "📅 Select Month to View",
                options=list(month_options.keys()),
                format_func=lambda x: month_options[x],
                index=list(month_options.keys()).index(default_month) if default_month in month_options else 0,
                key="month_selector"
            )
        
        selected_period = pd.Period(selected_month_str)
        selected_month = selected_period.month
        selected_year = selected_period.year
    else:
        selected_month = datetime.now().month
        selected_year = datetime.now().year
else:
    selected_month = datetime.now().month
    selected_year = datetime.now().year

# Calculate metrics for selected month
total_income = 0
total_expenses = 0
net_income = 0
current_month_df = pd.DataFrame()

if not transactions_df.empty:
    current_month_df = transactions_df[
        (transactions_df['date'].dt.month == selected_month) & 
        (transactions_df['date'].dt.year == selected_year)
    ]
    
    total_income = current_month_df[current_month_df['type'] == 'Income']['amount'].sum()
    total_expenses = current_month_df[current_month_df['type'] == 'Expense']['amount'].sum()
    net_income = total_income - total_expenses

# Top metrics - always show
selected_month_name = pd.Period(year=selected_year, month=selected_month, freq='M').strftime('%B %Y')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="income-card">
        <h3 style="margin:0; font-size: 1rem; opacity: 0.9; color: white;">Income - {selected_month_name}</h3>
        <h2 style="margin:0.5rem 0 0 0; font-size: 2rem; color: white;">${total_income:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="spending-card">
        <h3 style="margin:0; font-size: 1rem; opacity: 0.9; color: white;">Spent - {selected_month_name}</h3>
        <h2 style="margin:0.5rem 0 0 0; font-size: 2rem; color: white;">${total_expenses:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    remaining_class = "remaining-card-green" if net_income >= 20 else "remaining-card-red"
    st.markdown(f"""
    <div class="{remaining_class}">
        <h3 style="margin:0; font-size: 1rem; opacity: 0.9; color: white;">Remaining - {selected_month_name}</h3>
        <h2 style="margin:0.5rem 0 0 0; font-size: 2rem; color: white;">${net_income:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Spending Trends Over Time
st.subheader("Spending Trends")

if not transactions_df.empty:
    expense_data = transactions_df[transactions_df['type'] == 'Expense'].copy()
    
    if not expense_data.empty:
        # Create tabs for weekly and monthly views
        tab1, tab2 = st.tabs(["📅 Monthly", "📊 Weekly"])
        
        with tab1:
            # Monthly aggregation
            expense_data['month'] = expense_data['date'].dt.to_period('M')
            monthly_spending = expense_data.groupby('month')['amount'].sum().reset_index()
            monthly_spending['month'] = monthly_spending['month'].dt.to_timestamp()
            monthly_spending = monthly_spending.sort_values('month')
            
            # Only show last 12 months
            monthly_spending = monthly_spending.tail(12)
            
            if len(monthly_spending) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=monthly_spending['month'],
                    y=monthly_spending['amount'],
                    mode='lines+markers',
                    name='Monthly Spending',
                    line=dict(color='#FF6B6B', width=3),
                    marker=dict(size=8, color='#FF6B6B'),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.1)',
                    hovertemplate='<b>%{x|%B %Y}</b><br>Spent: $%{y:,.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(
                        title='Month',
                        gridcolor='rgba(200, 200, 200, 0.2)',
                        showgrid=True,
                        tickformat='%b %Y'
                    ),
                    yaxis=dict(
                        title='Amount ($)',
                        gridcolor='rgba(200, 200, 200, 0.2)',
                        showgrid=True
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#1a1a1a', family='Inter'),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                avg_monthly = monthly_spending['amount'].mean()
                max_month = monthly_spending.loc[monthly_spending['amount'].idxmax()]
                min_month = monthly_spending.loc[monthly_spending['amount'].idxmin()]
                
                with col1:
                    st.metric("Average Monthly", f"${avg_monthly:,.2f}")
                with col2:
                    st.metric("Highest Month", f"${max_month['amount']:,.2f}", 
                             delta=max_month['month'].strftime('%b %Y'))
                with col3:
                    st.metric("Lowest Month", f"${min_month['amount']:,.2f}",
                             delta=min_month['month'].strftime('%b %Y'))
            else:
                st.info("Not enough data yet to show monthly trends.")
        
        with tab2:
            # Weekly aggregation
            expense_data['week'] = expense_data['date'].dt.to_period('W')
            weekly_spending = expense_data.groupby('week')['amount'].sum().reset_index()
            weekly_spending['week'] = weekly_spending['week'].dt.to_timestamp()
            weekly_spending = weekly_spending.sort_values('week')
            
            # Only show last 12 weeks
            weekly_spending = weekly_spending.tail(12)
            
            if len(weekly_spending) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=weekly_spending['week'],
                    y=weekly_spending['amount'],
                    mode='lines+markers',
                    name='Weekly Spending',
                    line=dict(color='#4ECDC4', width=3),
                    marker=dict(size=8, color='#4ECDC4'),
                    fill='tozeroy',
                    fillcolor='rgba(78, 205, 196, 0.1)',
                    hovertemplate='<b>Week of %{x|%b %d}</b><br>Spent: $%{y:,.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(
                        title='Week',
                        gridcolor='rgba(200, 200, 200, 0.2)',
                        showgrid=True,
                        tickformat='%b %d'
                    ),
                    yaxis=dict(
                        title='Amount ($)',
                        gridcolor='rgba(200, 200, 200, 0.2)',
                        showgrid=True
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#1a1a1a', family='Inter'),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                avg_weekly = weekly_spending['amount'].mean()
                max_week = weekly_spending.loc[weekly_spending['amount'].idxmax()]
                min_week = weekly_spending.loc[weekly_spending['amount'].idxmin()]
                
                with col1:
                    st.metric("Average Weekly", f"${avg_weekly:,.2f}")
                with col2:
                    st.metric("Highest Week", f"${max_week['amount']:,.2f}",
                             delta=max_week['week'].strftime('%b %d'))
                with col3:
                    st.metric("Lowest Week", f"${min_week['amount']:,.2f}",
                             delta=min_week['week'].strftime('%b %d'))
            else:
                st.info("Not enough data yet to show weekly trends.")
    else:
        st.info("No expense data available yet.")
else:
    st.info("No transactions recorded yet.")

st.markdown("---")

# Spending by Category
st.subheader(f"Spending by Category - {selected_month_name}")

if not current_month_df.empty:
    expense_df = current_month_df[current_month_df['type'] == 'Expense']
    
    if not expense_df.empty:
        category_spending = expense_df.groupby('category')['amount'].sum().reset_index()
        category_spending = category_spending.sort_values('amount', ascending=False)
        
        # Get colors for each category in the chart
        colors = [get_category_color(cat) for cat in category_spending['category']]
        
        fig = px.pie(
            category_spending, 
            values='amount', 
            names='category',
            hole=0.4,
            color_discrete_sequence=colors
        )
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont=dict(color='white', size=12, family='Lato')
        )
        fig.update_layout(
            showlegend=False,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1a1a1a', family='Lato')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No expenses recorded this month yet.")
else:
    st.info("No expenses recorded this month yet.")

st.markdown("---")

# Budget Overview Section
st.subheader(f"Budget Overview - {selected_month_name}")

if not budgets_df.empty and not current_month_df.empty:
    expense_df = current_month_df[current_month_df['type'] == 'Expense']
    
    budget_data = []
    for _, budget_row in budgets_df.iterrows():
        category = budget_row['category']
        budget_amount = budget_row['budget']
        spent = expense_df[expense_df['category'] == category]['amount'].sum()
        
        budget_data.append({
            'Category': category,
            'Budget': budget_amount,
            'Spent': spent,
            'Remaining': max(0, budget_amount - spent),
            'Over': max(0, spent - budget_amount)
        })
    
    budget_df_chart = pd.DataFrame(budget_data)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add budget bars (lighter background)
    fig.add_trace(go.Bar(
        y=budget_df_chart['Category'],
        x=budget_df_chart['Budget'],
        name='Budget',
        orientation='h',
        marker=dict(color='rgba(200, 200, 200, 0.3)'),
        hovertemplate='<b>%{y}</b><br>Budget: $%{x:.2f}<extra></extra>'
    ))
    
    # Add spent bars using category colors
    colors = [get_category_color(cat) for cat in budget_df_chart['Category']]
    
    fig.add_trace(go.Bar(
        y=budget_df_chart['Category'],
        x=budget_df_chart['Spent'],
        name='Spent',
        orientation='h',
        marker=dict(color=colors),
        hovertemplate='<b>%{y}</b><br>Spent: $%{x:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='overlay',
        height=max(300, len(budget_df_chart) * 40),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title='Amount ($)', gridcolor='rgba(200, 200, 200, 0.2)'),
        yaxis=dict(title=''),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1a1a1a', family='Lato'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats below the chart
    col1, col2, col3 = st.columns(3)
    total_budget = budget_df_chart['Budget'].sum()
    total_spent = budget_df_chart['Spent'].sum()
    total_remaining = total_budget - total_spent
    
    with col1:
        st.metric("Total Budget", f"${total_budget:,.2f}")
    with col2:
        st.metric("Total Spent", f"${total_spent:,.2f}")
    with col3:
        st.metric("Remaining", f"${total_remaining:,.2f}", 
                 delta=f"{(total_remaining/total_budget*100):.1f}%" if total_budget > 0 else "0%")
    
elif not budgets_df.empty:
    st.info("No expenses this month yet to compare against budgets.")
else:
    st.info("Set budgets below to track your spending limits.")

# Recent transactions
st.markdown("---")
st.subheader("Recent Transactions")

# Column headers - always show
col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 2, 2, 3, 1])
with col1:
    st.markdown("**Date**")
with col2:
    st.markdown("**Type**")
with col3:
    st.markdown("**Category**")
with col4:
    st.markdown("**Amount**")
with col5:
    st.markdown("**Description**")
with col6:
    st.markdown("**Delete**")

st.markdown("---")

if not transactions_df.empty:
    recent_df = transactions_df.sort_values('date', ascending=False).head(10)
    
    if not recent_df.empty:
        # Transaction rows
        for idx, row in recent_df.iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 2, 2, 3, 1])
            
            with col1:
                st.write(row['date'].strftime('%Y-%m-%d'))
            with col2:
                st.write(row['type'])
            with col3:
                st.write(row['category'])
            with col4:
                if row['type'] == 'Income':
                    st.markdown(f"**:green[+${row['amount']:.2f}]**")
                else:
                    st.markdown(f"**:red[-${row['amount']:.2f}]**")
            with col5:
                st.write(row['description'] if row['description'] else "-")
            with col6:
                if st.button("🗑️", key=f"delete_{idx}", help="Delete transaction"):
                    delete_transaction(idx)
                    st.rerun()
            
            st.markdown("---")

# ============================================================================
# BUDGET CREATOR TAB
# ============================================================================
with tab2:
    st.markdown("### 💰 Create Your Monthly Budget")
    st.markdown("Build a realistic budget based on your income and fixed expenses")
    
    st.markdown("---")
    st.markdown("### Step 1: Enter Your Monthly Income")
    
    monthly_income = st.number_input(
        "Total Monthly Income (after taxes)",
        min_value=0.0,
        step=100.0,
        value=0.0,
        help="Enter your total take-home pay per month",
        key="budget_income"
    )
    
    st.markdown("---")
    st.markdown("### Step 2: Enter Your Fixed Monthly Costs")
    st.markdown("*These are expenses that stay the same every month and must be paid*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rent = st.number_input("Rent/Mortgage", min_value=0.0, step=50.0, value=0.0, key="budget_rent")
    with col2:
        parking = st.number_input("Monthly Parking", min_value=0.0, step=10.0, value=0.0, key="budget_parking")
    
    total_fixed = rent + parking
    
    if total_fixed > 0:
        st.info(f"💰 **Total Fixed Costs:** ${total_fixed:,.2f}/month")
    
    st.markdown("---")
    
    # Calculate remaining money
    remaining_for_budget = monthly_income - total_fixed
    
    if monthly_income > 0:
        st.markdown("### Step 3: Allocate Your Remaining Money")
        
        if remaining_for_budget > 0:
            st.success(f"✅ **Money Available for Budget:** ${remaining_for_budget:,.2f}")
            
            st.markdown("---")
            
            # Budget allocation method
            method = st.radio(
                "**Choose Your Budgeting Method:**",
                ["50/30/20 Rule", "Custom Percentages", "Manual Amounts"],
                help="50/30/20: 50% needs, 30% wants, 20% savings. Custom: Set your own percentages. Manual: Enter exact amounts.",
                key="budget_method"
            )
            
            budget_allocations = {}
            
            if method == "50/30/20 Rule":
                st.info("📊 **50/30/20 Rule**: 50% for needs (groceries, utilities, transportation), 30% for wants (entertainment, dining, shopping), 20% for savings")
                
                # Calculate allocations
                needs_amount = remaining_for_budget * 0.50
                wants_amount = remaining_for_budget * 0.30
                savings_amount = remaining_for_budget * 0.20
                
                # Distribute needs
                needs_categories = ["Groceries", "Utilities", "Transportation", "Healthcare"]
                needs_per_category = needs_amount / len(needs_categories)
                
                # Distribute wants
                wants_categories = ["Entertainment", "Dining Out", "Shopping"]
                wants_per_category = wants_amount / len(wants_categories)
                
                st.markdown("**💼 Needs (50%)**")
                col1, col2 = st.columns(2)
                for i, cat in enumerate(needs_categories):
                    with col1 if i % 2 == 0 else col2:
                        budget_allocations[cat] = st.number_input(
                            cat, 
                            value=float(needs_per_category),
                            step=10.0,
                            key=f"rule_needs_{cat}"
                        )
                
                st.markdown("**🎉 Wants (30%)**")
                col3, col4 = st.columns(2)
                for i, cat in enumerate(wants_categories):
                    with col3 if i % 2 == 0 else col4:
                        budget_allocations[cat] = st.number_input(
                            cat,
                            value=float(wants_per_category),
                            step=10.0,
                            key=f"rule_wants_{cat}"
                        )
                
                st.markdown("**💎 Savings & Other (20%)**")
                col5, col6 = st.columns(2)
                with col5:
                    budget_allocations["Savings"] = st.number_input(
                        "Savings",
                        value=float(savings_amount * 0.8),
                        step=10.0,
                        key="rule_savings"
                    )
                with col6:
                    budget_allocations["Other"] = st.number_input(
                        "Other/Emergency",
                        value=float(savings_amount * 0.2),
                        step=10.0,
                        key="rule_other"
                    )
            
            elif method == "Custom Percentages":
                st.info("💡 Set custom percentages for each category based on your priorities")
                
                percentages = {}
                
                col1, col2 = st.columns(2)
                
                with col1:
                    for cat in EXPENSE_CATEGORIES[:5]:
                        percentages[cat] = st.slider(
                            cat,
                            min_value=0,
                            max_value=50,
                            value=10,
                            step=1,
                            key=f"pct_{cat}",
                            format="%d%%"
                        )
                
                with col2:
                    for cat in EXPENSE_CATEGORIES[5:]:
                        percentages[cat] = st.slider(
                            cat,
                            min_value=0,
                            max_value=50,
                            value=10,
                            step=1,
                            key=f"pct_{cat}",
                            format="%d%%"
                        )
                
                total_percentage = sum(percentages.values())
                
                if total_percentage != 100:
                    st.warning(f"⚠️ Total percentage: {total_percentage}%. Adjust to equal 100%")
                else:
                    st.success("✅ Perfect! Your percentages add up to 100%")
                
                # Calculate amounts
                for cat, pct in percentages.items():
                    budget_allocations[cat] = (remaining_for_budget * pct / 100)
            
            else:  # Manual Amounts
                st.info("✏️ Enter the exact amount you want to budget for each category")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    for cat in EXPENSE_CATEGORIES[:5]:
                        budget_allocations[cat] = st.number_input(
                            cat,
                            min_value=0.0,
                            step=10.0,
                            value=0.0,
                            key=f"manual_{cat}"
                        )
                
                with col2:
                    for cat in EXPENSE_CATEGORIES[5:]:
                        budget_allocations[cat] = st.number_input(
                            cat,
                            min_value=0.0,
                            step=10.0,
                            value=0.0,
                            key=f"manual_{cat}"
                        )
            
            # Show summary
            st.markdown("---")
            st.markdown("### 📋 Budget Summary")
            
            total_budgeted = sum(budget_allocations.values())
            difference = remaining_for_budget - total_budgeted
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Budgeted", f"${total_budgeted:,.2f}")
            with col2:
                st.metric("Available", f"${remaining_for_budget:,.2f}")
            with col3:
                delta_color = "normal" if difference >= 0 else "inverse"
                st.metric("Difference", f"${difference:,.2f}", delta=f"${difference:,.2f}")
            
            # Show all allocations
            if total_budgeted > 0:
                budget_display = pd.DataFrame([
                    {
                        "Category": cat, 
                        "Monthly Budget": f"${amount:,.2f}",
                        "% of Available": f"{(amount/remaining_for_budget*100):.1f}%"
                    }
                    for cat, amount in budget_allocations.items()
                    if amount > 0
                ])
                
                st.dataframe(budget_display, use_container_width=True, hide_index=True)
            
            # Save button
            st.markdown("---")
            if st.button("💾 Save Budget", type="primary", use_container_width=True, key="save_budget_btn"):
                # Combine fixed costs and variable budget
                all_budgets = {}
                if rent > 0:
                    all_budgets["Rent/Mortgage"] = rent
                if parking > 0:
                    all_budgets["Parking"] = parking
                
                # Add variable budgets
                for cat, amount in budget_allocations.items():
                    if amount > 0:
                        all_budgets[cat] = amount
                
                # Save to CSV
                budget_save_df = pd.DataFrame([
                    {"category": cat, "budget": amount}
                    for cat, amount in all_budgets.items()
                ])
                
                save_budgets(budget_save_df)
                st.success("✅ Budget saved successfully! Scroll up to see your budget in action.")
                st.balloons()
                st.rerun()
        
        elif remaining_for_budget == 0:
            st.warning("⚠️ Your fixed costs equal your income. You have no money left to budget.")
        else:
            st.error("❌ Your fixed costs exceed your income! You need to either increase income or reduce fixed costs.")
    
    else:
        st.info("👆 Enter your monthly income above to get started")
    
    # Tips section
    st.markdown("---")
    st.markdown("### 💡 Budgeting Tips")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **50/30/20 Rule**
        - 50% Needs
        - 30% Wants  
        - 20% Savings
        
        *Great for beginners!*
        """)
    
    with col2:
        st.markdown("""
        **Be Realistic**
        - Review past spending
        - Start conservative
        - Adjust monthly
        
        *Track & adapt!*
        """)
    
    with col3:
        st.markdown("""
        **Priority Order**
        1. Fixed costs
        2. Essentials
        3. Savings
        4. Everything else
        """)
