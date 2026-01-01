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
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Lato:wght@300;400;700&display=swap');
    
    /* Main styling - Light Mode */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f8f8f8 100%);
        color: #1a1a1a;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
        color: #FFD700;
        font-weight: 600;
    }
    
    p, div, span, label {
        font-family: 'Lato', sans-serif;
        color: #1a1a1a;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.15);
        border: 2px solid #FFD700;
        margin-bottom: 1rem;
    }
    
    .budget-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #FFD700;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        border: 2px solid #FFD700;
    }
    
    .spending-card {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(220, 53, 69, 0.3);
        border: 2px solid #dc3545;
    }
    
    .income-card {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
        border: 2px solid #28a745;
    }
    
    .remaining-card-green {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
        border: 2px solid #28a745;
    }
    
    .remaining-card-red {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(220, 53, 69, 0.3);
        border: 2px solid #dc3545;
    }
    
    .poetry-card {
        background: linear-gradient(135deg, #fafafa 0%, #f0f0f0 100%);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        border-left: 4px solid #FFD700;
        box-shadow: 0 4px 20px rgba(212, 175, 55, 0.2);
    }
    
    .poetry-text {
        font-family: 'Lato', sans-serif;
        font-size: 1.1rem;
        font-style: italic;
        color: #1a1a1a;
        line-height: 1.8;
        text-align: center;
    }
    
    .poetry-author {
        font-family: 'Lato', sans-serif;
        font-size: 0.9rem;
        color: #4a4a4a;
        text-align: right;
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFD700 !important;
        font-family: 'Montserrat', sans-serif !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: #f5f5f5 !important;
        font-family: 'Lato', sans-serif !important;
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
    .stButton>button {
        background: linear-gradient(135deg, #FFD700 0%, #FFC700 100%);
        color: #000000 !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-family: 'Montserrat', sans-serif;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
        color: #000000 !important;
    }
    
    /* Force ALL button text to be black - every possible selector */
    button {
        color: #000000 !important;
    }
    
    button * {
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
    
    /* Specifically target form submit buttons */
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
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1a1a1a !important;
        border: 1px solid rgba(212, 175, 55, 0.5) !important;
        border-radius: 8px !important;
        font-family: 'Lato', sans-serif !important;
    }
    
    /* Dropdown options */
    option {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Streamlit selectbox */
    [data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1a1a1a !important;
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
        background: #f0f0f0 !important;
        color: #1a1a1a !important;
    }
    
    /* Dataframe/table styling */
    [data-testid="stDataFrame"] {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    /* Make sure text is readable */
    .stMarkdown, .stText {
        color: #1a1a1a;
        font-family: 'Lato', sans-serif;
    }
    
    /* Ensure Streamlit's default theme doesn't override */
    .stApp {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# File paths
TRANSACTIONS_FILE = "transactions.csv"
BUDGETS_FILE = "budgets.csv"

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

# Calculate current month metrics
current_month = datetime.now().month
current_year = datetime.now().year

# Initialize default values
total_income = 0
total_expenses = 0
net_income = 0
current_month_df = pd.DataFrame()

if not transactions_df.empty:
    current_month_df = transactions_df[
        (transactions_df['date'].dt.month == current_month) & 
        (transactions_df['date'].dt.year == current_year)
    ]
    
    total_income = current_month_df[current_month_df['type'] == 'Income']['amount'].sum()
    total_expenses = current_month_df[current_month_df['type'] == 'Expense']['amount'].sum()
    net_income = total_income - total_expenses

# Top metrics - always show
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="income-card">
        <h3 style="margin:0; font-size: 1rem; opacity: 0.9; color: white;">Income This Month</h3>
        <h2 style="margin:0.5rem 0 0 0; font-size: 2rem; color: white;">${total_income:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="spending-card">
        <h3 style="margin:0; font-size: 1rem; opacity: 0.9; color: white;">Spent This Month</h3>
        <h2 style="margin:0.5rem 0 0 0; font-size: 2rem; color: white;">${total_expenses:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    remaining_class = "remaining-card-green" if net_income >= 20 else "remaining-card-red"
    st.markdown(f"""
    <div class="{remaining_class}">
        <h3 style="margin:0; font-size: 1rem; opacity: 0.9; color: white;">Remaining</h3>
        <h2 style="margin:0.5rem 0 0 0; font-size: 2rem; color: white;">${net_income:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Spending by Category
st.subheader("Spending by Category")

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
st.subheader("Budget Overview")

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

# Budget Settings Section
st.markdown("---")
st.subheader("📝 Budget Settings")

with st.expander("Set Monthly Budgets by Category", expanded=False):
    st.write("Set spending limits for each category to help track your financial goals.")
    
    budget_form = st.form("budget_settings")
    
    budget_inputs = {}
    cols = st.columns(2)
    
    for idx, category in enumerate(EXPENSE_CATEGORIES):
        current_budget = 0
        if not budgets_df.empty:
            existing = budgets_df[budgets_df['category'] == category]
            if not existing.empty:
                current_budget = existing.iloc[0]['budget']
        
        with cols[idx % 2]:
            budget_inputs[category] = st.number_input(
                f"{category}", 
                min_value=0.0, 
                value=float(current_budget),
                step=10.0,
                key=f"budget_{category}"
            )
    
    if budget_form.form_submit_button("Save Budgets"):
        new_budgets = []
        for category, amount in budget_inputs.items():
            if amount > 0:
                new_budgets.append({'category': category, 'budget': amount})
        
        save_budgets(pd.DataFrame(new_budgets))
        st.success("Budgets saved successfully!")
        st.rerun()
