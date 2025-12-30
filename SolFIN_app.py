import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os

# Page config
st.set_page_config(
    page_title="Personal Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #f0f2f5 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #2c3e50;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #6c63ff;
        margin-bottom: 1rem;
    }
    
    .budget-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .spending-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    
    .income-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# File paths
TRANSACTIONS_FILE = "transactions.csv"
BUDGETS_FILE = "budgets.csv"

# Default categories
DEFAULT_CATEGORIES = [
    "Groceries", "Rent/Mortgage", "Utilities", "Transportation", 
    "Entertainment", "Dining Out", "Shopping", "Healthcare", 
    "Savings", "Other"
]

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
    # Make sure dates are strings before saving
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df.to_csv(TRANSACTIONS_FILE, index=False)

def delete_transaction(index):
    df = load_transactions()
    df = df.drop(index)
    df = df.reset_index(drop=True)
    # Make sure dates are strings before saving
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df.to_csv(TRANSACTIONS_FILE, index=False)

def save_budgets(budgets_df):
    budgets_df.to_csv(BUDGETS_FILE, index=False)

# Initialize
initialize_files()

# Sidebar - Add Transaction
with st.sidebar:
    st.title("💰 Finance Tracker")
    st.markdown("---")
    
    st.subheader("Add Transaction")
    
    with st.form("transaction_form", clear_on_submit=True):
        trans_date = st.date_input("Date", value=date.today())
        trans_type = st.selectbox("Type", ["Expense", "Income"])
        trans_amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)
        trans_category = st.selectbox("Category", DEFAULT_CATEGORIES)
        trans_description = st.text_input("Description (optional)")
        
        submitted = st.form_submit_button("Add Transaction")
        
        if submitted:
            save_transaction(trans_date, trans_amount, trans_category, trans_type, trans_description)
            st.success("Transaction added!")
            st.rerun()

# Load current data
transactions_df = load_transactions()
budgets_df = load_budgets()

# Main content
st.title("📊 Financial Overview")

# Calculate current month metrics
current_month = datetime.now().month
current_year = datetime.now().year

if not transactions_df.empty:
    current_month_df = transactions_df[
        (transactions_df['date'].dt.month == current_month) & 
        (transactions_df['date'].dt.year == current_year)
    ]
    
    total_income = current_month_df[current_month_df['type'] == 'Income']['amount'].sum()
    total_expenses = current_month_df[current_month_df['type'] == 'Expense']['amount'].sum()
    net_income = total_income - total_expenses
    
    # Top metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="income-card">
            <h3 style="margin:0; font-size: 1rem; opacity: 0.9;">Income This Month</h3>
            <h2 style="margin:0.5rem 0 0 0; font-size: 2rem;">${total_income:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="spending-card">
            <h3 style="margin:0; font-size: 1rem; opacity: 0.9;">Spent This Month</h3>
            <h2 style="margin:0.5rem 0 0 0; font-size: 2rem;">${total_expenses:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        net_color = "income-card" if net_income >= 0 else "spending-card"
        st.markdown(f"""
        <div class="{net_color}">
            <h3 style="margin:0; font-size: 1rem; opacity: 0.9;">Remaining</h3>
            <h2 style="margin:0.5rem 0 0 0; font-size: 2rem;">${net_income:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Spending by Category
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Spending by Category")
        expense_df = current_month_df[current_month_df['type'] == 'Expense']
        
        if not expense_df.empty:
            category_spending = expense_df.groupby('category')['amount'].sum().reset_index()
            category_spending = category_spending.sort_values('amount', ascending=False)
            
            fig = px.pie(
                category_spending, 
                values='amount', 
                names='category',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                showlegend=False,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expenses recorded this month yet.")
    
    with col2:
        st.subheader("Budget Progress")
        
        if not budgets_df.empty:
            budget_progress = []
            
            for _, budget_row in budgets_df.iterrows():
                category = budget_row['category']
                budget_amount = budget_row['budget']
                
                spent = expense_df[expense_df['category'] == category]['amount'].sum()
                remaining = budget_amount - spent
                progress = (spent / budget_amount * 100) if budget_amount > 0 else 0
                
                budget_progress.append({
                    'category': category,
                    'budget': budget_amount,
                    'spent': spent,
                    'remaining': remaining,
                    'progress': min(progress, 100)
                })
            
            for item in budget_progress:
                st.markdown(f"**{item['category']}**")
                st.markdown(f"${item['spent']:.2f} of ${item['budget']:.2f}")
                st.progress(item['progress'] / 100)
                
                if item['remaining'] < 0:
                    st.error(f"⚠️ Over budget by ${abs(item['remaining']):.2f}")
                else:
                    st.success(f"${item['remaining']:.2f} remaining")
                st.markdown("---")
        else:
            st.info("Set budgets below to track your spending limits.")
    
    # Recent transactions
    st.markdown("---")
    st.subheader("Recent Transactions")
    
    recent_df = transactions_df.sort_values('date', ascending=False).head(10)
    
    if not recent_df.empty:
        # Column headers
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
    else:
        st.info("No transactions yet.")

else:
    st.info("👋 Welcome! Start by adding your first transaction using the sidebar.")

# Budget Settings Section
st.markdown("---")
st.subheader("📝 Budget Settings")

with st.expander("Set Monthly Budgets by Category", expanded=False):
    st.write("Set spending limits for each category to help track your financial goals.")
    
    budget_form = st.form("budget_settings")
    
    budget_inputs = {}
    cols = st.columns(2)
    
    for idx, category in enumerate(DEFAULT_CATEGORIES):
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
