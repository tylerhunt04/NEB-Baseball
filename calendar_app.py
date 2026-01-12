"""
Tyler's Life Coordinator - Personal Calendar App

DATA PERSISTENCE NOTE:
----------------------
Your calendar data is saved in 'calendar_data.json' in the same directory as this app.
When you update this Python file, your data will NOT be lost as long as:
1. The 'calendar_data.json' file stays in the same directory
2. You don't delete or rename the data file

To backup your data: Just copy the 'calendar_data.json' file to a safe location
To restore data: Place the 'calendar_data.json' file back in the app directory
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
import calendar as cal

# Page config - Force light mode
st.set_page_config(
    page_title="Tyler's Life Coordinator",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Force light theme
st.markdown("""
<script>
    var stApp = window.parent.document.querySelector('.stApp');
    if (stApp) {
        stApp.classList.remove('dark-theme');
    }
</script>
""", unsafe_allow_html=True)

# Tyler's Custom Color Scheme - Light Mode
NEBRASKA_RED = "#E41C38"  # Keep for baseball

# Light mode earth tones
DARK_BROWN = "#3E2723"  # Text color (dark for visibility)
WOOD_DARK = "#5D4037"   # Headers, important text
WOOD_MED = "#6F4E37"    # Subtext
WOOD_LIGHT = "#8D6E63"  # Less important text

# Light backgrounds
CREAM = "#FAF7F2"
LIGHT_TAN = "#F5E6D3"
LIGHT_BEIGE = "#E8DCC8"
SAND_LIGHT = "#DCC9B0"

# Earth tone palette for categories
TAN_DARK = "#8B7355"
BEIGE_DARK = "#9A7B4F"
SAND_DARK = "#8B6F47"
BROWN = "#6F4E37"

# Category colors - Nebraska red for baseball, earth tones for everything else
CATEGORY_COLORS = {
    "Classes": TAN_DARK,
    "Baseball - Practice": NEBRASKA_RED,
    "Baseball - Game": "#DC143C",
    "Baseball - Travel": "#8B0000",
    "Analytics Work": SAND_DARK,
    "Study Session": BROWN,
    "Personal": BEIGE_DARK,
    "Exam": WOOD_DARK,
    "Assignment": WOOD_LIGHT
}

# File for data persistence - use current directory
DATA_FILE = os.path.join(os.getcwd(), "calendar_data.json")

# Custom CSS - Light Mode Wood & Earth Tone Theme
st.markdown(f"""
<style>
    /* Main app background */
    .main {{
        background-color: {CREAM};
        padding-top: 20px;
    }}
    
    /* Clean, minimalist navigation buttons */
    .stButton>button {{
        background: transparent;
        color: {WOOD_MED};
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        padding: 10px 16px;
        font-weight: 500;
        font-size: 0.95em;
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    
    .stButton>button:hover {{
        background: transparent;
        color: {WOOD_DARK};
        border-bottom-color: {WOOD_LIGHT};
    }}
    
    /* Active/selected navigation button */
    .stButton>button[kind="primary"] {{
        background: transparent;
        color: {WOOD_DARK};
        border-bottom: 3px solid {WOOD_DARK};
        font-weight: 600;
    }}
    
    .stButton>button[kind="primary"]:hover {{
        background: transparent;
        color: {WOOD_DARK};
        border-bottom-color: {WOOD_DARK};
    }}
    
    /* Metric cards - Light backgrounds with dark text */
    .metric-card {{
        background: linear-gradient(135deg, {LIGHT_TAN} 0%, {SAND_LIGHT} 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid {SAND_DARK};
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }}
    
    /* Event cards - Light wood backgrounds */
    .event-card {{
        background: linear-gradient(135deg, {LIGHT_BEIGE} 0%, {LIGHT_TAN} 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 10px 0;
        color: {DARK_BROWN};
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }}
    
    /* Category badges */
    .category-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
        margin-right: 8px;
        color: white;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    }}
    
    /* Headers */
    h1 {{
        color: {WOOD_DARK};
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }}
    h2, h3 {{
        color: {WOOD_MED};
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {LIGHT_TAN} 0%, {CREAM} 100%);
    }}
    
    /* Mini calendar navigation buttons (arrows) */
    [data-testid="stSidebar"] button[key^="mini_prev"],
    [data-testid="stSidebar"] button[key^="mini_next"] {{
        background-color: transparent;
        color: {WOOD_DARK};
        border: none;
        border-radius: 50%;
        padding: 2px 6px;
        font-size: 0.9em;
        min-height: 24px;
        width: 24px;
        transition: background-color 0.2s ease;
    }}
    
    [data-testid="stSidebar"] button[key^="mini_prev"]:hover,
    [data-testid="stSidebar"] button[key^="mini_next"]:hover {{
        background-color: {LIGHT_BEIGE};
        color: {WOOD_DARK};
    }}
    
    /* Filter buttons in sidebar - different styling */
    [data-testid="stSidebar"] button[kind="secondary"] {{
        background: linear-gradient(135deg, {LIGHT_BEIGE} 0%, {SAND_LIGHT} 100%);
        color: {WOOD_DARK};
        border: 1px solid {WOOD_LIGHT};
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        width: 100%;
        min-height: 40px;
        transition: all 0.2s ease;
    }}
    
    [data-testid="stSidebar"] button[kind="secondary"]:hover {{
        background: linear-gradient(135deg, {WOOD_LIGHT} 0%, {TAN_DARK} 100%);
        color: {CREAM};
        border-color: {WOOD_MED};
    }}
    
    [data-testid="stSidebar"] button[kind="primary"] {{
        background: linear-gradient(135deg, {WOOD_DARK} 0%, {WOOD_MED} 100%);
        color: {CREAM};
        border: 1px solid {WOOD_DARK};
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        width: 100%;
        min-height: 40px;
    }}
    
    [data-testid="stSidebar"] button[kind="primary"]:hover {{
        background: linear-gradient(135deg, {WOOD_DARK} 0%, {WOOD_MED} 100%);
        color: {CREAM};
    }}
    
    /* Radio buttons and labels in sidebar */
    .stRadio > label {{
        color: {DARK_BROWN} !important;
        font-weight: 500;
    }}
    
    /* Form labels */
    label {{
        color: {WOOD_DARK} !important;
        font-weight: 500;
    }}
    
    /* Text in sidebar */
    [data-testid="stSidebar"] * {{
        color: {DARK_BROWN} !important;
    }}
    
    /* Fix selectbox and input backgrounds */
    .stSelectbox > div > div {{
        background-color: white !important;
        border: 1px solid {WOOD_LIGHT} !important;
    }}
    
    .stTextInput > div > div > input {{
        background-color: white !important;
        border: 1px solid {WOOD_LIGHT} !important;
        color: {DARK_BROWN} !important;
    }}
    
    .stTextArea > div > div > textarea {{
        background-color: white !important;
        border: 1px solid {WOOD_LIGHT} !important;
        color: {DARK_BROWN} !important;
    }}
    
    /* Date and time inputs */
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {{
        background-color: white !important;
        border: 1px solid {WOOD_LIGHT} !important;
        color: {DARK_BROWN} !important;
    }}
    
    /* Info boxes */
    .stAlert {{
        background-color: {LIGHT_BEIGE} !important;
        color: {DARK_BROWN} !important;
        border-left: 4px solid {SAND_DARK} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'events' not in st.session_state:
    st.session_state.events = []
    
if 'view_date' not in st.session_state:
    st.session_state.view_date = datetime.now()

# Data persistence functions
def load_events():
    """Load events from JSON file"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
                # Convert string dates back to datetime
                for event in data:
                    event['start'] = datetime.fromisoformat(event['start'])
                    event['end'] = datetime.fromisoformat(event['end'])
                return data
    except Exception as e:
        st.warning(f"Could not load saved events: {str(e)}")
    return []

def save_events():
    """Save events to JSON file"""
    try:
        # Convert datetime to string for JSON serialization
        events_to_save = []
        for event in st.session_state.events:
            event_copy = event.copy()
            event_copy['start'] = event['start'].isoformat()
            event_copy['end'] = event['end'].isoformat()
            events_to_save.append(event_copy)
        
        with open(DATA_FILE, 'w') as f:
            json.dump(events_to_save, f, indent=2)
    except Exception as e:
        st.error(f"Error saving events: {str(e)}")
        # Still continue - events are in session state

# Load events on startup
if not st.session_state.events:
    st.session_state.events = load_events()

def add_event(title, category, start, end, location="", notes="", priority="Medium", 
              is_travel=False, checklist_items=None, time_tba=False, single_time=False):
    """Add a new event"""
    event = {
        'id': len(st.session_state.events),
        'title': title,
        'category': category,
        'start': start,
        'end': end,
        'location': location,
        'notes': notes,
        'priority': priority,
        'is_travel': is_travel,
        'checklist': checklist_items or [],
        'completed_checklist': [],
        'time_tba': time_tba,
        'single_time': single_time
    }
    st.session_state.events.append(event)
    save_events()
    return event

def delete_event(event_id):
    """Delete an event"""
    st.session_state.events = [e for e in st.session_state.events if e['id'] != event_id]
    save_events()

def get_events_for_date(date):
    """Get all events for a specific date, respecting active filter"""
    event_filter = st.session_state.get('event_filter', 'All')
    
    events = []
    for event in st.session_state.events:
        if event['start'].date() == date.date():
            events.append(event)
    
    # Apply filter
    if event_filter != "All":
        if event_filter == "Baseball":
            events = [e for e in events if "Baseball" in e['category']]
        elif event_filter == "Academic":
            events = [e for e in events if e['category'] in ["Exam", "Assignment", "Study Session", "Classes"]]
        elif event_filter == "Analytics":
            events = [e for e in events if e['category'] == "Analytics Work"]
    
    return sorted(events, key=lambda x: x['start'])

def get_events_for_range(start_date, end_date):
    """Get all events within a date range, respecting active filter"""
    event_filter = st.session_state.get('event_filter', 'All')
    
    events = []
    for event in st.session_state.events:
        if start_date.date() <= event['start'].date() <= end_date.date():
            events.append(event)
    
    # Apply filter
    if event_filter != "All":
        if event_filter == "Baseball":
            events = [e for e in events if "Baseball" in e['category']]
        elif event_filter == "Academic":
            events = [e for e in events if e['category'] in ["Exam", "Assignment", "Study Session", "Classes"]]
        elif event_filter == "Analytics":
            events = [e for e in events if e['category'] == "Analytics Work"]
    
    return sorted(events, key=lambda x: x['start'])

def render_mini_calendar_sidebar():
    """Render Google Calendar style mini calendar in sidebar"""
    
    # Initialize mini calendar date in session state
    if 'mini_cal_date' not in st.session_state:
        st.session_state.mini_cal_date = datetime.now()
    
    # Month navigation - compact
    col1, col2, col3 = st.sidebar.columns([0.8, 3, 0.8])
    
    with col1:
        if st.button("◀", key="mini_prev", help="Previous month"):
            if st.session_state.mini_cal_date.month == 1:
                st.session_state.mini_cal_date = st.session_state.mini_cal_date.replace(
                    year=st.session_state.mini_cal_date.year - 1, month=12, day=1
                )
            else:
                st.session_state.mini_cal_date = st.session_state.mini_cal_date.replace(
                    month=st.session_state.mini_cal_date.month - 1, day=1
                )
            st.rerun()
    
    with col2:
        st.markdown(f"<div style='text-align: center; font-weight: 600; font-size: 0.9em; color: {WOOD_DARK}; padding: 4px 0;'>{st.session_state.mini_cal_date.strftime('%B %Y')}</div>", unsafe_allow_html=True)
    
    with col3:
        if st.button("▶", key="mini_next", help="Next month"):
            if st.session_state.mini_cal_date.month == 12:
                st.session_state.mini_cal_date = st.session_state.mini_cal_date.replace(
                    year=st.session_state.mini_cal_date.year + 1, month=1, day=1
                )
            else:
                st.session_state.mini_cal_date = st.session_state.mini_cal_date.replace(
                    month=st.session_state.mini_cal_date.month + 1, day=1
                )
            st.rerun()
    
    # Get calendar for the month
    year = st.session_state.mini_cal_date.year
    month = st.session_state.mini_cal_date.month
    month_cal = cal.monthcalendar(year, month)
    
    # Day headers
    days = ['S', 'M', 'T', 'W', 'T', 'F', 'S']
    header_parts = []
    header_parts.append('<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; text-align: center; font-weight: 600; color: ' + WOOD_MED + '; margin: 8px 0 4px 0; font-size: 0.7em;">')
    for day in days:
        header_parts.append(f'<div style="padding: 2px;">{day}</div>')
    header_parts.append('</div>')
    st.sidebar.markdown(''.join(header_parts), unsafe_allow_html=True)
    
    # Calendar grid - HTML only, not clickable
    for week in month_cal:
        week_html = []
        week_html.append('<div style="display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; margin-bottom: 2px;">')
        
        for day in week:
            if day == 0:
                week_html.append('<div style="height: 24px;"></div>')
            else:
                date = datetime(year, month, day)
                events = get_events_for_date(date)
                
                is_today = date.date() == datetime.now().date()
                has_events = len(events) > 0
                
                # Styling
                if is_today:
                    # Blue circle for today
                    bg_style = f"background-color: #1a73e8; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-weight: 600; margin: 0 auto; font-size: 0.75em;"
                elif has_events:
                    # Subtle background for days with events
                    bg_style = f"background-color: {LIGHT_BEIGE}; color: {WOOD_DARK}; border-radius: 3px; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-weight: 500; font-size: 0.75em;"
                else:
                    # Normal day
                    bg_style = f"color: {WOOD_MED}; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 0.75em;"
                
                # Add day number
                week_html.append(f'<div style="{bg_style}" title="{date.strftime("%b %d, %Y")} - {len(events)} event(s)">{day}</div>')
        
        week_html.append('</div>')
        st.sidebar.markdown(''.join(week_html), unsafe_allow_html=True)
    
    # Add spacing after calendar
    st.sidebar.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

def create_quick_event_templates():
    """Deprecated - not used"""
    pass



def render_event_card(event):
    """Render a styled event card"""
    color = CATEGORY_COLORS.get(event['category'], "#888888")
    
    duration = event['end'] - event['start']
    duration_str = f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"
    
    priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(event['priority'], "⚪")
    travel_badge = " ✈️ TRAVEL" if event.get('is_travel') else ""
    
    # Build HTML parts separately
    html_parts = []
    
    # Start card
    html_parts.append(f'<div class="event-card" style="border-left-color: {color};">')
    html_parts.append('<div style="display: flex; justify-content: space-between; align-items: start;">')
    html_parts.append('<div style="width: 100%;">')
    
    # Title
    html_parts.append(f'<h3 style="margin: 0; color: {DARK_BROWN};">{priority_emoji} {event["title"]}{travel_badge}</h3>')
    
    # Category badge
    html_parts.append(f'<p style="margin: 5px 0; color: {WOOD_MED};">')
    html_parts.append(f'<span class="category-badge" style="background-color: {color};">{event["category"]}</span>')
    html_parts.append('</p>')
    
    # Date and time
    html_parts.append(f'<p style="margin: 5px 0; color: {WOOD_DARK};">')
    html_parts.append(f'📅 {event["start"].strftime("%b %d, %Y")} | ')
    
    # Check if time is TBA
    if event.get('time_tba', False):
        html_parts.append('🕐 <strong style="color: #F0AD4E;">Time TBA</strong>')
    elif event.get('single_time', False) or event['start'] == event['end']:
        # Single time event (deadline/assignment)
        html_parts.append(f'🕐 Due at {event["start"].strftime("%I:%M %p")}')
    else:
        html_parts.append(f'🕐 {event["start"].strftime("%I:%M %p")} - {event["end"].strftime("%I:%M %p")} ')
        html_parts.append(f'({duration_str})')
    
    html_parts.append('</p>')
    
    # Location (if exists)
    if event.get('location'):
        html_parts.append(f'<p style="margin: 5px 0; color: {WOOD_DARK};">📍 {event["location"]}</p>')
    
    # Notes (if exists)
    if event.get('notes'):
        html_parts.append(f'<p style="margin: 5px 0; color: {WOOD_MED};">📝 {event["notes"]}</p>')
    
    # Close divs
    html_parts.append('</div>')
    html_parts.append('</div>')
    html_parts.append('</div>')
    
    # Join and render
    full_html = ''.join(html_parts)
    st.markdown(full_html, unsafe_allow_html=True)
    
    # Show checklist if exists
    if event.get('checklist'):
        st.markdown("**Travel Checklist:**")
        for i, item in enumerate(event['checklist']):
            checked = item in event.get('completed_checklist', [])
            if st.checkbox(item, value=checked, key=f"check_{event['id']}_{i}"):
                if item not in event['completed_checklist']:
                    event['completed_checklist'].append(item)
                    save_events()
            else:
                if item in event['completed_checklist']:
                    event['completed_checklist'].remove(item)
                    save_events()

def create_workload_heatmap():
    """Create a heatmap showing workload by day and hour"""
    # Create hour-by-day matrix
    days = 7
    hours = 24
    matrix = [[0 for _ in range(hours)] for _ in range(days)]
    
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for event in st.session_state.events:
        event_start = event['start']
        event_end = event['end']
        
        # Check if event is within next 7 days
        if start_date <= event_start < start_date + timedelta(days=7):
            day_offset = (event_start.date() - start_date.date()).days
            start_hour = event_start.hour
            end_hour = event_end.hour if event_end.date() == event_start.date() else 23
            
            for hour in range(start_hour, min(end_hour + 1, 24)):
                matrix[day_offset][hour] += 1
    
    # Create heatmap
    day_labels = [(start_date + timedelta(days=i)).strftime('%a %m/%d') for i in range(days)]
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=hour_labels,
        y=day_labels,
        colorscale=[[0, CREAM], [0.3, LIGHT_TAN], [0.6, SAND_DARK], [1, WOOD_DARK]],
        showscale=True,
        hovertemplate='%{y}<br>%{x}<br>Events: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Workload Heatmap - Next 7 Days",
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        height=400,
        paper_bgcolor=CREAM,
        plot_bgcolor=CREAM,
        font=dict(color=DARK_BROWN),
        title_font=dict(color=WOOD_DARK, size=18)
    )
    
    return fig

def create_category_breakdown():
    """Create pie chart of time by category"""
    category_time = defaultdict(float)
    
    for event in st.session_state.events:
        duration = (event['end'] - event['start']).total_seconds() / 3600  # hours
        category_time[event['category']] += duration
    
    if not category_time:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=list(category_time.keys()),
        values=list(category_time.values()),
        marker=dict(colors=[CATEGORY_COLORS.get(cat, '#888888') for cat in category_time.keys()]),
        hole=0.4,
        textinfo='label+percent',
        textfont=dict(color='white')
    )])
    
    fig.update_layout(
        title="Time Allocation by Category",
        height=400,
        paper_bgcolor=CREAM,
        plot_bgcolor=CREAM,
        font=dict(color=DARK_BROWN),
        title_font=dict(color=WOOD_DARK, size=18),
        showlegend=True,
        legend=dict(font=dict(color=DARK_BROWN))
    )
    
    return fig

def render_next_48_hours():
    """Render quick dashboard for next 48 hours"""
    
    now = datetime.now()
    end_time = now + timedelta(hours=48)
    
    events = get_events_for_range(now, end_time)
    
    if not events:
        st.info("No events in the next 48 hours. You're free!")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: {DARK_BROWN};">{len(events)}</h3>
            <p style="margin: 5px 0; color: {WOOD_MED};">Total Events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        travel_events = sum(1 for e in events if e.get('is_travel'))
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: {DARK_BROWN};">{travel_events}</h3>
            <p style="margin: 5px 0; color: {WOOD_MED};">Travel Required</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_priority = sum(1 for e in events if e.get('priority') == 'High')
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: {DARK_BROWN};">{high_priority}</h3>
            <p style="margin: 5px 0; color: {WOOD_MED};">High Priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    for event in events:
        time_until = event['start'] - now
        hours_until = time_until.total_seconds() / 3600
        
        if hours_until < 0:
            time_str = "In Progress"
            time_color = "#DC143C"  # Red
        elif hours_until < 2:
            time_str = f"{int(hours_until * 60)}m"
            time_color = "#DC143C"  # Red
        elif hours_until < 6:
            time_str = f"{hours_until:.1f}h"
            time_color = "#F0AD4E"  # Orange/Yellow
        else:
            time_str = f"{hours_until:.1f}h"
            time_color = "#5CB85C"  # Green
        
        col1, col2 = st.columns([3, 1])
        with col1:
            render_event_card(event)
        with col2:
            st.markdown(f"""
            <div style="
                background-color: {LIGHT_BEIGE}; 
                border-left: 3px solid {time_color}; 
                border-radius: 6px; 
                padding: 12px 8px;
                text-align: center;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <span style="
                    font-size: 0.95em; 
                    font-weight: 500; 
                    color: {WOOD_DARK};
                ">{time_str}</span>
            </div>
            """, unsafe_allow_html=True)

def render_day_view():
    """Render single day view"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ Previous Day"):
            st.session_state.view_date -= timedelta(days=1)
            st.rerun()
    
    with col2:
        selected_date = st.date_input(
            "Select Date",
            value=st.session_state.view_date,
            key="day_view_date"
        )
        st.session_state.view_date = datetime.combine(selected_date, datetime.min.time())
    
    with col3:
        if st.button("Next Day ▶"):
            st.session_state.view_date += timedelta(days=1)
            st.rerun()
    
    events = get_events_for_date(st.session_state.view_date)
    
    if not events:
        st.info("No events scheduled for this day.")
    else:
        total_hours = sum((e['end'] - e['start']).total_seconds() / 3600 for e in events)
        st.markdown(f"**{len(events)} events | {total_hours:.1f} hours scheduled**")
        
        for event in events:
            render_event_card(event)
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Delete", key=f"del_{event['id']}"):
                    delete_event(event['id'])
                    st.rerun()

def render_week_view():
    """Render Google Calendar-style week view with time grid"""
    
    # Week navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ Previous Week"):
            st.session_state.view_date -= timedelta(days=7)
            st.rerun()
    
    with col2:
        week_start = st.session_state.view_date - timedelta(days=st.session_state.view_date.weekday())
        week_end = week_start + timedelta(days=6)
        st.markdown(f"<p style='text-align: center; color: {WOOD_MED}; font-size: 0.9em;'>{week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}</p>", unsafe_allow_html=True)
    
    with col3:
        if st.button("Next Week ▶"):
            st.session_state.view_date += timedelta(days=7)
            st.rerun()
    
    # Calculate week start
    week_start = st.session_state.view_date - timedelta(days=st.session_state.view_date.weekday())
    
    # Build HTML for calendar grid
    html_parts = []
    
    # Start calendar container
    html_parts.append('<div style="width: 100%; overflow-x: auto; border: 1px solid ' + WOOD_LIGHT + '; border-radius: 8px; background-color: white;">')
    html_parts.append('<div style="display: grid; grid-template-columns: 60px repeat(7, 1fr); min-width: 800px;">')
    
    # Header row with day names
    html_parts.append('<div style="position: sticky; top: 0; background-color: white; z-index: 10;"></div>')  # Empty corner
    
    for i in range(7):
        day = week_start + timedelta(days=i)
        is_today = day.date() == datetime.now().date()
        
        if is_today:
            header_style = "background: linear-gradient(135deg, " + WOOD_DARK + " 0%, " + WOOD_MED + " 100%); color: " + CREAM + ";"
        else:
            header_style = "background-color: " + LIGHT_BEIGE + "; color: " + WOOD_DARK + ";"
        
        html_parts.append('<div style="' + header_style + ' padding: 12px 8px; text-align: center; font-weight: 600; font-size: 0.9em; border-bottom: 2px solid ' + WOOD_LIGHT + '; position: sticky; top: 0; z-index: 10;">')
        html_parts.append('<div>' + day.strftime("%a") + '</div>')
        html_parts.append('<div style="font-size: 1.1em; margin-top: 2px;">' + day.strftime("%d") + '</div>')
        html_parts.append('</div>')
    
    # Time rows (6 AM to 11 PM = 18 hours)
    start_hour = 6  # 6 AM
    end_hour = 23   # 11 PM
    
    # Get all events for the week
    week_events = {}
    for i in range(7):
        day = week_start + timedelta(days=i)
        week_events[i] = get_events_for_date(day)
    
    # Create time slots
    for hour in range(start_hour, end_hour + 1):
        # Time label
        display_hour = hour % 12
        if display_hour == 0:
            display_hour = 12
        period = "AM" if hour < 12 else "PM"
        
        html_parts.append('<div style="padding: 8px 4px; text-align: right; font-size: 0.75em; color: ' + WOOD_MED + '; border-top: 1px solid ' + LIGHT_BEIGE + '; border-right: 1px solid ' + WOOD_LIGHT + '; font-weight: 500;">')
        html_parts.append(str(display_hour) + period)
        html_parts.append('</div>')
        
        # Day columns for this hour
        for day_idx in range(7):
            html_parts.append('<div style="min-height: 60px; border-top: 1px solid ' + LIGHT_BEIGE + '; border-left: 1px solid #e8e8e8; position: relative; padding: 2px;">')
            
            # Find events that occur during this hour
            events = week_events[day_idx]
            for event in events:
                event_start_hour = event['start'].hour + event['start'].minute / 60.0
                event_end_hour = event['end'].hour + event['end'].minute / 60.0
                
                # Check if event overlaps with this hour slot
                if event_start_hour < hour + 1 and event_end_hour > hour:
                    # Only show event in the slot where it starts
                    if hour <= event_start_hour < hour + 1:
                        # Calculate full height across multiple slots
                        total_duration = event_end_hour - event_start_hour
                        height_px = int(total_duration * 60)  # 60px per hour
                        
                        # Offset from top of slot
                        offset_minutes = (event_start_hour - hour) * 60
                        top_offset = int(offset_minutes)
                        
                        color = CATEGORY_COLORS.get(event['category'], "#888888")
                        
                        # Build event HTML
                        html_parts.append('<div style="position: absolute; top: ' + str(top_offset) + 'px; left: 4px; right: 4px; height: ' + str(height_px) + 'px; background-color: ' + color + '; color: white; padding: 4px 6px; border-radius: 4px; font-size: 0.75em; overflow: hidden; cursor: pointer; border-left: 3px solid ' + color + '; filter: brightness(0.95); box-shadow: 0 1px 3px rgba(0,0,0,0.1);">')
                        html_parts.append('<div style="font-weight: 600; margin-bottom: 2px;">' + event['title'][:20] + '</div>')
                        
                        # Show time if event is tall enough
                        if height_px > 30:
                            if event.get('time_tba'):
                                html_parts.append('<div style="font-size: 0.9em;">TBA</div>')
                            elif event.get('single_time'):
                                html_parts.append('<div style="font-size: 0.9em;">' + event["start"].strftime("%I:%M %p") + '</div>')
                            else:
                                html_parts.append('<div style="font-size: 0.9em;">' + event["start"].strftime("%I:%M %p") + '</div>')
                        
                        html_parts.append('</div>')
            
            html_parts.append('</div>')
    
    # Close grid container
    html_parts.append('</div>')
    html_parts.append('</div>')
    
    # Join all HTML and render
    full_html = ''.join(html_parts)
    
    # Render the calendar
    st.markdown(full_html, unsafe_allow_html=True)
    
    # Show legend
    st.markdown("---")
    st.markdown("**Legend:**")
    legend_cols = st.columns(len(CATEGORY_COLORS))
    for idx, (category, color) in enumerate(CATEGORY_COLORS.items()):
        with legend_cols[idx]:
            st.markdown(f'<div style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; text-align: center;">{category}</div>', unsafe_allow_html=True)

def render_month_view():
    """Render month view"""
    
    # Month navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ Previous Month"):
            # Go to previous month
            if st.session_state.view_date.month == 1:
                st.session_state.view_date = st.session_state.view_date.replace(year=st.session_state.view_date.year - 1, month=12, day=1)
            else:
                st.session_state.view_date = st.session_state.view_date.replace(month=st.session_state.view_date.month - 1, day=1)
            st.rerun()
    
    with col2:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)  # Spacer for alignment
    
    with col3:
        if st.button("Next Month ▶"):
            # Go to next month
            if st.session_state.view_date.month == 12:
                st.session_state.view_date = st.session_state.view_date.replace(year=st.session_state.view_date.year + 1, month=1, day=1)
            else:
                st.session_state.view_date = st.session_state.view_date.replace(month=st.session_state.view_date.month + 1, day=1)
            st.rerun()
    
    # Get calendar for month
    year = st.session_state.view_date.year
    month = st.session_state.view_date.month
    
    month_cal = cal.monthcalendar(year, month)
    
    # Display calendar header
    cols = st.columns(7)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for i, day in enumerate(days):
        cols[i].markdown(f"**{day}**")
    
    # Display calendar days
    for week in month_cal:
        cols = st.columns(7)
        for i, day in enumerate(week):
            with cols[i]:
                if day == 0:
                    st.markdown("")
                else:
                    date = datetime(year, month, day)
                    events = get_events_for_date(date)
                    
                    is_today = date.date() == datetime.now().date()
                    
                    # Build HTML parts
                    html_parts = []
                    
                    # Determine styling
                    if is_today:
                        bg_style = f"background: linear-gradient(135deg, {WOOD_DARK} 0%, {WOOD_MED} 100%); color: {CREAM};"
                        text_color = CREAM
                    else:
                        bg_style = f"background-color: {LIGHT_BEIGE}; color: {DARK_BROWN}; border: 1px solid {WOOD_LIGHT};"
                        text_color = WOOD_MED
                    
                    # Start div
                    html_parts.append(f'<div style="{bg_style} padding: 10px; border-radius: 5px; min-height: 100px;">')
                    
                    # Day number
                    html_parts.append(f'<div style="text-align: center; font-weight: bold;">{day}</div>')
                    
                    # Event count
                    html_parts.append(f'<div style="font-size: 0.75em; margin-top: 5px; color: {text_color}; text-align: center;">')
                    if events:
                        event_count = len(events)
                        if event_count == 1:
                            html_parts.append("1 event")
                        else:
                            html_parts.append(f"{event_count} events")
                    html_parts.append('</div>')
                    
                    # Close div
                    html_parts.append('</div>')
                    
                    # Render
                    full_html = ''.join(html_parts)
                    st.markdown(full_html, unsafe_allow_html=True)

def render_baseball_season_tracker():
    """Track baseball season progress"""
    st.markdown("## Baseball Season Tracker")
    
    # Get baseball events
    baseball_events = [e for e in st.session_state.events if 'Baseball' in e['category']]
    games = [e for e in baseball_events if e['category'] == 'Baseball - Game']
    practices = [e for e in baseball_events if e['category'] == 'Baseball - Practice']
    
    # Season stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {NEBRASKA_RED};">
            <h2 style="margin: 0; color: {DARK_BROWN};">{len(games)}</h2>
            <p style="margin: 5px 0; color: {WOOD_MED};">Games Scheduled</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {NEBRASKA_RED};">
            <h2 style="margin: 0; color: {DARK_BROWN};">{len(practices)}</h2>
            <p style="margin: 5px 0; color: {WOOD_MED};">Practices Scheduled</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        away_games = sum(1 for g in games if g.get('is_travel'))
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {NEBRASKA_RED};">
            <h2 style="margin: 0; color: {DARK_BROWN};">{away_games}</h2>
            <p style="margin: 5px 0; color: {WOOD_MED};">Away Games</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        upcoming_games = [g for g in games if g['start'] > datetime.now()]
        next_game = min(upcoming_games, key=lambda x: x['start']) if upcoming_games else None
        
        if next_game:
            days_until = (next_game['start'].date() - datetime.now().date()).days
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {NEBRASKA_RED};">
                <h2 style="margin: 0; color: {DARK_BROWN};">{days_until}</h2>
                <p style="margin: 5px 0; color: {WOOD_MED};">Days to Next Game</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {NEBRASKA_RED};">
                <h2 style="margin: 0; color: {DARK_BROWN};">-</h2>
                <p style="margin: 5px 0; color: {WOOD_MED};">No Games Scheduled</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Upcoming games
    if upcoming_games:
        st.markdown("### Upcoming Games")
        for game in upcoming_games[:5]:
            render_event_card(game)

def render_academic_tracker():
    """Track academic deadlines and exams"""
    st.markdown("## Academic Tracker")
    
    # Get academic events
    exams = [e for e in st.session_state.events if e['category'] == 'Exam']
    assignments = [e for e in st.session_state.events if e['category'] == 'Assignment']
    classes = [e for e in st.session_state.events if e['category'] == 'Classes']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        upcoming_exams = [e for e in exams if e['start'] > datetime.now()]
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; color: {DARK_BROWN};">{len(upcoming_exams)}</h2>
            <p style="margin: 5px 0; color: {WOOD_MED};">Upcoming Exams</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        upcoming_assignments = [e for e in assignments if e['start'] > datetime.now()]
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; color: {DARK_BROWN};">{len(upcoming_assignments)}</h2>
            <p style="margin: 5px 0; color: {WOOD_MED};">Pending Assignments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        next_exam = min(upcoming_exams, key=lambda x: x['start']) if upcoming_exams else None
        if next_exam:
            days_until = (next_exam['start'].date() - datetime.now().date()).days
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0; color: {DARK_BROWN};">{days_until}</h2>
                <p style="margin: 5px 0; color: {WOOD_MED};">Days to Next Exam</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0; color: {DARK_BROWN};">-</h2>
                <p style="margin: 5px 0; color: {WOOD_MED};">No Exams Scheduled</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Crunch time alerts
    week_from_now = datetime.now() + timedelta(days=7)
    upcoming_week_deadlines = [e for e in exams + assignments if datetime.now() < e['start'] < week_from_now]
    
    if len(upcoming_week_deadlines) >= 3:
        st.warning(f"⚠️ **CRUNCH TIME ALERT**: {len(upcoming_week_deadlines)} deadlines/exams in the next 7 days!")
    
    # Show upcoming deadlines
    if upcoming_exams or upcoming_assignments:
        st.markdown("### Upcoming Deadlines")
        all_upcoming = sorted(upcoming_exams + upcoming_assignments, key=lambda x: x['start'])
        for item in all_upcoming[:5]:
            render_event_card(item)

def render_analytics_dashboard():
    """Render time management analytics"""
    st.markdown("## Time Management Analytics")
    
    # Workload heatmap
    heatmap = create_workload_heatmap()
    st.plotly_chart(heatmap, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category breakdown
        pie_chart = create_category_breakdown()
        if pie_chart:
            st.plotly_chart(pie_chart, use_container_width=True)
    
    with col2:
        # Free time finder
        st.markdown("### Free Time Windows - Next 7 Days")
        
        now = datetime.now()
        end_of_week = now + timedelta(days=7)
        
        events = get_events_for_range(now, end_of_week)
        
        # Find gaps between events
        free_blocks = []
        events_sorted = sorted(events, key=lambda x: x['start'])
        
        for i in range(len(events_sorted) - 1):
            gap_start = events_sorted[i]['end']
            gap_end = events_sorted[i + 1]['start']
            gap_hours = (gap_end - gap_start).total_seconds() / 3600
            
            if gap_hours >= 2:  # At least 2 hours free
                free_blocks.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration': gap_hours
                })
        
        if free_blocks:
            for block in free_blocks[:5]:
                st.markdown(f"""
                <div class="metric-card">
                    <p style="margin: 0; color: {DARK_BROWN};">
                        <strong>{block['start'].strftime('%a %m/%d')}</strong><br>
                        {block['start'].strftime('%I:%M %p')} - {block['end'].strftime('%I:%M %p')}<br>
                        <span style="color: {WOOD_DARK};">{block['duration']:.1f} hours free</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant free time blocks found. Consider optimizing your schedule!")

def render_add_event_form():
    """Form to add new events with recurring option"""
    
    # Recurring checkbox OUTSIDE the form so it updates immediately
    is_recurring = st.checkbox("Recurring Event (e.g., classes, practices)", key="is_recurring_checkbox")
    
    # Time TBA checkbox OUTSIDE the form with explicit key
    time_tba = st.checkbox("Time TBA (game time not announced yet)", key="time_tba_checkbox")
    
    # Single time checkbox for deadlines/assignments
    single_time = st.checkbox("Single time (assignment/deadline due at specific time)", key="single_time_checkbox")
    
    if is_recurring:
        st.info("Select the days of the week, date range, and time for your recurring event")
    
    if time_tba:
        st.info("✓ Event will be created with time marked as TBA. You can edit it later once the time is announced.")
    
    if single_time:
        st.info("✓ Event will be created with a single due time (e.g., assignment due at 11:59 PM)")
    
    st.markdown("---")
    
    with st.form("add_event_form"):
        title = st.text_input("Event Title*")
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category*", list(CATEGORY_COLORS.keys()))
        with col2:
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        
        if is_recurring:
            st.markdown("### Recurring Event Settings")
            
            # Days of week selection
            st.markdown("**Select Days of Week***")
            day_cols = st.columns(7)
            days_selected = {}
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_abbrev = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            for i, (name, abbr) in enumerate(zip(day_names, day_abbrev)):
                with day_cols[i]:
                    days_selected[name] = st.checkbox(abbr, key=f"day_{i}")
            
            st.markdown("---")
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                recur_start_date = st.date_input("Start Date*", key="recur_start")
            with col2:
                recur_end_date = st.date_input("End Date*", key="recur_end")
            
            # Time (same for all occurrences)
            st.markdown("**Event Time (applies to all days)***")
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.markdown("**Start Time**")
                scol1, scol2, scol3 = st.columns([2, 2, 1])
                with scol1:
                    start_hour = st.selectbox("Hour", range(1, 13), key="recur_start_hour", index=1)
                with scol2:
                    start_minute = st.number_input("Minute", min_value=0, max_value=59, value=0, key="recur_start_min")
                with scol3:
                    start_period = st.selectbox("AM/PM", ["AM", "PM"], key="recur_start_period", index=1)
            
            with tcol2:
                st.markdown("**End Time**")
                ecol1, ecol2, ecol3 = st.columns([2, 2, 1])
                with ecol1:
                    end_hour = st.selectbox("Hour", range(1, 13), key="recur_end_hour", index=2)
                with ecol2:
                    end_minute = st.number_input("Minute", min_value=0, max_value=59, value=15, key="recur_end_min")
                with ecol3:
                    end_period = st.selectbox("AM/PM", ["AM", "PM"], key="recur_end_period", index=1)
            
        else:
            st.markdown("### Single Event")
            
            # Single event
            if time_tba:
                # Just show date field when time is TBA
                st.markdown("**Event Date***")
                start_date = st.date_input("Select the date", label_visibility="collapsed")
                
                st.markdown(f"""
                <div style='
                    background-color: {LIGHT_BEIGE}; 
                    border-left: 3px solid #F0AD4E;
                    border-radius: 6px;
                    padding: 12px;
                    margin: 10px 0;
                '>
                    <p style='margin: 0; color: {WOOD_DARK}; font-size: 0.9em;'>
                        ⏰ <strong>Time:</strong> TBA (to be announced)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Set dummy values for time (all-day placeholder)
                start_hour, start_minute, start_period = 12, 0, "AM"
                end_hour, end_minute, end_period = 11, 59, "PM"
                end_date = start_date
            elif single_time:
                # Single time selection for deadlines
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Due Date*")
                with col2:
                    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)  # Spacer
                
                st.markdown("**Due Time***")
                tcol1, tcol2, tcol3 = st.columns([2, 2, 1])
                with tcol1:
                    start_hour = st.selectbox("Hour", range(1, 13), key="single_hour", index=10)  # Default 11
                with tcol2:
                    start_minute = st.number_input("Minute", min_value=0, max_value=59, value=59, key="single_min")
                with tcol3:
                    start_period = st.selectbox("AM/PM", ["AM", "PM"], key="single_period", index=1)  # Default PM
                
                # End time is same as start for single time events
                end_hour, end_minute, end_period = start_hour, start_minute, start_period
                end_date = start_date
            else:
                # Normal time selection with time span
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date*")
                    
                    # 12-hour format time selection with any minute
                    st.markdown("**Start Time***")
                    scol1, scol2, scol3 = st.columns([2, 2, 1])
                    with scol1:
                        start_hour = st.selectbox("Hour", range(1, 13), key="start_hour", index=1)
                    with scol2:
                        start_minute = st.number_input("Minute", min_value=0, max_value=59, value=0, key="start_min")
                    with scol3:
                        start_period = st.selectbox("AM/PM", ["AM", "PM"], key="start_period", index=1)
                
                with col2:
                    end_date = st.date_input("End Date*")
                    
                    # 12-hour format time selection with any minute
                    st.markdown("**End Time***")
                    ecol1, ecol2, ecol3 = st.columns([2, 2, 1])
                    with ecol1:
                        end_hour = st.selectbox("Hour", range(1, 13), key="end_hour", index=2)
                    with ecol2:
                        end_minute = st.number_input("Minute", min_value=0, max_value=59, value=0, key="end_min")
                    with ecol3:
                        end_period = st.selectbox("AM/PM", ["AM", "PM"], key="end_period", index=1)
        
        st.markdown("---")
        location = st.text_input("Location")
        notes = st.text_area("Notes")
        
        is_travel = st.checkbox("Requires Travel")
        
        checklist_input = ""
        if is_travel:
            checklist_input = st.text_area(
                "Travel Checklist (one item per line)",
                value="Pack analytics equipment\nPack clothes\nCharge laptop\nDownload data"
            )
        
        submitted = st.form_submit_button("Add Event(s)", type="primary")
        
        if submitted:
            if not title:
                st.error("Please provide an event title")
            elif is_recurring:
                # Validate recurring event
                selected_days = [day for day, selected in days_selected.items() if selected]
                if not selected_days:
                    st.error("⚠️ Please select at least one day of the week")
                elif recur_end_date < recur_start_date:
                    st.error("⚠️ End date must be after start date")
                else:
                    # Convert times
                    start_hour_24 = start_hour if start_period == "AM" and start_hour != 12 else \
                                   0 if start_period == "AM" and start_hour == 12 else \
                                   start_hour if start_period == "PM" and start_hour == 12 else \
                                   start_hour + 12
                    
                    end_hour_24 = end_hour if end_period == "AM" and end_hour != 12 else \
                                 0 if end_period == "AM" and end_hour == 12 else \
                                 end_hour if end_period == "PM" and end_hour == 12 else \
                                 end_hour + 12
                    
                    # Create recurring events
                    events_created = 0
                    current_date = recur_start_date
                    checklist = [item.strip() for item in checklist_input.split('\n') if item.strip()] if is_travel else None
                    
                    while current_date <= recur_end_date:
                        # Check if this day is in selected days
                        weekday_name = day_names[current_date.weekday()]
                        if weekday_name in selected_days:
                            start_datetime = datetime.combine(current_date, datetime.min.time()).replace(hour=start_hour_24, minute=start_minute)
                            end_datetime = datetime.combine(current_date, datetime.min.time()).replace(hour=end_hour_24, minute=end_minute)
                            
                            add_event(
                                title=title,
                                category=category,
                                start=start_datetime,
                                end=end_datetime,
                                location=location,
                                notes=notes,
                                priority=priority,
                                is_travel=is_travel,
                                checklist_items=checklist
                            )
                            events_created += 1
                        
                        current_date += timedelta(days=1)
                    
                    st.success(f"✅ Created {events_created} recurring events for: {', '.join(selected_days)}")
                    st.balloons()
                    st.rerun()
            else:
                # Single event
                # Convert 12-hour format to 24-hour format
                start_hour_24 = start_hour if start_period == "AM" and start_hour != 12 else \
                               0 if start_period == "AM" and start_hour == 12 else \
                               start_hour if start_period == "PM" and start_hour == 12 else \
                               start_hour + 12
                
                end_hour_24 = end_hour if end_period == "AM" and end_hour != 12 else \
                             0 if end_period == "AM" and end_hour == 12 else \
                             end_hour if end_period == "PM" and end_hour == 12 else \
                             end_hour + 12
                
                start_datetime = datetime.combine(start_date, datetime.min.time()).replace(hour=start_hour_24, minute=start_minute)
                end_datetime = datetime.combine(end_date, datetime.min.time()).replace(hour=end_hour_24, minute=end_minute)
                
                if end_datetime <= start_datetime and not time_tba and not single_time:
                    st.error("⚠️ End time must be after start time")
                else:
                    checklist = [item.strip() for item in checklist_input.split('\n') if item.strip()] if is_travel else None
                    
                    add_event(
                        title=title,
                        category=category,
                        start=start_datetime,
                        end=end_datetime,
                        location=location,
                        notes=notes,
                        priority=priority,
                        is_travel=is_travel,
                        checklist_items=checklist,
                        time_tba=time_tba,
                        single_time=single_time
                    )
                    
                    if time_tba:
                        st.success(f"✅ Event '{title}' added with time TBA!")
                    elif single_time:
                        st.success(f"✅ Deadline '{title}' added for {start_datetime.strftime('%I:%M %p')}!")
                    else:
                        st.success(f"✅ Event '{title}' added successfully!")
                    st.rerun()

# Main App
def main():
    # Stoic quotes
    stoic_quotes = [
        ("You have power over your mind - not outside events. Realize this, and you will find strength.", "Marcus Aurelius"),
        ("He who is brave is free.", "Seneca"),
        ("It's not what happens to you, but how you react to it that matters.", "Epictetus"),
        ("The best revenge is to be unlike him who performed the injury.", "Marcus Aurelius"),
        ("Waste no more time arguing what a good man should be. Be one.", "Marcus Aurelius"),
        ("We suffer more often in imagination than in reality.", "Seneca"),
        ("If it is not right, do not do it; if it is not true, do not say it.", "Marcus Aurelius"),
        ("The impediment to action advances action. What stands in the way becomes the way.", "Marcus Aurelius"),
        ("He who fears death will never do anything worth of a man who is alive.", "Seneca"),
        ("First say to yourself what you would be; and then do what you have to do.", "Epictetus"),
        ("Don't explain your philosophy. Embody it.", "Epictetus"),
        ("The happiness of your life depends upon the quality of your thoughts.", "Marcus Aurelius"),
        ("Luck is what happens when preparation meets opportunity.", "Seneca"),
        ("Difficulties strengthen the mind, as labor does the body.", "Seneca"),
        ("No person has the power to have everything they want, but it is in their power not to want what they don't have.", "Seneca")
    ]
    
    # Select quote based on day of year (same quote all day)
    day_of_year = datetime.now().timetuple().tm_yday
    quote, author = stoic_quotes[day_of_year % len(stoic_quotes)]
    
    # Display stoic quote at the very top
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {LIGHT_BEIGE} 0%, {SAND_LIGHT} 100%); 
        border-left: 3px solid {WOOD_DARK};
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 20px;
    '>
        <p style='
            margin: 0;
            font-style: italic;
            color: {WOOD_MED};
            font-size: 0.95em;
            line-height: 1.4;
        '>"{quote}"</p>
        <p style='
            margin: 8px 0 0 0;
            text-align: right;
            color: {WOOD_DARK};
            font-size: 0.85em;
            font-weight: 500;
        '>— {author}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize view date if not set
    if 'view_date' not in st.session_state:
        st.session_state.view_date = datetime.now()
    
    # Determine what date to display based on current view
    current_view = st.session_state.get('selected_view', 'Dashboard')
    
    if current_view == "Dashboard":
        date_display = "Next 48 Hours"
    elif current_view == "Day":
        date_display = st.session_state.view_date.strftime("%A, %B %d, %Y")
    elif current_view == "Add Event":
        date_display = "Add New Event"
    else:  # Week, Month
        date_display = st.session_state.view_date.strftime("%B %Y")
    
    # Display date prominently at top
    st.markdown(f"<h2 style='color: {WOOD_DARK}; font-weight: 600; margin-bottom: 20px;'>{date_display}</h2>", unsafe_allow_html=True)
    
    # Simple, clean navigation - VIEW TYPES ONLY
    nav_options = [
        "Dashboard",
        "Day",
        "Week", 
        "Month",
        "Add Event"
    ]
    
    # Initialize selected view in session state
    if 'selected_view' not in st.session_state:
        st.session_state.selected_view = "Dashboard"
    
    # Initialize event filter in session state
    if 'event_filter' not in st.session_state:
        st.session_state.event_filter = "All"
    
    # Create clean navigation bar with proper spacing
    nav_cols = st.columns([1.2, 0.7, 0.8, 0.8, 1.1])
    for idx, view_name in enumerate(nav_options):
        with nav_cols[idx]:
            is_active = st.session_state.selected_view == view_name
            if st.button(
                view_name,
                key=f"nav_{view_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.selected_view = view_name
                st.rerun()
    
    view = st.session_state.selected_view
    
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    # Sidebar with mini calendar and filters
    render_mini_calendar_sidebar()
    
    st.sidebar.markdown("---")
    
    # EVENT FILTERS
    st.sidebar.markdown("### Filters")
    
    # Show active filter
    current_filter = st.session_state.event_filter
    if current_filter != "All":
        st.sidebar.markdown(f"<p style='font-size: 0.85em; color: {WOOD_MED}; margin-bottom: 10px;'>Showing: <strong>{current_filter}</strong> events</p>", unsafe_allow_html=True)
    
    filter_options = ["All", "Baseball", "Academic", "Analytics"]
    
    for filter_name in filter_options:
        is_active = st.session_state.event_filter == filter_name
        if st.sidebar.button(
            filter_name,
            key=f"filter_{filter_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.event_filter = filter_name
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Stats
    st.sidebar.markdown("### Quick Stats")
    total_events = len(st.session_state.events)
    upcoming_events = len([e for e in st.session_state.events if e['start'] > datetime.now()])
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <p style="margin: 0; color: {DARK_BROWN};">
            <strong>{total_events}</strong> Total Events<br>
            <strong>{upcoming_events}</strong> Upcoming
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Data backup section
    st.sidebar.markdown("### Data Backup")
    st.sidebar.markdown("""
    <p style='font-size: 0.85em; color: #6F4E37;'>
    Auto-saved to <code>calendar_data.json</code>
    </p>
    """, unsafe_allow_html=True)
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data_json = f.read()
        st.sidebar.download_button(
            label="Download Backup",
            data=data_json,
            file_name=f"calendar_backup_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            help="Download a backup of your calendar data"
        )
    
    uploaded_file = st.sidebar.file_uploader(
        "Restore Backup",
        type=['json'],
        help="Upload a previously saved backup file"
    )
    
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            # Convert string dates back to datetime
            for event in data:
                event['start'] = datetime.fromisoformat(event['start'])
                event['end'] = datetime.fromisoformat(event['end'])
            st.session_state.events = data
            save_events()
            st.sidebar.success("Backup restored!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error restoring backup: {str(e)}")
    
    # Render selected view with applied filter
    if view == "Dashboard":
        render_next_48_hours()
    elif view == "Day":
        render_day_view()
    elif view == "Week":
        render_week_view()
    elif view == "Month":
        render_month_view()
    elif view == "Add Event":
        render_add_event_form()

if __name__ == "__main__":
    main()
