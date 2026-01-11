import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
import calendar as cal

# Page config
st.set_page_config(
    page_title="Tyler's Life Coordinator",
    page_icon="📅",
    layout="wide"
)

# Nebraska colors
NEBRASKA_RED = "#E41C38"
NEBRASKA_CREAM = "#F7F7F7"
DARK_BG = "#1a1a1a"

# Category colors
CATEGORY_COLORS = {
    "Classes": "#4A90E2",
    "Baseball - Practice": "#E41C38",
    "Baseball - Game": "#DC143C",
    "Baseball - Travel": "#8B0000",
    "Analytics Work": "#50C878",
    "Study Session": "#9B59B6",
    "Personal": "#F39C12",
    "Exam": "#E74C3C",
    "Assignment": "#3498DB"
}

# File for data persistence
DATA_FILE = "/home/claude/calendar_data.json"

# Custom CSS
st.markdown(f"""
<style>
    .main {{
        background-color: #0F0F0F;
    }}
    .stButton>button {{
        background-color: {NEBRASKA_RED};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: #C41230;
    }}
    .metric-card {{
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid {NEBRASKA_RED};
        margin: 10px 0;
    }}
    .event-card {{
        background: #1a1a1a;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 10px 0;
        color: white;
    }}
    .category-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
        margin-right: 8px;
    }}
    h1, h2, h3 {{
        color: {NEBRASKA_RED};
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
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            # Convert string dates back to datetime
            for event in data:
                event['start'] = datetime.fromisoformat(event['start'])
                event['end'] = datetime.fromisoformat(event['end'])
            return data
    return []

def save_events():
    """Save events to JSON file"""
    # Convert datetime to string for JSON serialization
    events_to_save = []
    for event in st.session_state.events:
        event_copy = event.copy()
        event_copy['start'] = event['start'].isoformat()
        event_copy['end'] = event['end'].isoformat()
        events_to_save.append(event_copy)
    
    with open(DATA_FILE, 'w') as f:
        json.dump(events_to_save, f, indent=2)

# Load events on startup
if not st.session_state.events:
    st.session_state.events = load_events()

def add_event(title, category, start, end, location="", notes="", priority="Medium", 
              is_travel=False, checklist_items=None):
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
        'completed_checklist': []
    }
    st.session_state.events.append(event)
    save_events()
    return event

def delete_event(event_id):
    """Delete an event"""
    st.session_state.events = [e for e in st.session_state.events if e['id'] != event_id]
    save_events()

def get_events_for_date(date):
    """Get all events for a specific date"""
    events = []
    for event in st.session_state.events:
        if event['start'].date() == date.date():
            events.append(event)
    return sorted(events, key=lambda x: x['start'])

def get_events_for_range(start_date, end_date):
    """Get all events within a date range"""
    events = []
    for event in st.session_state.events:
        if start_date.date() <= event['start'].date() <= end_date.date():
            events.append(event)
    return sorted(events, key=lambda x: x['start'])

def create_quick_event_templates():
    """Create quick-add templates for common events"""
    st.sidebar.markdown("### ⚡ Quick Add Templates")
    
    template = st.sidebar.selectbox(
        "Select Template",
        ["Custom", "Practice", "Home Game", "Away Game", "Class", "Study Session", "Exam"]
    )
    
    if template == "Practice":
        if st.sidebar.button("Add Practice"):
            today = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
            add_event(
                "Baseball Practice",
                "Baseball - Practice",
                today,
                today + timedelta(hours=3),
                location="Haymarket Park",
                notes="Regular practice session"
            )
            st.sidebar.success("Practice added!")
            st.rerun()
    
    elif template == "Home Game":
        if st.sidebar.button("Add Home Game"):
            today = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
            add_event(
                "Home Game",
                "Baseball - Game",
                today,
                today + timedelta(hours=3),
                location="Haymarket Park",
                priority="High"
            )
            st.sidebar.success("Home game added!")
            st.rerun()
    
    elif template == "Away Game":
        if st.sidebar.button("Add Away Game"):
            today = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)
            checklist = ["Pack analytics equipment", "Pack clothes", "Charge laptop", "Download data"]
            add_event(
                "Away Game",
                "Baseball - Game",
                today,
                today + timedelta(hours=3),
                priority="High",
                is_travel=True,
                checklist_items=checklist
            )
            st.sidebar.success("Away game added!")
            st.rerun()

def render_event_card(event):
    """Render a styled event card"""
    color = CATEGORY_COLORS.get(event['category'], "#888888")
    
    duration = event['end'] - event['start']
    duration_str = f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"
    
    priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(event['priority'], "⚪")
    travel_badge = "✈️ TRAVEL" if event.get('is_travel') else ""
    
    st.markdown(f"""
    <div class="event-card" style="border-left-color: {color};">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h3 style="margin: 0; color: white;">{priority_emoji} {event['title']} {travel_badge}</h3>
                <p style="margin: 5px 0; color: #aaa;">
                    <span class="category-badge" style="background-color: {color};">{event['category']}</span>
                </p>
                <p style="margin: 5px 0; color: #ccc;">
                    📅 {event['start'].strftime('%b %d, %Y')} | 
                    🕐 {event['start'].strftime('%I:%M %p')} - {event['end'].strftime('%I:%M %p')} 
                    ({duration_str})
                </p>
                {f"<p style='margin: 5px 0; color: #ccc;'>📍 {event['location']}</p>" if event.get('location') else ""}
                {f"<p style='margin: 5px 0; color: #ddd;'>📝 {event['notes']}</p>" if event.get('notes') else ""}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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
        colorscale=[[0, '#1a1a1a'], [0.5, '#E41C38'], [1, '#8B0000']],
        showscale=True,
        hovertemplate='%{y}<br>%{x}<br>Events: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Workload Heatmap - Next 7 Days",
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        height=400,
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#0F0F0F',
        font=dict(color='white'),
        title_font=dict(color=NEBRASKA_RED)
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
        paper_bgcolor='#0F0F0F',
        plot_bgcolor='#0F0F0F',
        font=dict(color='white'),
        title_font=dict(color=NEBRASKA_RED),
        showlegend=True,
        legend=dict(font=dict(color='white'))
    )
    
    return fig

def render_next_48_hours():
    """Render quick dashboard for next 48 hours"""
    st.markdown("## ⚡ Next 48 Hours")
    
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
            <h3 style="margin: 0; color: white;">{len(events)}</h3>
            <p style="margin: 5px 0; color: #aaa;">Total Events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        travel_events = sum(1 for e in events if e.get('is_travel'))
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: white;">{travel_events}</h3>
            <p style="margin: 5px 0; color: #aaa;">Travel Required</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_priority = sum(1 for e in events if e.get('priority') == 'High')
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: white;">{high_priority}</h3>
            <p style="margin: 5px 0; color: #aaa;">High Priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    for event in events:
        time_until = event['start'] - now
        hours_until = time_until.total_seconds() / 3600
        
        if hours_until < 0:
            time_str = "🔴 IN PROGRESS"
        elif hours_until < 2:
            time_str = f"🔴 in {int(hours_until * 60)} minutes"
        elif hours_until < 6:
            time_str = f"🟡 in {hours_until:.1f} hours"
        else:
            time_str = f"🟢 in {hours_until:.1f} hours"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            render_event_card(event)
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h4 style="margin: 0; color: white;">{time_str}</h4>
            </div>
            """, unsafe_allow_html=True)

def render_day_view():
    """Render single day view"""
    st.markdown("## 📅 Day View")
    
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
    
    st.markdown(f"### {st.session_state.view_date.strftime('%A, %B %d, %Y')}")
    
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
    """Render week view"""
    st.markdown("## 📅 Week View")
    
    # Week navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ Previous Week"):
            st.session_state.view_date -= timedelta(days=7)
            st.rerun()
    
    with col2:
        week_start = st.session_state.view_date - timedelta(days=st.session_state.view_date.weekday())
        week_end = week_start + timedelta(days=6)
        st.markdown(f"### {week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}")
    
    with col3:
        if st.button("Next Week ▶"):
            st.session_state.view_date += timedelta(days=7)
            st.rerun()
    
    # Display week
    week_start = st.session_state.view_date - timedelta(days=st.session_state.view_date.weekday())
    
    cols = st.columns(7)
    
    for i in range(7):
        day = week_start + timedelta(days=i)
        events = get_events_for_date(day)
        
        with cols[i]:
            is_today = day.date() == datetime.now().date()
            day_style = f"background-color: {NEBRASKA_RED};" if is_today else ""
            
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; {day_style} border-radius: 5px;">
                <strong>{day.strftime('%a')}</strong><br>
                {day.strftime('%m/%d')}
            </div>
            """, unsafe_allow_html=True)
            
            if events:
                st.markdown(f"**{len(events)} events**")
                for event in events:
                    color = CATEGORY_COLORS.get(event['category'], "#888888")
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 5px; margin: 5px 0; border-radius: 3px; font-size: 0.8em;">
                        {event['start'].strftime('%I:%M %p')}<br>
                        <strong>{event['title']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("_No events_")

def render_month_view():
    """Render month view"""
    st.markdown("## 📅 Month View")
    
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
        st.markdown(f"### {st.session_state.view_date.strftime('%B %Y')}")
    
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
    
    # Display calendar
    cols = st.columns(7)
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for i, day in enumerate(days):
        cols[i].markdown(f"**{day}**")
    
    for week in month_cal:
        cols = st.columns(7)
        for i, day in enumerate(week):
            if day == 0:
                cols[i].markdown("")
            else:
                date = datetime(year, month, day)
                events = get_events_for_date(date)
                
                is_today = date.date() == datetime.now().date()
                style = f"background-color: {NEBRASKA_RED}; color: white;" if is_today else "background-color: #1a1a1a;"
                
                with cols[i]:
                    st.markdown(f"""
                    <div style="{style} padding: 10px; border-radius: 5px; min-height: 100px;">
                        <div style="text-align: center; font-weight: bold;">{day}</div>
                        <div style="font-size: 0.75em; margin-top: 5px;">
                            {f"{len(events)} event{'s' if len(events) != 1 else ''}" if events else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def render_baseball_season_tracker():
    """Track baseball season progress"""
    st.markdown("## ⚾ Baseball Season Tracker")
    
    # Get baseball events
    baseball_events = [e for e in st.session_state.events if 'Baseball' in e['category']]
    games = [e for e in baseball_events if e['category'] == 'Baseball - Game']
    practices = [e for e in baseball_events if e['category'] == 'Baseball - Practice']
    
    # Season stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; color: white;">{len(games)}</h2>
            <p style="margin: 5px 0; color: #aaa;">Games Scheduled</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; color: white;">{len(practices)}</h2>
            <p style="margin: 5px 0; color: #aaa;">Practices Scheduled</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        away_games = sum(1 for g in games if g.get('is_travel'))
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; color: white;">{away_games}</h2>
            <p style="margin: 5px 0; color: #aaa;">Away Games</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        upcoming_games = [g for g in games if g['start'] > datetime.now()]
        next_game = min(upcoming_games, key=lambda x: x['start']) if upcoming_games else None
        
        if next_game:
            days_until = (next_game['start'].date() - datetime.now().date()).days
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0; color: white;">{days_until}</h2>
                <p style="margin: 5px 0; color: #aaa;">Days to Next Game</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0; color: white;">-</h2>
                <p style="margin: 5px 0; color: #aaa;">No Games Scheduled</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Upcoming games
    if upcoming_games:
        st.markdown("### Upcoming Games")
        for game in upcoming_games[:5]:
            render_event_card(game)

def render_academic_tracker():
    """Track academic deadlines and exams"""
    st.markdown("## 🎓 Academic Tracker")
    
    # Get academic events
    exams = [e for e in st.session_state.events if e['category'] == 'Exam']
    assignments = [e for e in st.session_state.events if e['category'] == 'Assignment']
    classes = [e for e in st.session_state.events if e['category'] == 'Classes']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        upcoming_exams = [e for e in exams if e['start'] > datetime.now()]
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; color: white;">{len(upcoming_exams)}</h2>
            <p style="margin: 5px 0; color: #aaa;">Upcoming Exams</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        upcoming_assignments = [e for e in assignments if e['start'] > datetime.now()]
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; color: white;">{len(upcoming_assignments)}</h2>
            <p style="margin: 5px 0; color: #aaa;">Pending Assignments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        next_exam = min(upcoming_exams, key=lambda x: x['start']) if upcoming_exams else None
        if next_exam:
            days_until = (next_exam['start'].date() - datetime.now().date()).days
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0; color: white;">{days_until}</h2>
                <p style="margin: 5px 0; color: #aaa;">Days to Next Exam</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin: 0; color: white;">-</h2>
                <p style="margin: 5px 0; color: #aaa;">No Exams Scheduled</p>
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
    st.markdown("## 📊 Time Management Analytics")
    
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
        st.markdown("### 🆓 Free Time Windows - Next 7 Days")
        
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
                    <p style="margin: 0; color: white;">
                        <strong>{block['start'].strftime('%a %m/%d')}</strong><br>
                        {block['start'].strftime('%I:%M %p')} - {block['end'].strftime('%I:%M %p')}<br>
                        <span style="color: {NEBRASKA_RED};">{block['duration']:.1f} hours free</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant free time blocks found. Consider optimizing your schedule!")

def render_add_event_form():
    """Form to add new events"""
    st.markdown("## ➕ Add New Event")
    
    with st.form("add_event_form"):
        title = st.text_input("Event Title*")
        
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category*", list(CATEGORY_COLORS.keys()))
        with col2:
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date*")
            start_time = st.time_input("Start Time*")
        with col2:
            end_date = st.date_input("End Date*")
            end_time = st.time_input("End Time*")
        
        location = st.text_input("Location")
        notes = st.text_area("Notes")
        
        is_travel = st.checkbox("Requires Travel")
        
        checklist_input = ""
        if is_travel:
            checklist_input = st.text_area(
                "Travel Checklist (one item per line)",
                value="Pack analytics equipment\nPack clothes\nCharge laptop\nDownload data"
            )
        
        submitted = st.form_submit_button("Add Event", type="primary")
        
        if submitted:
            if not title:
                st.error("Please provide an event title")
            else:
                start_datetime = datetime.combine(start_date, start_time)
                end_datetime = datetime.combine(end_date, end_time)
                
                if end_datetime <= start_datetime:
                    st.error("End time must be after start time")
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
                        checklist_items=checklist
                    )
                    
                    st.success(f"✅ Event '{title}' added successfully!")
                    st.rerun()

# Main App
def main():
    st.title("📅 Tyler's Life Coordinator")
    st.markdown(f"*Manage your final semester like a champion*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    view = st.sidebar.radio(
        "Select View",
        ["Dashboard (48hrs)", "Day View", "Week View", "Month View", 
         "Baseball Season", "Academic Tracker", "Analytics", "Add Event"]
    )
    
    st.sidebar.markdown("---")
    
    # Quick add templates
    create_quick_event_templates()
    
    st.sidebar.markdown("---")
    
    # Stats
    st.sidebar.markdown("### 📊 Quick Stats")
    total_events = len(st.session_state.events)
    upcoming_events = len([e for e in st.session_state.events if e['start'] > datetime.now()])
    
    st.sidebar.markdown(f"""
    <div class="metric-card">
        <p style="margin: 0; color: white;">
            <strong>{total_events}</strong> Total Events<br>
            <strong>{upcoming_events}</strong> Upcoming
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render selected view
    if view == "Dashboard (48hrs)":
        render_next_48_hours()
    elif view == "Day View":
        render_day_view()
    elif view == "Week View":
        render_week_view()
    elif view == "Month View":
        render_month_view()
    elif view == "Baseball Season":
        render_baseball_season_tracker()
    elif view == "Academic Tracker":
        render_academic_tracker()
    elif view == "Analytics":
        render_analytics_dashboard()
    elif view == "Add Event":
        render_add_event_form()

if __name__ == "__main__":
    main()
