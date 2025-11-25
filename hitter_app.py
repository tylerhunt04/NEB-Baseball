# hitter_app.py

import os
import re
import glob
import math
import base64
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from matplotlib.patches import Rectangle, Wedge, Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from matplotlib import colors

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nebraska Baseball — Hitter Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Color scheme
HUSKER_RED = "#E60026"
DARK_GRAY = "#2B2B2B"
LIGHT_GRAY = "#F5F5F5"
HUSKER_CREAM = "#FEFDFA"

# ──────────────────────────────────────────────────────────────────────────────
# PROFESSIONAL UI STYLING
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Global Styles */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}
    
    /* Section Headers */
    .section-header {{
        background: linear-gradient(135deg, {HUSKER_RED} 0%, #B8001F 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: 700;
        margin: 25px 0 15px 0;
        box-shadow: 0 2px 8px rgba(230, 0, 38, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .subsection-header {{
        color: {DARK_GRAY};
        font-size: 18px;
        font-weight: 600;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 3px solid {HUSKER_RED};
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, white 0%, {LIGHT_GRAY} 100%);
        border-left: 4px solid {HUSKER_RED};
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}
    
    .metric-label {{
        color: #666;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }}
    
    .metric-value {{
        color: {DARK_GRAY};
        font-size: 28px;
        font-weight: 700;
        margin: 5px 0;
    }}
    
    .metric-sublabel {{
        color: #888;
        font-size: 11px;
        margin-top: 5px;
    }}
    
    /* Filter Section */
    .filter-section {{
        background-color: {LIGHT_GRAY};
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #E0E0E0;
    }}
    
    /* Info/Warning Boxes */
    .info-box {{
        background-color: #F0F7FF;
        border-left: 4px solid #1E88E5;
        border-radius: 6px;
        padding: 15px 20px;
        margin: 15px 0;
        color: #1565C0;
    }}
    
    .warning-box {{
        background-color: #FFF8E1;
        border-left: 4px solid #FFA726;
        border-radius: 6px;
        padding: 15px 20px;
        margin: 15px 0;
        color: #EF6C00;
    }}
    
    /* Tables */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    
    .dataframe thead tr th {{
        background: linear-gradient(135deg, {HUSKER_RED} 0%, #B8001F 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
        text-align: left !important;
        border: none !important;
    }}
    
    .dataframe tbody tr:hover {{
        background-color: #F8F9FA !important;
    }}
    
    .dataframe tbody td {{
        padding: 10px !important;
        border-bottom: 1px solid #E0E0E0 !important;
    }}
    
    /* Buttons */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, {HUSKER_RED} 0%, #B8001F 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 2px 6px rgba(230, 0, 38, 0.2);
    }}
    
    .stDownloadButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(230, 0, 38, 0.3);
    }}
    
    /* Radio Buttons */
    .stRadio > div {{
        background-color: {LIGHT_GRAY};
        border-radius: 8px;
        padding: 10px;
    }}
    
    /* Selectbox */
    .stSelectbox > div > div {{
        border-radius: 6px;
        border: 2px solid #E0E0E0;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {LIGHT_GRAY};
        border-radius: 8px;
        padding: 5px;
        gap: 5px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        border-radius: 6px;
        padding: 10px 20px;
        border: 1px solid #E0E0E0;
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {HUSKER_RED} 0%, #B8001F 100%);
        color: white;
        border: none;
    }}
    
    /* Divider */
    .professional-divider {{
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, {HUSKER_RED} 50%, transparent 100%);
        margin: 30px 0;
        border: none;
    }}
    
    /* Caption Text */
    .caption-text {{
        color: #666;
        font-size: 13px;
        font-style: italic;
        margin: 8px 0;
    }}
    
    /* Player Header Card */
    .player-header {{
        background: linear-gradient(135deg, {HUSKER_RED} 0%, #B8001F 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(230, 0, 38, 0.3);
    }}
    
    .player-name {{
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }}
    
    .player-subtitle {{
        font-size: 16px;
        opacity: 0.9;
    }}
    
    /* Color Guide Box */
    .color-guide {{
        background-color: {LIGHT_GRAY};
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 4px solid {HUSKER_RED};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        font-weight: 600;
    }}
    
    .streamlit-expanderHeader:hover {{
        border-color: {HUSKER_RED};
    }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# UI HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def section_header(text: str):
    """Display a major section header with Nebraska branding"""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def subsection_header(text: str):
    """Display a subsection header"""
    st.markdown(f'<div class="subsection-header">{text}</div>', unsafe_allow_html=True)

def metric_card(label: str, value: str, sublabel: str = ""):
    """Display a metric in a styled card"""
    sublabel_html = f'<div class="metric-sublabel">{sublabel}</div>' if sublabel else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sublabel_html}
    </div>
    """, unsafe_allow_html=True)

def info_message(text: str):
    """Display an info message box"""
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)

def warning_message(text: str):
    """Display a warning message box"""
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)

def professional_divider():
    """Display a professional divider line"""
    st.markdown('<hr class="professional-divider">', unsafe_allow_html=True)

def themed_table(df: pd.DataFrame):
    """Apply themed styling to a dataframe for display"""
    return df

# Default data paths per period
DATA_PATH_2025   = "B10C25_hitter_app_columns.csv"
DATA_PATH_SCRIM  = "Scrimmage(27).csv"
DATA_PATH_2026   = "B10C26_hitter_app_columns.csv"

PROB_LOOKUP_PATH = "EV_LA_probabilities.csv"

BANNER_CANDIDATES = [
    "NebraskaChampions.jpg",
    "/mnt/data/NebraskaChampions.jpg",
]

# Big Ten / opponents pretty names
TEAM_NAME_MAP = {
    "ILL_ILL": "Illinois",
    "MIC_SPA": "Michigan State",
    "UCLA": "UCLA",
    "IOWA_HAW": "Iowa",
    "IU": "Indiana",
    "MAR_TER": "Maryland",
    "MIC_WOL": "Michigan",
    "MIN_GOL": "Minnesota",
    "NEB": "Nebraska",
    "NOR_CAT": "Northwestern",
    "ORE_DUC": "Oregon",
    "OSU_BUC": "Ohio State",
    "PEN_NIT": "Penn State",
    "PUR_BOI": "Purdue",
    "RUT_SCA": "Rutgers",
    "SOU_TRO": "USC",
    "WAS_HUS": "Washington",
    "WIC_SHO": "Wichita State",
}

# wOBA CONSTANTS (FanGraphs 2025)
WOBAC_2025 = {
    "wBB": 0.693, "wHBP": 0.723,
    "w1B": 0.883, "w2B": 1.253, "w3B": 1.585, "wHR": 2.037,
    "wOBAScale": 1.23,
    "lg_wOBA": 0.314,
}

# D1 AVERAGES FOR COMPARISON
D1_AVERAGES = {
    "ZWhiff%": 15.7,
    "ZContact%": 84.3,  # Derived from 100 - ZWhiff%
    "Whiff%": 22.9,
    "Barrel%": 17.4,
    "wOBA": 0.364,
    "K%": 19.2,
    "BB%": 11.4,
    "Chase%": 24.2,
    "AVG": 0.280,
    "OBP": 0.385,
    "SLG": 0.442,
    "OPS": 0.827,
    "HardHit%": 36.0,
    "Avg Exit Velocity": 86.2,
    "Max Exit Velocity": 103.1,
}

def get_performance_color_gradient(stat_name: str, value: float) -> str:
    """
    Return a background color based on how the value compares to D1 average.
    Returns a hex color - greener for better, redder for worse, white for average
    """
    if pd.isna(value) or stat_name not in D1_AVERAGES:
        return "#ffffff"  # white for no comparison
    
    avg = D1_AVERAGES[stat_name]
    
    # For stats where LOWER is better (strikeouts, whiffs, chase)
    if stat_name in ["K%", "Whiff%", "Chase%", "ZWhiff%"]:
        diff_pct = (avg - value) / avg  # Positive if player is better (lower)
    # For stats where HIGHER is better (including HardHit%, exit velocities, ZContact%)
    else:
        diff_pct = (value - avg) / avg  # Positive if player is better (higher)
    
    # Create gradient based on difference
    if diff_pct >= 0.20:  # 20%+ better - darkest green
        return "#28a745"
    elif diff_pct >= 0.10:  # 10-20% better - medium green
        return "#5cb85c"
    elif diff_pct >= 0.05:  # 5-10% better - light green
        return "#90ee90"
    elif diff_pct >= -0.05:  # Within 5% - white/neutral
        return "#ffffff"
    elif diff_pct >= -0.10:  # 5-10% worse - light red
        return "#ffb3b3"
    elif diff_pct >= -0.20:  # 10-20% worse - medium red
        return "#ff8080"
    else:  # 20%+ worse - darkest red
        return "#dc3545"

def style_performance_table(df: pd.DataFrame, stat_name_col='Metric') -> pd.DataFrame:
    """Apply color gradient styling to performance tables based on D1 averages."""
    def apply_color(row):
        stat_name = row[stat_name_col]
        value_str = row['Value']
        
        # Map display names to D1_AVERAGES keys
        stat_mapping = {
            'Avg Exit Velocity': 'Avg Exit Velocity',
            'Max Exit Velocity': 'Max Exit Velocity',
            'Hard Hit%': 'HardHit%',
            'Barrel%': 'Barrel%',
            'AVG': 'AVG',
            'OBP': 'OBP',
            'SLG': 'SLG',
            'OPS': 'OPS',
            'wOBA': 'wOBA',
            'K%': 'K%',
            'BB%': 'BB%',
            'Swing%': 'Swing%',
            'Whiff%': 'Whiff%',
            'Chase%': 'Chase%',
            'Z-Swing%': 'ZSwing%',
            'Z-Contact%': 'ZContact%',
            'Z-Whiff%': 'ZWhiff%'
        }
        
        # Get the correct key for D1_AVERAGES lookup
        d1_key = stat_mapping.get(stat_name, stat_name)
        
        # Extract numeric value from string
        try:
            if '%' in value_str:
                value = float(value_str.replace('%', ''))
            elif 'mph' in value_str:
                value = float(value_str.replace('mph', '').strip())
            elif '°' in value_str:
                value = float(value_str.replace('°', '').strip())
            elif value_str == "—":
                return [''] * len(row)
            else:
                value = float(value_str)
        except:
            return [''] * len(row)
        
        bg_color = get_performance_color_gradient(d1_key, value)
        
        # Make text white if background is dark
        if bg_color in ["#28a745", "#dc3545"]:
            text_color = "white"
        else:
            text_color = "black"
        
        return [f'background-color: {bg_color}; color: {text_color}' if col == 'Value' else '' 
                for col in row.index]
    
    styled = df.style.apply(apply_color, axis=1)
    
    # Apply Husker Red header
    header_props = f'background-color: {HUSKER_RED}; color: white; white-space: nowrap;'
    styled = styled.set_table_styles([
        {'selector': 'thead th', 'props': header_props},
        {'selector': 'th.col_heading', 'props': header_props},
        {'selector': 'th', 'props': header_props},
    ]).hide(axis="index")
    
    return styled

# ──────────────────────────────────────────────────────────────────────────────
# DATE & NAME HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"

def format_date_long(d) -> str:
    if d is None or pd.isna(d):
        return ""
    d = pd.to_datetime(d).date()
    return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"

MONTH_CHOICES = [
    (1,"January"), (2,"February"), (3,"March"), (4,"April"),
    (5,"May"), (6,"June"), (7,"July"), (8,"August"),
    (9,"September"), (10,"October"), (11,"November"), (12,"December")
]
MONTH_NAME_BY_NUM = {n: name for n, name in MONTH_CHOICES}

def normalize_name(name: str) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ", ", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    parts = [p.title() for p in parts]
    return ", ".join(parts)

def parse_date_robust(df: pd.DataFrame) -> pd.Series:
    cand_cols = []
    lower = {c.lower(): c for c in df.columns}
    for name in ["date", "gamedate", "game date", "datetime", "game_datetime", "gamedatetime", "datelocal", "game_date"]:
        if name in lower:
            cand_cols.append(lower[name])

    series = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    def _try_parse(s: pd.Series):
        out = pd.to_datetime(s, errors="coerce", utc=False, infer_datetime_format=True)
        if out.isna().mean() > 0.8:
            out2 = pd.to_datetime(s, errors="coerce", utc=False, dayfirst=True)
            if out2.isna().mean() < out.isna().mean():
                out = out2
        return out

    for c in cand_cols:
        parsed = _try_parse(df[c])
        series = series.fillna(parsed)

    if "GameID" in df.columns:
        gid = df["GameID"].astype(str)
        ymd = gid.str.extract(r"(20\d{6})", expand=False)
        if not ymd.dropna().empty:
            parsed_gid = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
            series = series.fillna(parsed_gid)

    series = pd.to_datetime(series.dt.date, errors="coerce")
    return series

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = parse_date_robust(df)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# STRIKE ZONE / VIEW / COLORS
# ──────────────────────────────────────────────────────────────────────────────
def draw_strikezone(ax, left=-0.83, right=0.83, bottom=1.5, top=3.5):
    ax.add_patch(Rectangle((left, bottom), right-left, top-bottom,
                           fill=False, linewidth=2, color='black'))
    dx, dy = (right-left)/3, (top-bottom)/3
    for i in (1, 2):
        ax.add_line(Line2D([left+i*dx]*2, [bottom, top], linestyle='--', color='gray', linewidth=1))
        ax.add_line(Line2D([left, right], [bottom+i*dy]*2, linestyle='--', color='gray', linewidth=1))

X_LIM = (-3, 3)
Y_LIM = (0, 5)

custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
    N=256,
)

# ──────────────────────────────────────────────────────────────────────────────
# SPRAY CHART HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def draw_dirt_diamond(
    ax,
    origin=(0.0, 0.0),
    size: float = 80,
    base_size: float = 8,
    grass_scale: float = 0.4,
    base_offset_scale: float = 1.1,
    outfield_scale: float = 3.0,
    path_width: float = 8,
    foul_line_extend: float = 1.1,
    arc_extend_scale: float = 1.7,
    custom_outfield_radius: float = None,
    custom_wall_distances: list = None
):
    home = np.array(origin)

    if custom_wall_distances is not None:
        angles = [item[0] for item in custom_wall_distances]
        distances = [item[1] for item in custom_wall_distances]
        
        outfield_points = []
        for angle, dist in zip(angles, distances):
            rad = math.radians(angle)
            x = home[0] + dist * math.cos(rad)
            y = home[1] + dist * math.sin(rad)
            outfield_points.append([x, y])
        
        outfield_points.append(home.tolist())
        ax.add_patch(Polygon(outfield_points, closed=True, facecolor='#228B22', edgecolor='black', linewidth=2))
        
        outfield_radius = max(distances)
        left_corner_dist = distances[0]
        right_corner_dist = distances[-1]
    elif custom_outfield_radius is not None:
        outfield_radius = custom_outfield_radius
        ax.add_patch(Wedge(home, outfield_radius, 45, 135, facecolor='#228B22', edgecolor='black', linewidth=2))
        left_corner_dist = outfield_radius
        right_corner_dist = outfield_radius
    else:
        outfield_radius = size * arc_extend_scale
        ax.add_patch(Wedge(home, outfield_radius, 45, 135, facecolor='#228B22', edgecolor='black', linewidth=2))
        left_corner_dist = outfield_radius
        right_corner_dist = outfield_radius
    
    ax.add_patch(Wedge(home, size, 45, 135, facecolor='#ED8B00', edgecolor='black', linewidth=2))
    
    for angle, corner_dist in [(45, left_corner_dist), (135, right_corner_dist)]:
        rad = math.radians(angle)
        end = home + np.array([corner_dist * math.cos(rad), corner_dist * math.sin(rad)])
        perp = np.array([-math.sin(rad), math.cos(rad)])
        off = perp * (path_width / 2)
        corners = [home + off, home - off, end - off, end + off]
        ax.add_patch(Polygon(corners, closed=True, facecolor='#ED8B00', edgecolor='black', linewidth=1))

    gsize = size * grass_scale
    gfirst = home + np.array((gsize, gsize))
    gsecond = home + np.array((0.0, 2 * gsize))
    gthird = home + np.array((-gsize, gsize))
    
    arc_angles = np.linspace(45, 135, 50)
    arc_radius = gsize * 1.8
    arc_points = []
    for angle in arc_angles:
        rad = math.radians(angle)
        x = home[0] + arc_radius * math.cos(rad)
        y = home[1] + arc_radius * math.sin(rad)
        arc_points.append([x, y])
    
    grass_polygon = [gfirst.tolist()] + arc_points + [gthird.tolist(), home.tolist()]
    ax.add_patch(Polygon(grass_polygon, closed=True, facecolor='#228B22', edgecolor='none'))
    
    for pos in [gfirst, gsecond, gthird]:
        ax.add_patch(Rectangle((pos[0] - base_size/2, pos[1] - base_size/2), base_size, base_size,
                               facecolor='white', edgecolor='black', linewidth=1))

    half = base_size / 2
    plate = Polygon([
        (home[0] - half, home[1]),
        (home[0] + half, home[1]),
        (home[0] + half * 0.6, home[1] - half * 0.8),
        (home[0], home[1] - base_size),
        (home[0] - half * 0.6, home[1] - half * 0.8)
    ], closed=True, facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(plate)

    for angle, corner_dist in [(45, left_corner_dist), (135, right_corner_dist)]:
        rad = math.radians(angle)
        end = home + np.array([corner_dist * foul_line_extend * math.cos(rad),
                               corner_dist * foul_line_extend * math.sin(rad)])
        ax.plot([home[0], end[0]], [home[1], end[1]], color='white', linewidth=2)

    ax.set_xlim(-outfield_radius, outfield_radius)
    ax.set_ylim(-base_size * 1.5, outfield_radius)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax

def bearing_distance_to_xy(bearing, distance):
    angle_rad = np.radians(90 - bearing)
    x = distance * np.cos(angle_rad)
    y = distance * np.sin(angle_rad)
    return x, y

# ──────────────────────────────────────────────────────────────────────────────
# DENSITY
# ──────────────────────────────────────────────────────────────────────────────
def compute_density_hitter(x, y, xi_m, yi_m):
    coords = np.vstack([x, y])
    mask = np.isfinite(coords).all(axis=0)
    if mask.sum() <= 1:
        return np.zeros(xi_m.shape)
    try:
        kde = gaussian_kde(coords[:, mask])
        return kde(np.vstack([xi_m.ravel(), yi_m.ravel()])).reshape(xi_m.shape)
    except Exception:
        return np.zeros(xi_m.shape)

# ──────────────────────────────────────────────────────────────────────────────
# FORMATTERS / TABLE STYLE
# ──────────────────────────────────────────────────────────────────────────────
def fmt_pct(x, decimals=1):
    try:
        if pd.isna(x): return "—"
        return f"{round(float(x), decimals)}%"
    except Exception:
        return "—"

def fmt_pct2(x):
    try:
        if pd.isna(x): return "—"
        return f"{round(float(x), 2)}%"
    except Exception:
        return "—"

def fmt_avg3(x):
    try:
        if pd.isna(x): return "—"
        val = float(x)
        out = f"{val:.3f}"
        return out[1:] if val < 1 else out
    except Exception:
        return "—"

def themed_styler(df: pd.DataFrame, nowrap=True):
    header_props = f'background-color: {HUSKER_RED}; color: white; white-space: nowrap;'
    styles = [
        {'selector': 'thead th', 'props': header_props},
        {'selector': 'th.col_heading', 'props': header_props},
        {'selector': 'th', 'props': header_props},
    ]
    if nowrap:
        styles.append({'selector': 'td', 'props': 'white-space: nowrap;'})
    return (df.style
            .hide(axis="index")
            .set_table_styles(styles))

# ──────────────────────────────────────────────────────────────────────────────
# BANNER
# ──────────────────────────────────────────────────────────────────────────────
def _img_to_b64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def render_nb_banner(image_candidates=BANNER_CANDIDATES, title="Nebraska Baseball", subtitle="Hitter Analytics Platform", height_px=280):
    b64 = None
    for p in image_candidates:
        b64 = _img_to_b64(p)
        if b64:
            break
    if not b64:
        return
    st.markdown(
        f"""
        <div style="position: relative; width: 100%; height: {height_px}px; border-radius: 12px; overflow: hidden; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.15);">
          <img src="data:image/jpeg;base64,{b64}" style="width:100%; height:100%; object-fit:cover;" />
          <div style="position:absolute; inset:0; background: linear-gradient(rgba(0,0,0,0.5), rgba(230,0,38,0.3));"></div>
          <div style="position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;">
            <div style="font-size:48px; font-weight:800; color:white; text-shadow: 0 2px 12px rgba(0,0,0,.9); text-transform:uppercase; letter-spacing:1px;">
              {title}
            </div>
            <div style="font-size:20px; font-weight:500; color:white; text-shadow: 0 2px 8px rgba(0,0,0,.7); margin-top:10px; letter-spacing:0.5px;">
              {subtitle}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# [Rest of the helper functions remain the same: batted ball, discipline, stats, barrel mask, xwOBA, pitch type helpers, split metrics, fall summary functions, profile tables, rankings, spray charts, hitter report, heatmaps, and loaders]

# ──────────────────────────────────────────────────────────────────────────────
# BATTED BALL / DISCIPLINE / STATS
# ──────────────────────────────────────────────────────────────────────────────
def assign_spray_category(row):
    ang  = row.get('Bearing', np.nan)
    side = str(row.get('BatterSide', "")).upper()[:1]
    if not np.isfinite(ang):
        return np.nan
    if -15 <= ang <= 15:
        return 'Straight'
    if ang < -15:
        return 'Pull' if side == 'R' else 'Opposite'
    return 'Opposite' if side == 'R' else 'Pull'

def create_batted_ball_profile(df: pd.DataFrame):
    inplay = df[df.get('PitchCall', pd.Series(dtype=object)) == 'InPlay'].copy()
    if 'TaggedHitType' not in inplay.columns:
        inplay['TaggedHitType'] = pd.NA
    if 'Bearing' not in inplay.columns:
        inplay['Bearing'] = np.nan
    if 'BatterSide' not in inplay.columns:
        inplay['BatterSide'] = ""

    inplay['spray_cat'] = inplay.apply(assign_spray_category, axis=1)

    def pct(mask):
        try:
            mask = pd.Series(mask).astype(bool)
            return round(100 * float(mask.mean()), 1) if len(mask) else 0.0
        except Exception:
            return 0.0

    bb = pd.DataFrame([{
        "LD%": pct(inplay["TaggedHitType"].astype(str).str.contains("LineDrive", case=False, na=False)),
        "GB%": pct(inplay["TaggedHitType"].astype(str).str.contains("GroundBall", case=False, na=False)),
        "FB%": pct(inplay["TaggedHitType"].astype(str).str.contains("FlyBall",   case=False, na=False)),
        "Pull%":    pct(inplay["spray_cat"].astype(str).eq("Pull")),
        "Middle%":  pct(inplay["spray_cat"].astype(str).eq("Straight")),
        "Oppo%":    pct(inplay["spray_cat"].astype(str).eq("Opposite")),
    }])
    return bb

def create_plate_discipline_profile(df: pd.DataFrame):
    s_call = df.get('PitchCall', pd.Series(dtype=object))
    lside  = pd.to_numeric(df.get('PlateLocSide', pd.Series(dtype=float)), errors="coerce")
    lht    = pd.to_numeric(df.get('PlateLocHeight', pd.Series(dtype=float)), errors="coerce")

    isswing   = s_call.isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
    iswhiff   = s_call.eq('StrikeSwinging')
    iscontact = s_call.isin(['InPlay','FoulBallNotFieldable','FoulBallFieldable'])
    isinzone  = lside.between(-0.83, 0.83) & lht.between(1.5, 3.5)

    zone_pitches = int(isinzone.sum()) if len(isinzone) else 0
    zone_pct     = round(isinzone.mean()*100, 1) if len(isinzone) else 0.0
    zone_sw      = round(isswing[isinzone].mean()*100, 1) if isinzone.sum() else 0.0
    zone_ct      = round((iscontact & isinzone).sum() / max(isswing[isinzone].sum(), 1) * 100, 1) if isinzone.sum() else 0.0
    chase        = round(isswing[~isinzone].mean()*100, 1) if (~isinzone).sum() else 0.0
    swing        = round(isswing.mean()*100, 1) if len(isswing) else 0.0
    whiff        = round(iswhiff.sum() / max(isswing.sum(), 1) * 100, 1) if len(isswing) else 0.0

    return pd.DataFrame([{
        "Zone Pitches": zone_pitches,
        "Zone %":       zone_pct,
        "Zone Swing %": zone_sw,
        "Zone Contact %": zone_ct,
        "Chase %":      chase,
        "Swing %":      swing,
        "Whiff %":      whiff,
    }])

def create_batting_stats_profile(df: pd.DataFrame):
    s_call   = df.get('PitchCall', pd.Series(dtype=object))
    play     = df.get('PlayResult', pd.Series(dtype=object))
    korbb    = df.get('KorBB', pd.Series(dtype=object))
    exitv    = pd.to_numeric(df.get('ExitSpeed', pd.Series(dtype=float)), errors="coerce")
    angle    = pd.to_numeric(df.get('Angle', pd.Series(dtype=float)), errors="coerce")
    pitchofpa= pd.to_numeric(df.get('PitchofPA', pd.Series(dtype=float)), errors="coerce")

    pa_mask   = pitchofpa.eq(1)
    hit_mask  = (s_call.eq('InPlay') & play.isin(['Single','Double','Triple','HomeRun']))
    so_mask   = korbb.eq('Strikeout')
    bbout     = s_call.eq('InPlay') & play.eq('Out')
    fc_mask   = play.eq('FieldersChoice')
    err_mask  = play.eq('Error')
    walk_mask = korbb.eq('Walk')
    hbp_mask  = s_call.eq('HitByPitch')

    hits   = int(hit_mask.sum())
    so     = int(so_mask.sum())
    bbouts = int(bbout.sum())
    fc     = int(fc_mask.sum())
    err    = int(err_mask.sum())
    ab     = hits + so + bbouts + fc + err

    walks = int(walk_mask.sum())
    hbp   = int(hbp_mask.sum())
    pa    = int(pa_mask.sum())

    inplay_mask = s_call.eq('InPlay')
    bases = (play.eq('Single').sum()
             + 2*play.eq('Double').sum()
             + 3*play.eq('Triple').sum()
             + 4*play.eq('HomeRun').sum())

    avg_exit  = exitv[inplay_mask].mean()
    max_exit  = exitv[inplay_mask].max()
    avg_angle = angle[inplay_mask].mean()

    ba  = hits/ab if ab else 0.0
    obp = (hits + walks + hbp)/pa if pa else 0.0
    slg = bases/ab if ab else 0.0
    ops = obp + slg
    hard = (exitv[inplay_mask] >= 95).mean()*100 if inplay_mask.any() else 0.0
    k_pct = (so/pa*100) if pa else 0.0
    bb_pct = (walks/pa*100) if pa else 0.0

    stats = pd.DataFrame([{
        "Avg Exit Vel": round(avg_exit, 2) if pd.notna(avg_exit) else np.nan,
        "Max Exit Vel": round(max_exit, 2) if pd.notna(max_exit) else np.nan,
        "Avg Angle":    round(avg_angle, 2) if pd.notna(avg_angle) else np.nan,
        "Hits":         hits,
        "SO":           so,
        "AVG":          ba,
        "OBP":          obp,
        "SLG":          slg,
        "OPS":          ops,
        "HardHit %":    round(hard, 1) if pd.notna(hard) else np.nan,
        "K %":          k_pct,
        "BB %":         bb_pct,
    }])

    return stats, pa, ab

# ──────────────────────────────────────────────────────────────────────────────
# COLLEGE BARREL MASK
# ──────────────────────────────────────────────────────────────────────────────
def college_barrel_mask(ev_series: pd.Series, la_series: pd.Series) -> pd.Series:
    ev = pd.to_numeric(ev_series, errors="coerce")
    la = pd.to_numeric(la_series, errors="coerce")
    return (ev >= 95.0) & la.between(10.0, 35.0, inclusive="both")

# ──────────────────────────────────────────────────────────────────────────────
# xwOBA SUPPORT
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_prob_lookup(path: str):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
    except Exception:
        pass
    return None

PROB_LOOKUP = load_prob_lookup(PROB_LOOKUP_PATH)

def _bin_ev_la(df: pd.DataFrame):
    df = df.copy()
    df["ExitSpeed"] = pd.to_numeric(df.get("ExitSpeed"), errors="coerce")
    df["Angle"]     = pd.to_numeric(df.get("Angle"), errors="coerce")
    ev_bins = np.arange(40, 115+3, 3)
    la_bins = np.arange(-30, 50+5, 5)
    df["EV_bin"] = pd.cut(df["ExitSpeed"], bins=ev_bins, right=False)
    df["LA_bin"] = pd.cut(df["Angle"],     bins=la_bins, right=False)
    return df

def merge_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    if PROB_LOOKUP is None or df.empty:
        return df
    if "EV_bin" not in df.columns or "LA_bin" not in df.columns:
        df = _bin_ev_la(df)

    lk = PROB_LOOKUP.copy()
    for c in ["1B","2B","3B","HR","Out"]:
        if c not in lk.columns:
            lk[c] = 0.0

    left = df.copy()
    left["EV_bin_str"] = left["EV_bin"].astype(str)
    left["LA_bin_str"] = left["LA_bin"].astype(str)

    right = lk.copy()
    right["EV_bin_str"] = right["EV_bin"].astype(str)
    right["LA_bin_str"] = right["LA_bin"].astype(str)

    merged = left.merge(
        right[["EV_bin_str","LA_bin_str","1B","2B","3B","HR"]],
        on=["EV_bin_str","LA_bin_str"],
        how="left",
        suffixes=("","_p")
    )
    merged = merged.rename(columns={"1B":"p1B","2B":"p2B","3B":"p3B","HR":"pHR"})
    return merged

def _derive_ev_counts(df: pd.DataFrame):
    s_call = df.get('PitchCall', pd.Series(dtype=object))
    play   = df.get('PlayResult', pd.Series(dtype=object))
    korbb  = df.get('KorBB', pd.Series(dtype=object))
    pitchofpa = pd.to_numeric(df.get('PitchofPA', pd.Series(dtype=float)), errors="coerce")

    inplay = s_call.eq('InPlay')
    singles = int((inplay & play.eq('Single')).sum())
    doubles = int((inplay & play.eq('Double')).sum())
    triples = int((inplay & play.eq('Triple')).sum())
    hrs     = int((inplay & play.eq('HomeRun')).sum())
    hits    = singles + doubles + triples + hrs

    bb      = int((korbb == 'Walk').sum())
    ibb     = 0
    hbp     = int((s_call == 'HitByPitch').sum())
    sf_mask = play.astype(str).str.contains(r'(?i)sacrifice\s*fly|^SF$', na=False)
    sf      = int(sf_mask.sum())

    bbout   = inplay & play.eq('Out')
    fc_mask = play.eq('FieldersChoice')
    err_mask= play.eq('Error')
    so_mask = (korbb == 'Strikeout')

    ab = int(hits + so_mask.sum() + bbout.sum() + fc_mask.sum() + err_mask.sum())
    pa = int((pitchofpa == 1).sum())

    return {
        "1B": singles, "2B": doubles, "3B": triples, "HR": hrs,
        "BB": bb, "IBB": ibb, "HBP": hbp, "SF": sf,
        "AB": ab, "PA": pa,
    }

def _compute_woba(df: pd.DataFrame) -> float:
    ev = _derive_ev_counts(df)
    den = ev["AB"] + ev["BB"] - ev["IBB"] + ev["HBP"] + ev["SF"]
    if den <= 0:
        return float('nan')
    num = (
        WOBAC_2025["wBB"]  * (ev["BB"] - ev["IBB"])
      + WOBAC_2025["wHBP"] * ev["HBP"]
      + WOBAC_2025["w1B"]  * ev["1B"]
      + WOBAC_2025["w2B"]  * ev["2B"]
      + WOBAC_2025["w3B"]  * ev["3B"]
      + WOBAC_2025["wHR"]  * ev["HR"]
    )
    return num / den

def _compute_xwoba(df: pd.DataFrame) -> float:
    need = ["p1B","p2B","p3B","pHR"]
    if not all(c in df.columns for c in need):
        return float('nan')

    p1 = pd.to_numeric(df["p1B"], errors="coerce").fillna(0.0)
    p2 = pd.to_numeric(df["p2B"], errors="coerce").fillna(0.0)
    p3 = pd.to_numeric(df["p3B"], errors="coerce").fillna(0.0)
    pH = pd.to_numeric(df["pHR"], errors="coerce").fillna(0.0)

    exp_contact_num = (
        WOBAC_2025["w1B"]*p1 + WOBAC_2025["w2B"]*p2
      + WOBAC_2025["w3B"]*p3 + WOBAC_2025["wHR"]*pH
    ).sum()

    s_call = df.get('PitchCall', pd.Series(dtype=object))
    korbb  = df.get('KorBB', pd.Series(dtype=object))
    exp_num = exp_contact_num \
              + WOBAC_2025["wBB"]  * int((korbb == 'Walk').sum()) \
              + WOBAC_2025["wHBP"] * int((s_call == 'HitByPitch').sum())

    ev = _derive_ev_counts(df)
    den = ev["AB"] + ev["BB"] - ev["IBB"] + ev["HBP"] + ev["SF"]
    if den <= 0:
        return float('nan')
    return exp_num / den

# ──────────────────────────────────────────────────────────────────────────────
# PITCH TYPE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _pitch_type_col(df: pd.DataFrame) -> str:
    for c in ["AutoPitchType", "TaggedPitchType", "PitchType", "Pitch_Name"]:
        if c in df.columns:
            return c
    df["AutoPitchType"] = ""
    return "AutoPitchType"

def _norm_text(x) -> str:
    try:
        return str(x).strip().lower()
    except Exception:
        return ""

def pitch_group_of(label: str) -> str:
    t = _norm_text(label)

    if any(k in t for k in ["fast", "four", "2-seam", "two seam", "sinker", "cutter", "cut"]):
        return "Fastball"
    if any(k in t for k in ["change", "split", "fork", "vulcan", "palm"]):
        return "Offspeed"
    if any(k in t for k in ["slider", "sweeper", "curve", "slurve", "knuck"]):
        return "Breaking"

    if t in {"ff","fa","fb","ft","si","fc"}: return "Fastball"
    if t in {"ch","sp"}:                      return "Offspeed"
    if t in {"sl","cu","kc"}:                 return "Breaking"

    return "Unknown"

# ──────────────────────────────────────────────────────────────────────────────
# SPLIT METRICS CORE
# ──────────────────────────────────────────────────────────────────────────────
def _compute_split_core(df: pd.DataFrame) -> dict:
    s_call = df.get('PitchCall', pd.Series(dtype=object))
    play   = df.get('PlayResult', pd.Series(dtype=object))
    korbb  = df.get('KorBB', pd.Series(dtype=object))
    exitv  = pd.to_numeric(df.get('ExitSpeed', pd.Series(dtype=float)), errors="coerce")
    angle  = pd.to_numeric(df.get('Angle', pd.Series(dtype=float)), errors="coerce")
    pitchofpa = pd.to_numeric(df.get('PitchofPA', pd.Series(dtype=float)), errors="coerce")
    lside  = pd.to_numeric(df.get('PlateLocSide', pd.Series(dtype=float)), errors='coerce')
    lht    = pd.to_numeric(df.get('PlateLocHeight', pd.Series(dtype=float)), errors='coerce')

    pa = int((pitchofpa == 1).sum())

    isswing   = s_call.isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
    iswhiff   = s_call.eq('StrikeSwinging')
    iscontact = s_call.isin(['InPlay','FoulBallNotFieldable','FoulBallFieldable'])
    isinzone  = lside.between(-0.83,0.83) & lht.between(1.5,3.5)

    so_mask   = korbb.eq('Strikeout')
    walk_mask = korbb.eq('Walk')
    hbp_mask  = s_call.eq('HitByPitch')
    inplay    = s_call.eq('InPlay')

    hits_mask = inplay & play.isin(['Single','Double','Triple','HomeRun'])
    hits = int(hits_mask.sum())
    so   = int(so_mask.sum())
    bb   = int(walk_mask.sum())

    doubles = int(play.eq('Double').sum())
    triples = int(play.eq('Triple').sum())
    hrs     = int(play.eq('HomeRun').sum())

    bbout   = inplay & play.eq('Out')
    fc_mask = play.eq('FieldersChoice')
    err_mask= play.eq('Error')
    ab      = int(hits + so + bbout.sum() + fc_mask.sum() + err_mask.sum())

    bases = (play.eq('Single').sum()
             + 2*play.eq('Double').sum()
             + 3*play.eq('Triple').sum()
             + 4*play.eq('HomeRun').sum())

    ba  = hits/ab if ab else 0.0
    obp = (hits + bb + hbp_mask.sum())/pa if pa else 0.0
    slg = bases/ab if ab else 0.0
    ops = obp + slg

    avg_ev = exitv[inplay].mean()
    max_ev = exitv[inplay].max()
    avg_la = angle[inplay].mean()
    hard   = (exitv[inplay] >= 95).mean()*100 if inplay.any() else 0.0
    
    if inplay.any():
        barrel_mask = college_barrel_mask(exitv[inplay], angle[inplay])
        barrel = float(barrel_mask.mean()) * 100.0
    else:
        barrel = 0.0

    swing = isswing.mean()*100 if len(isswing) else 0.0
    whiff = (iswhiff.sum() / max(isswing.sum(),1) * 100) if len(isswing) else 0.0
    chase = (isswing[~isinzone].mean()*100) if (~isinzone).sum() else 0.0
    z_swing = (isswing[isinzone].mean()*100) if isinzone.sum() else 0.0
    z_contact = ((iscontact & isinzone).sum() / max(isswing[isinzone].sum(),1) * 100) if isinzone.sum() else 0.0
    z_whiff = ((iswhiff & isinzone).sum() / max(isswing[isinzone].sum(),1) * 100) if isinzone.sum() else 0.0

    woba  = _compute_woba(df)
    xwoba = _compute_xwoba(df)

    return {
        "PA": pa, "AB": ab, "SO": so, "BB": bb, "Hits": hits,
        "2B": doubles, "3B": triples, "HR": hrs,
        "AVG": ba, "OBP": obp, "SLG": slg, "OPS": ops,
        "Avg EV": avg_ev, "Max EV": max_ev, "Avg LA": avg_la,
        "HardHit%": hard, "Barrel%": barrel,
        "Swing%": swing, "Whiff%": whiff, "Chase%": chase,
        "ZSwing%": z_swing, "ZContact%": z_contact, "ZWhiff%": z_whiff,
        "wOBA": woba, "xwOBA": xwoba,
    }

# ──────────────────────────────────────────────────────────────────────────────
# FALL SUMMARY HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def create_game_by_game_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    df['DateOnly'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    game_rows = []
    
    for date, game_df in df.groupby('DateOnly'):
        if pd.isna(date):
            continue
        
        stats = _compute_split_core(game_df)
        opponent = game_df['PitcherTeam'].mode()[0] if not game_df['PitcherTeam'].empty else "—"
        opponent = TEAM_NAME_MAP.get(str(opponent), str(opponent))
        
        game_rows.append({
            'Date': format_date_long(date),
            'Opponent': opponent,
            'PA': stats['PA'],
            'AB': stats['AB'],
            'H': stats['Hits'],
            'AVG': stats['AVG'],
            'OBP': stats['OBP'],
            'SLG': stats['SLG'],
            'HR': stats['HR'],
            'BB': stats['BB'],
            'SO': stats['SO'],
            'HardHit%': stats['HardHit%'],
        })
    
    if not game_rows:
        return pd.DataFrame()
    
    result = pd.DataFrame(game_rows)
    
    for col in ['AVG', 'OBP', 'SLG']:
        result[col] = result[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    
    result['HardHit%'] = result['HardHit%'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    
    return result

def create_pitch_type_splits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    col = _pitch_type_col(df)
    df['PitchGroup'] = df[col].map(pitch_group_of)
    
    rows = []
    for group in ['Fastball', 'Breaking', 'Offspeed']:
        sub = df[df['PitchGroup'] == group]
        if sub.empty:
            continue
        
        stats = _compute_split_core(sub)
        rows.append({
            'Pitch Type': group,
            'PA': stats['PA'],
            'AB': stats['AB'],
            'AVG': stats['AVG'],
            'OBP': stats['OBP'],
            'SLG': stats['SLG'],
            'OPS': stats['OPS'],
            'wOBA': stats['wOBA'],
            'Whiff%': stats['Whiff%'],
            'HardHit%': stats['HardHit%'],
            'Barrel%': stats['Barrel%'],
        })
    
    if not rows:
        return pd.DataFrame()
    
    result = pd.DataFrame(rows)
    
    for col in ['AVG', 'OBP', 'SLG', 'OPS', 'wOBA']:
        result[col] = result[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    
    for col in ['Whiff%', 'HardHit%', 'Barrel%']:
        result[col] = result[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    
    return result

def create_handedness_splits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    rows = []
    for hand, label in [('L', 'vs LHP'), ('R', 'vs RHP')]:
        sub = df[df.get('PitcherThrows', pd.Series(dtype=object)).astype(str).str.upper().str.startswith(hand)]
        if sub.empty:
            continue
        
        stats = _compute_split_core(sub)
        rows.append({
            'Split': label,
            'PA': stats['PA'],
            'AB': stats['AB'],
            'AVG': stats['AVG'],
            'OBP': stats['OBP'],
            'SLG': stats['SLG'],
            'OPS': stats['OPS'],
            'wOBA': stats['wOBA'],
            'K%': (stats['SO']/stats['PA']*100) if stats['PA'] > 0 else 0,
            'BB%': (stats['BB']/stats['PA']*100) if stats['PA'] > 0 else 0,
        })
    
    if not rows:
        return pd.DataFrame()
    
    result = pd.DataFrame(rows)
    
    for col in ['AVG', 'OBP', 'SLG', 'OPS', 'wOBA']:
        result[col] = result[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    
    for col in ['K%', 'BB%']:
        result[col] = result[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    
    return result

def create_count_splits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'Balls' not in df.columns or 'Strikes' not in df.columns:
        return pd.DataFrame()
    
    df['Balls'] = pd.to_numeric(df['Balls'], errors='coerce')
    df['Strikes'] = pd.to_numeric(df['Strikes'], errors='coerce')
    
    df['CountSituation'] = 'Other'
    df.loc[(df['Balls'] > df['Strikes']), 'CountSituation'] = 'Hitter Ahead'
    df.loc[(df['Balls'] < df['Strikes']), 'CountSituation'] = 'Pitcher Ahead'
    df.loc[(df['Balls'] == df['Strikes']), 'CountSituation'] = 'Even'
    df.loc[(df['Strikes'] == 2), 'CountSituation'] = 'Two Strike'
    
    rows = []
    for situation in ['Hitter Ahead', 'Even', 'Pitcher Ahead', 'Two Strike']:
        sub = df[df['CountSituation'] == situation]
        if sub.empty:
            continue
        
        stats = _compute_split_core(sub)
        rows.append({
            'Count': situation,
            'PA': stats['PA'],
            'AVG': stats['AVG'],
            'OBP': stats['OBP'],
            'SLG': stats['SLG'],
            'Whiff%': stats['Whiff%'],
            'Chase%': stats['Chase%'],
        })
    
    if not rows:
        return pd.DataFrame()
    
    result = pd.DataFrame(rows)
    
    for col in ['AVG', 'OBP', 'SLG']:
        result[col] = result[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    
    for col in ['Whiff%', 'Chase%']:
        result[col] = result[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    
    return result

def create_progress_chart(df: pd.DataFrame, metric='OPS') -> plt.Figure:
    if df.empty:
        return None
    
    df['DateOnly'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    dates = sorted(df['DateOnly'].dropna().unique())
    
    if len(dates) < 2:
        return None
    
    # Calculate cumulative stats up to each date
    stats_by_date = []
    for i, date in enumerate(dates):
        # Get all games up to and including this date
        cumulative_df = df[df['DateOnly'].isin(dates[:i+1])]
        stats = _compute_split_core(cumulative_df)
        stats_by_date.append({
            'Date': date,
            'OPS': stats['OPS'],
            'AVG': stats['AVG'],
            'wOBA': stats['wOBA'],
            'HardHit%': stats['HardHit%'],
            'Barrel%': stats['Barrel%'],
            'Whiff%': stats['Whiff%'],
        })
    
    df_stats = pd.DataFrame(stats_by_date)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_values = df_stats[metric].values
    dates_plot = range(len(dates))
    
    ax.plot(dates_plot, metric_values, marker='o', linewidth=2.5, 
            markersize=8, color=HUSKER_RED, label=metric)
    
    # Final value line
    final_val = metric_values[-1]
    ax.axhline(y=final_val, color='gray', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'Final: {final_val:.3f}')
    
    ax.set_xlabel('Game', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Progression - Fall 2025', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(dates_plot)
    ax.set_xticklabels([f"Game {i+1}" for i in dates_plot], rotation=45, ha='right')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# PROFILE TABLES
# ──────────────────────────────────────────────────────────────────────────────
def _pretty_pitch_name(p: str) -> str:
    alias_map = {
        "FourSeamFastBall":"Fastball",
        "FourSeam":"Fastball",
        "FS":"Fastball",
        "SL":"Slider",
        "CH":"Changeup",
        "CU":"Curveball",
    }
    name = alias_map.get(p, p)
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name

def _sorted_unique_pitches(series: pd.Series) -> list:
    preferred = ["Fastball","FourSeam","TwoSeam","Sinker","Cutter","Slider","Curveball","Changeup","Splitter","Sweeper","Knuckleball","Other"]
    vals = series.astype(str).dropna().tolist()
    uniq = [v for v in sorted(set(vals)) if v and v.lower() != "nan"]
    def key(p):
        name = _pretty_pitch_name(p)
        try:
            base = name.replace(" ", "")
            for pref in preferred:
                if pref.lower() in [name.lower(), base.lower()]:
                    return (preferred.index(pref), name)
        except ValueError:
            pass
        return (len(preferred)+1, name)
    return sorted(uniq, key=key)

def build_profile_tables(df_profiles: pd.DataFrame):
    if df_profiles.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    core_total = _compute_split_core(df_profiles)
    row_total_counts = {
        "Split": "Total",
        **{k: core_total[k] for k in ["PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS"]},
        "wOBA": core_total.get("wOBA", np.nan),
        "xwOBA": core_total.get("xwOBA", np.nan),
    }
    row_total_rates  = {
        "Split": "Total",
        **{k: core_total[k] for k in ["Avg EV","Max EV","Avg LA","HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]}
    }

    t1_rows = [row_total_counts]
    t2_rows = [row_total_rates]

    col = _pitch_type_col(df_profiles)
    groups_series = df_profiles[col].map(pitch_group_of)

    for g in ["Fastball", "Offspeed", "Breaking"]:
        sub = df_profiles[groups_series == g]
        if sub.empty:
            t1_rows.append({
                "Split": g,
                **{k: np.nan for k in ["PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS"]},
                "wOBA": np.nan, "xwOBA": np.nan
            })
            t2_rows.append({
                "Split": g,
                **{k: np.nan for k in ["Avg EV","Max EV","Avg LA","HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]}
            })
            continue

        core = _compute_split_core(sub)
        t1_rows.append({
            "Split": g,
            **{k: core[k] for k in ["PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS"]},
            "wOBA": core.get("wOBA", np.nan),
            "xwOBA": core.get("xwOBA", np.nan),
        })
        t2_rows.append({
            "Split": g,
            **{k: core[k] for k in ["Avg EV","Max EV","Avg LA","HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]}
        })

    t1 = pd.DataFrame(
        t1_rows,
        columns=["Split","PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS","wOBA","xwOBA"]
    )
    t2 = pd.DataFrame(
        t2_rows,
        columns=["Split","Avg EV","Max EV","Avg LA","HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]
    )

    for c in ["AVG","OBP","SLG","OPS","wOBA","xwOBA"]:
        if c in t1.columns:
            t1[c] = t1[c].apply(lambda v: "—" if pd.isna(v) else (f"{float(v):.3f}"[1:] if float(v) < 1.0 else f"{float(v):.3f}"))

    for c in ["Avg EV","Max EV","Avg LA"]:
        if c in t2.columns:
            t2[c] = t2[c].apply(lambda v: "—" if pd.isna(v) else f"{float(v):.2f}")
    for c in ["HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]:
        if c in t2.columns:
            t2[c] = t2[c].apply(lambda v: "—" if pd.isna(v) else f"{round(float(v),1)}%")

    t3 = create_batted_ball_profile(df_profiles).copy()
    for c in t3.columns:
        t3[c] = t3[c].apply(lambda v: "—" if pd.isna(v) else f"{float(v):.1f}%")

    return t1, t2, t3

# ──────────────────────────────────────────────────────────────────────────────
# RANKINGS
# ──────────────────────────────────────────────────────────────────────────────
RANKABLE_COLS = [
    "PA","AB","SO","BB","Hits","2B","3B","HR",
    "AVG","OBP","SLG","OPS",
    "wOBA","xwOBA",
    "Avg EV","Max EV","HardHit%","Barrel%",
    "ZWhiff%","Chase%", "Whiff%"
]

def build_rankings_numeric(df_player_scope: pd.DataFrame, display_name_by_key: dict) -> pd.DataFrame:
    rows = []
    for key, g in df_player_scope.groupby("BatterKey"):
        if not key:
            continue
        core = _compute_split_core(g)
        row = {"Player": display_name_by_key.get(key, key)}
        row.update({k: core.get(k, np.nan) for k in RANKABLE_COLS})
        rows.append(row)

    out = pd.DataFrame(rows, columns=["Player"] + RANKABLE_COLS)
    for c in RANKABLE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def style_rankings(df: pd.DataFrame):
    numeric_cols = [c for c in RANKABLE_COLS if c in df.columns]
    inverted_cols = {"SO", "ZWhiff%", "Chase%", "Whiff%"}

    def color_leader_last(col: pd.Series):
        if col.name not in numeric_cols:
            return [''] * len(col)
        s = pd.to_numeric(col, errors="coerce")
        if s.dropna().empty:
            return [''] * len(col)

        max_val = s.max()
        min_val = s.min()
        styles = []
        invert = col.name in inverted_cols

        for v in s:
            if pd.isna(v) or max_val == min_val:
                styles.append('')
            else:
                if not invert:
                    if v == max_val:
                        styles.append('background-color: #b6f2b0;')
                    elif v == min_val:
                        styles.append('background-color: #f9b0b0;')
                    else:
                        styles.append('')
                else:
                    if v == min_val:
                        styles.append('background-color: #b6f2b0;')
                    elif v == max_val:
                        styles.append('background-color: #f9b0b0;')
                    else:
                        styles.append('')
        return styles

    header_props = f'background-color: {HUSKER_RED}; color: white; white-space: nowrap;'
    sty = (
        df.style
          .hide(axis="index")
          .set_table_styles([
              {'selector': 'thead th', 'props': header_props},
              {'selector': 'th.col_heading', 'props': header_props},
              {'selector': 'th', 'props': header_props},
          ])
          .apply(color_leader_last, axis=0)
          .format({
              "PA":"{:.0f}", "AB":"{:.0f}", "SO":"{:.0f}", "BB":"{:.0f}",
              "Hits":"{:.0f}", "2B":"{:.0f}", "3B":"{:.0f}", "HR":"{:.0f}",
              "AVG":"{:.3f}", "OBP":"{:.3f}", "SLG":"{:.3f}", "OPS":"{:.3f}",
              "wOBA":"{:.3f}", "xwOBA":"{:.3f}",
              "Avg EV":"{:.2f}", "Max EV":"{:.2f}",
              "HardHit%":"{:.1f}%", "Barrel%":"{:.1f}%",
              "ZWhiff%":"{:.1f}%", "Chase%":"{:.1f}%", "Whiff%":"{:.1f}%"
          }, na_rep="—")
    )
    return sty

# ──────────────────────────────────────────────────────────────────────────────
# SPRAY CHARTS
# ──────────────────────────────────────────────────────────────────────────────
def create_spray_chart(df_game: pd.DataFrame, batter_display_name: str):
    inplay = df_game[df_game.get('PitchCall') == 'InPlay'].copy()
    
    if inplay.empty:
        warning_message("No balls in play found for this game.")
        return None
    
    if 'Bearing' not in inplay.columns or 'Distance' not in inplay.columns:
        warning_message("Missing 'Bearing' or 'Distance' columns in data.")
        return None
    
    inplay['Bearing'] = pd.to_numeric(inplay['Bearing'], errors='coerce')
    inplay['Distance'] = pd.to_numeric(inplay['Distance'], errors='coerce')
    inplay = inplay.dropna(subset=['Bearing', 'Distance'])
    
    if inplay.empty:
        warning_message("No balls in play with valid Bearing and Distance data.")
        return None
    
    inplay = inplay.sort_values(['GameID', 'Inning', 'Top/Bottom', 'PAofInning', 'PitchofPA'])
    inplay['PA_num'] = inplay.groupby(['GameID', 'Inning', 'Top/Bottom', 'PAofInning']).ngroup() + 1
    
    coords = [bearing_distance_to_xy(row['Bearing'], row['Distance']) 
              for _, row in inplay.iterrows()]
    inplay['x'] = [c[0] for c in coords]
    inplay['y'] = [c[1] for c in coords]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    angles = np.linspace(45, 135, 100)
    wall_data = []
    for angle in angles:
        if angle <= 90:
            t = (angle - 45) / (90 - 45)
            dist = 335 + t * (395 - 335)
        else:
            t = (angle - 90) / (135 - 90)
            dist = 395 + t * (325 - 395)
        wall_data.append((angle, dist))
    
    draw_dirt_diamond(ax, origin=(0.0, 0.0), size=140, custom_wall_distances=wall_data)
    
    wall_x = [dist * np.cos(np.radians(ang)) for ang, dist in wall_data]
    wall_y = [dist * np.sin(np.radians(ang)) for ang, dist in wall_data]
    ax.plot(wall_x, wall_y, 'k-', linewidth=3, zorder=10)
    
    for angle, dist, label in [(45, 335, '335'), (90, 395, '395'), (135, 325, '325')]:
        rad = np.radians(angle)
        x = dist * np.cos(rad)
        y = dist * np.sin(rad)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', 
                facecolor='yellow', edgecolor='black', linewidth=2, alpha=0.9), zorder=11)
    
    n_pas = inplay['PA_num'].nunique()
    colors_list = plt.cm.tab20(np.linspace(0, 1, min(n_pas, 20)))
    pa_colors = {pa: colors_list[i % 20] for i, pa in enumerate(sorted(inplay['PA_num'].unique()))}
    
    for idx, row in inplay.iterrows():
        pa_num = row['PA_num']
        play_result = str(row.get('PlayResult', ''))
        exit_speed = row.get('ExitSpeed', np.nan)
        
        if pd.notna(exit_speed):
            marker_size = max(150, min(exit_speed * 5, 600))
        else:
            marker_size = 250
        
        if play_result in ['Single', 'Double', 'Triple', 'HomeRun']:
            marker = 'o'
            edgecolor = 'darkgreen'
            linewidth = 3
        else:
            marker = 'X'
            edgecolor = 'darkred'
            linewidth = 3
        
        ax.scatter(row['x'], row['y'], 
                  c=[pa_colors[pa_num]], 
                  s=marker_size, 
                  marker=marker,
                  edgecolors=edgecolor, 
                  linewidths=linewidth,
                  alpha=0.95,
                  zorder=20)
        
        ax.text(row['x'], row['y'], str(pa_num), 
               ha='center', va='center', 
               fontsize=11, fontweight='bold',
               color='white',
               bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', alpha=0.5),
               zorder=21)
    
    legend_elements = []
    for pa in sorted(pa_colors.keys()):
        pa_rows = inplay[inplay['PA_num'] == pa]
        if pa_rows.empty:
            continue
        
        row = pa_rows.iloc[-1]
        
        ev = row.get('ExitSpeed', np.nan)
        ev_str = f"{ev:.1f}" if pd.notna(ev) else "—"
        
        la = row.get('Angle', np.nan)
        la_str = f"{la:.1f}°" if pd.notna(la) else "—"
        
        dist = row.get('Distance', np.nan)
        dist_str = f"{dist:.0f}'" if pd.notna(dist) else "—"
        
        hit_type = row.get('TaggedHitType', '')
        if pd.notna(hit_type) and str(hit_type):
            ht = str(hit_type).replace('GroundBall', 'GB').replace('LineDrive', 'LD').replace('FlyBall', 'FB').replace('Popup', 'PU')
        else:
            ht = "—"
        
        outcome = row.get('PlayResult', '')
        outcome_str = str(outcome) if pd.notna(outcome) and str(outcome) else "Out"
        
        label = f"PA {pa}: {outcome_str} | {ev_str} mph, {la_str}, {dist_str} | {ht}"
        
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=pa_colors[pa], markersize=10,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=label)
        )
    
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='darkgreen', markeredgewidth=2.5,
               label='Hit'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='darkred', markeredgewidth=2.5,
               label='Out')
    ])
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(0.02, 0.98), frameon=True, 
             fancybox=True, shadow=True, fontsize=9)
    
    max_dist = max(inplay['Distance'].max(), 400)
    ax.set_xlim(-max_dist * 0.85, max_dist * 0.85)
    ax.set_ylim(-30, max_dist * 1.1)
    ax.set_aspect('equal')
    
    date_str = format_date_long(inplay['Date'].iloc[0]) if 'Date' in inplay.columns else ""
    ax.set_title(f"{batter_display_name} — Spray Chart\n{date_str}", 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_profile_spray_chart(df_profiles: pd.DataFrame, batter_display_name: str):
    inplay = df_profiles[df_profiles.get('PitchCall') == 'InPlay'].copy()
    
    if inplay.empty:
        return None
    
    if 'Bearing' not in inplay.columns or 'Distance' not in inplay.columns:
        return None
    
    inplay['Bearing'] = pd.to_numeric(inplay['Bearing'], errors='coerce')
    inplay['Distance'] = pd.to_numeric(inplay['Distance'], errors='coerce')
    inplay = inplay.dropna(subset=['Bearing', 'Distance'])
    
    if inplay.empty:
        return None
    
    coords = [bearing_distance_to_xy(row['Bearing'], row['Distance']) 
              for _, row in inplay.iterrows()]
    inplay['x'] = [c[0] for c in coords]
    inplay['y'] = [c[1] for c in coords]
    
    def categorize_hit_type(hit_type):
        if pd.isna(hit_type):
            return 'Other'
        ht = str(hit_type).lower()
        if 'ground' in ht:
            return 'GroundBall'
        elif 'line' in ht:
            return 'LineDrive'
        elif 'fly' in ht:
            return 'FlyBall'
        elif 'popup' in ht or 'pop' in ht:
            return 'Popup'
        else:
            return 'Other'
    
    inplay['HitCategory'] = inplay.get('TaggedHitType', pd.Series(dtype=object)).apply(categorize_hit_type)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    angles = np.linspace(45, 135, 100)
    wall_data = []
    for angle in angles:
        if angle <= 90:
            t = (angle - 45) / (90 - 45)
            dist = 335 + t * (395 - 335)
        else:
            t = (angle - 90) / (135 - 90)
            dist = 395 + t * (325 - 395)
        wall_data.append((angle, dist))
    
    draw_dirt_diamond(ax, origin=(0.0, 0.0), size=100, custom_wall_distances=wall_data)
    
    wall_x = [dist * np.cos(np.radians(ang)) for ang, dist in wall_data]
    wall_y = [dist * np.sin(np.radians(ang)) for ang, dist in wall_data]
    ax.plot(wall_x, wall_y, 'k-', linewidth=3, zorder=10)
    
    for angle, dist, label in [(45, 335, '335'), (90, 395, '395'), (135, 325, '325')]:
        rad = np.radians(angle)
        x = dist * np.cos(rad)
        y = dist * np.sin(rad)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', 
                facecolor='yellow', edgecolor='black', linewidth=2, alpha=0.9), zorder=11)
    
    hit_type_colors = {
        'GroundBall': '#DC143C',
        'LineDrive': '#FFD700',
        'FlyBall': '#1E90FF',
        'Popup': '#FF69B4',
        'Other': '#A9A9A9'
    }
    
    for idx, row in inplay.iterrows():
        hit_cat = row['HitCategory']
        play_result = str(row.get('PlayResult', ''))
        
        marker_size = 120
        
        if play_result in ['Single', 'Double', 'Triple', 'HomeRun']:
            edgecolor = 'black'
            linewidth = 2
        else:
            edgecolor = 'black'
            linewidth = 1.5
        
        ax.scatter(row['x'], row['y'], 
                  c=hit_type_colors.get(hit_cat, '#A9A9A9'), 
                  s=marker_size, 
                  marker='o',
                  edgecolors=edgecolor, 
                  linewidths=linewidth,
                  alpha=0.85,
                  zorder=20)
    
    legend_elements = []
    
    for hit_type in ['GroundBall', 'LineDrive', 'FlyBall', 'Popup']:
        count = (inplay['HitCategory'] == hit_type).sum()
        if count > 0:
            label = hit_type.replace('GroundBall', 'Ground Ball').replace('LineDrive', 'Line Drive').replace('FlyBall', 'Fly Ball')
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=hit_type_colors[hit_type], 
                       markersize=10,
                       markeredgecolor='black', 
                       markeredgewidth=1.5,
                       label=f'{label} ({count})')
            )
    
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, markeredgecolor='black', markeredgewidth=2,
               label='Hit (thick edge)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='black', markeredgewidth=1,
               label='Out (thin edge)')
    ])
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(0.02, 0.98), frameon=True, 
             fancybox=True, shadow=True, fontsize=10)
    
    max_dist = max(inplay['Distance'].max(), 400)
    ax.set_xlim(-max_dist * 0.85, max_dist * 0.85)
    ax.set_ylim(-30, max_dist * 1.1)
    ax.set_aspect('equal')
    
    ax.set_title(f"{batter_display_name} — Spray Chart (All Batted Balls)", 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# HITTER REPORT
# ──────────────────────────────────────────────────────────────────────────────
def create_hitter_report(df, batter_display_name, ncols=3):
    bdf = df.copy()
    if "PitchofPA" in bdf.columns:
        bdf = bdf.sort_values(["GameID","Inning","Top/Bottom","PAofInning","PitchofPA"]).copy()

    pa_groups = list(bdf.groupby(['GameID','Inning','Top/Bottom','PAofInning']))
    n_pa = len(pa_groups)
    nrows = max(1, math.ceil(n_pa / ncols))

    def _pretty_hit_type(s):
        if pd.isna(s) or s is None:
            return None
        t = str(s)
        t = t.replace("_", " ")
        t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
        return t.strip().title()

    def adjudicate_pa(pa_df: pd.DataFrame) -> dict:
        s_call = pa_df.get('PitchCall')
        play   = pa_df.get('PlayResult')
        korbb  = pa_df.get('KorBB')

        kor = str(korbb.iloc[-1]) if not korbb.empty else ""
        if kor == "Walk":
            return {"label": "▶ PA Result: Walk 🚶", "klass": "Walk", "k_type": None, "res": None, "ev": None, "tag": None}
        if kor == "Strikeout":
            last_call = str(s_call.iloc[-1]) if not s_call.empty else ""
            if last_call == "StrikeSwinging":
                k_type = "Swinging"
            elif last_call == "StrikeCalled":
                k_type = "Looking"
            else:
                k_type = "Unknown"
            return {"label": f"▶ PA Result: Strikeout ({k_type}) 💥", "klass": "K", "k_type": k_type, "res": None, "ev": None, "tag": None}

        strikes = 0
        balls = 0
        k_type = None
        for _, p in pa_df.iterrows():
            call = str(p.get('PitchCall'))
            if call in ("StrikeCalled","StrikeSwinging"):
                strikes += 1
                if strikes == 3:
                    k_type = "Swinging" if call == "StrikeSwinging" else "Looking"
                    break
            elif call in ("FoulBallNotFieldable","FoulBallFieldable"):
                if strikes < 2:
                    strikes += 1
            elif call == "BallCalled":
                balls += 1
                if balls == 4:
                    return {"label": "▶ PA Result: Walk 🚶", "klass": "Walk", "k_type": None, "res": None, "ev": None, "tag": None}
            elif call == "HitByPitch":
                return {"label": "▶ PA Result: HBP", "klass": "Other", "k_type": None, "res": None, "ev": None, "tag": None}

        if strikes >= 3:
            return {"label": f"▶ PA Result: Strikeout ({k_type or 'Unknown'}) 💥", "klass": "K", "k_type": k_type or "Unknown",
                    "res": None, "ev": None, "tag": None}

        inplay = pa_df[s_call == 'InPlay']
        if not inplay.empty:
            last = inplay.iloc[-1]
            res  = last.get('PlayResult', 'InPlay') or 'InPlay'
            es   = last.get('ExitSpeed', np.nan)
            tag  = _pretty_hit_type(last.get('TaggedHitType'))
            bits = [str(res)]
            if pd.notna(es):
                bits[-1] = f"{bits[-1]} ({float(es):.1f} MPH)"
            if tag:
                bits.append(f"— {tag}")
            return {"label": f"▶ PA Result: {' '.join(bits)}", "klass": "InPlay", "k_type": None,
                    "res": res, "ev": float(es) if pd.notna(es) else None, "tag": tag}

        return {"label": "▶ PA Result: —", "klass": "Other", "k_type": None, "res": None, "ev": None, "tag": None}

    descriptions = []
    for _, pa_df in pa_groups:
        lines = []
        for _, p in pa_df.iterrows():
            velo = p.get('EffectiveVelo', np.nan)
            velo_str = f"{float(velo):.1f}" if pd.notna(velo) else "—"
            lines.append(f"{int(p.get('PitchofPA', 0))} / {p.get('AutoPitchType', '—')}  {velo_str} MPH / {p.get('PitchCall', '—')}")
        verdict = adjudicate_pa(pa_df)
        lines.append("  " + verdict["label"])
        descriptions.append(lines)

    fig = plt.figure(figsize=(3 + 4*ncols + 1, 4*nrows))
    gs = GridSpec(nrows, ncols+1, width_ratios=[0.8] + [1]*ncols, wspace=0.15, hspace=0.55)

    date_str = ""
    if pa_groups:
        d0 = pa_groups[0][1].get('Date').iloc[0]
        date_str = format_date_long(d0)
    if batter_display_name or date_str:
        fig.text(0.985, 0.985, f"{batter_display_name} — {date_str}".strip(" —"),
                 ha='right', va='top', fontsize=9, fontweight='normal')

    gd = pd.concat([grp for _, grp in pa_groups]) if pa_groups else pd.DataFrame()
    whiffs = (gd.get('PitchCall')=='StrikeSwinging').sum() if not gd.empty else 0
    hardhits = (pd.to_numeric(gd.get('ExitSpeed'), errors="coerce") > 95).sum() if not gd.empty else 0
    chases = 0
    if not gd.empty:
        pls = pd.to_numeric(gd.get('PlateLocSide'), errors='coerce')
        plh = pd.to_numeric(gd.get('PlateLocHeight'), errors='coerce')
        is_swing = gd.get('PitchCall').isin(['StrikeSwinging'])
        chases = (is_swing & ((pls<-0.83)|(pls>0.83)|(plh<1.5)|(plh>3.5))).sum()
    fig.text(0.55, 0.965, f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",
             ha='center', va='top', fontsize=12)

    for idx, ((_, inn, tb, _), pa_df) in enumerate(pa_groups):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col+1])
        draw_strikezone(ax)
        hand_lbl = "RHP"
        thr = str(pa_df.get('PitcherThrows').iloc[0]) if not pa_df.empty else ""
        if thr.upper().startswith('L'): hand_lbl = "LHP"
        pitcher = str(pa_df.get('Pitcher').iloc[0]) if not pa_df.empty else "—"

        for _, p in pa_df.iterrows():
            mk = {'Fastball':'o', 'Curveball':'s', 'Slider':'^', 'Changeup':'D'}.get(str(p.get('AutoPitchType')), 'o')
            clr = {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan',
                   'FoulBallFieldable':'tan','InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.get(str(p.get('PitchCall')), 'black')
            sz = 200 if str(p.get('AutoPitchType'))=='Slider' else 150
            x = p.get('PlateLocSide'); y = p.get('PlateLocHeight')
            if pd.notna(x) and pd.notna(y):
                ax.scatter(x, y, marker=mk, c=clr, s=sz, edgecolor='white', linewidth=1, zorder=2)
                yoff = -0.05 if str(p.get('AutoPitchType'))=='Slider' else 0
                ax.text(x, y + yoff, str(int(p.get('PitchofPA', 0))), ha='center', va='center',
                        fontsize=6, fontweight='bold', zorder=3)

        ax.set_xlim(*X_LIM); ax.set_ylim(*Y_LIM)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}", fontsize=10, fontweight='bold', pad=6)
        ax.text(0.5, 0.1, f"vs {pitcher} ({hand_lbl})", transform=ax.transAxes,
                ha='center', va='top', fontsize=9, style='italic')

    axd = fig.add_subplot(gs[:, 0]); axd.axis('off')
    y0 = 1.0; dy = 1.0 / (max(1, n_pa) * 5.0)
    for i, lines in enumerate(descriptions, start=1):
        axd.hlines(y0 - dy*0.1, 0, 1, transform=axd.transAxes, color='black', linewidth=1)
        axd.text(0.02, y0, f"PA {i}", fontsize=6, fontweight='bold', transform=axd.transAxes)
        yln = y0 - dy
        for ln in lines:
            axd.text(0.02, yln, ln, fontsize=6, transform=axd.transAxes)
            yln -= dy
        y0 = yln - dy*0.05

    res_handles = [Line2D([0],[0], marker='o', color='w', label=k,
                          markerfacecolor=v, markersize=10, markeredgecolor='k')
                   for k,v in {'StrikeCalled':'#CCCC00','BallCalled':'green',
                               'FoulBallNotFieldable':'tan','FoulBallFieldable':'tan',
                               'InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.items()]
    pitch_handles = [Line2D([0],[0], marker=m, color='w', label=k,
                             markerfacecolor='gray', markersize=10, markeredgecolor='k')
                     for k,m in {'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}.items()]

    fig.legend(
        res_handles, [h.get_label() for h in res_handles], title='Result',
        loc='upper center', bbox_to_anchor=(0.42, 0.035), bbox_transform=fig.transFigure,
        ncol=3, frameon=True, fancybox=True, framealpha=0.95, edgecolor='black',
        borderpad=0.8, columnspacing=1.6, handlelength=1.6, handletextpad=0.6, labelspacing=0.7
    )
    fig.legend(
        pitch_handles, [h.get_label() for h in pitch_handles], title='Pitches',
        loc='upper center', bbox_to_anchor=(0.72, 0.035), bbox_transform=fig.transFigure,
        ncol=4, frameon=True, fancybox=True, framealpha=0.95, edgecolor='black',
        borderpad=0.8, columnspacing=1.6, handlelength=1.6, handletextpad=0.6, labelspacing=0.7
    )

    plt.tight_layout(rect=[0.12, 0.08, 1, 0.94])
    return fig

def hitter_heatmaps(df_filtered_for_profiles: pd.DataFrame, batter_key: str):
    sub = df_filtered_for_profiles[df_filtered_for_profiles.get('BatterKey') == batter_key].copy()
    if sub.empty:
        return None

    sub['iscontact'] = sub.get('PitchCall').isin(['InPlay','FoulBallFieldable','FoulBallNotFieldable'])
    sub['iswhiff']   = sub.get('PitchCall').eq('StrikeSwinging')
    sub['is95plus']  = pd.to_numeric(sub.get('ExitSpeed'), errors="coerce") >= 95

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.25, hspace=0.15)

    def _panel(ax, title, frame):
        draw_strikezone(ax)
        x = pd.to_numeric(frame.get('PlateLocSide'), errors='coerce').to_numpy()
        y = pd.to_numeric(frame.get('PlateLocHeight'), errors='coerce').to_numpy()
        mask = np.isfinite(x) & np.isfinite(y); x, y = x[mask], y[mask]
        if len(x) < 10:
            ax.plot(x, y, 'o', color='deepskyblue', alpha=0.8, markersize=6)
        else:
            xi = np.linspace(*X_LIM, 200)
            yi = np.linspace(*Y_LIM, 200)
            xi_m, yi_m = np.meshgrid(xi, yi)
            zi = compute_density_hitter(x, y, xi_m, yi_m)
            ax.imshow(zi, origin='lower', extent=[*X_LIM, *Y_LIM], aspect='equal', cmap=custom_cmap)
            draw_strikezone(ax)
        ax.set_xlim(*X_LIM); ax.set_ylim(*Y_LIM)
        ax.set_aspect('equal', 'box'); ax.set_title(title, fontsize=10, pad=6, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 0]); _panel(ax1, "Contact", sub[sub['iscontact']])
    ax2 = fig.add_subplot(gs[0, 1]); _panel(ax2, "Whiffs",  sub[sub['iswhiff']])
    ax3 = fig.add_subplot(gs[0, 2]); _panel(ax3, "Damage (95+ EV)", sub[sub['is95plus']])

    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────────────────────────────────────
def _expand_paths(path_like: str):
    if not path_like:
        return []
    if os.path.isdir(path_like):
        files = sorted(glob.glob(os.path.join(path_like, "*.csv")))
        return files
    if os.path.isfile(path_like):
        return [path_like]
    files = sorted(glob.glob(path_like))
    return files

@st.cache_data(show_spinner=True)
def load_many_csv(paths: list) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(p, low_memory=False, encoding="latin-1")
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    return ensure_date_column(out)

@st.cache_data(show_spinner=True)
def load_single_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    return ensure_date_column(df)

def load_for_period(period_label: str, path_2025: str, path_scrim: str, path_2026: str) -> pd.DataFrame:
    if period_label == "2025 season":
        return load_single_csv(path_2025)
    elif period_label == "2025/26 Scrimmages":
        paths = _expand_paths(path_scrim)
        if not paths:
            return pd.DataFrame()
        return load_many_csv(paths)
    elif period_label == "2026 season":
        paths = _expand_paths(path_2026)
        if not paths:
            try:
                return load_single_csv(path_2026)
            except Exception:
                return pd.DataFrame()
        return load_many_csv(paths)
    else:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
render_nb_banner(title="Nebraska Baseball", subtitle="Hitter Analytics Platform")

section_header("Configuration")

col_period, col_expand = st.columns([2, 3])

with col_period:
    period = st.selectbox(
        "📅 Time Period",
        options=["2025 season", "2025/26 Scrimmages", "2026 season"],
        index=0
    )

with col_expand:
    with st.expander("⚙️ Data Paths (optional quick edit)", expanded=False):
        st.caption("Paste a CSV path, a directory path, or a glob pattern.")
        path_2025  = st.text_input("2025 season path", value=DATA_PATH_2025,  key="path_2025")
        path_scrim = st.text_input("2025/26 Scrimmages path/pattern", value=DATA_PATH_SCRIM, key="path_scrim")
        path_2026  = st.text_input("2026 season path/pattern", value=DATA_PATH_2026,  key="path_2026")

path_2025  = st.session_state.get("path_2025", DATA_PATH_2025)
path_scrim = st.session_state.get("path_scrim", DATA_PATH_SCRIM)
path_2026  = st.session_state.get("path_2026", DATA_PATH_2026)

df_all = load_for_period(period, path_2025, path_scrim, path_2026)
if df_all.empty:
    st.error(f"No data loaded for '{period}'. Check the path(s) above.")
    st.stop()

# Build hitter keyspace
for col in ["Batter", "BatterTeam", "PitchofPA", "PitcherThrows", "PitcherTeam",
            "PlayResult", "KorBB", "PitchCall", "AutoPitchType", "ExitSpeed", "Angle",
            "PlateLocSide", "PlateLocHeight", "TaggedHitType", "Bearing", "BatterSide",
            "Distance", "Balls", "Strikes"]:
    if col not in df_all.columns:
        df_all[col] = pd.NA

df_all = ensure_date_column(df_all)

df_neb_bat = df_all[df_all["BatterTeam"].astype(str).str.upper().eq("NEB")].copy()
df_neb_bat["PitchofPA"] = pd.to_numeric(df_neb_bat["PitchofPA"], errors="coerce")
df_neb_bat["BatterKey"]  = df_neb_bat["Batter"].map(normalize_name)
df_neb_bat["BatterDisp"] = df_neb_bat["BatterKey"]

df_neb_bat = _bin_ev_la(df_neb_bat)
df_neb_bat = merge_probabilities(df_neb_bat)

has_pa = df_neb_bat[(df_neb_bat["PitchofPA"] == 1) & df_neb_bat["BatterKey"].ne("")]
if has_pa.empty:
    st.error(f"No Nebraska hitters with plate appearances found for '{period}'.")
    st.stop()

batters_keys = sorted(has_pa["BatterKey"].dropna().unique().tolist())
display_name_by_key = (
    df_neb_bat.groupby("BatterKey")["BatterDisp"]
    .agg(lambda s: s.dropna().value_counts().index[0] if not s.dropna().empty else "")
    .to_dict()
)

professional_divider()

# ══════════════════════════════════════════════════════════════════════════════
# VIEW MODE SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
section_header("View Selection")
view_mode = st.radio(
    "Select View Mode",
    ["Standard Hitter Report", "Profiles & Heatmaps", "Rankings", "Fall Summary"],
    horizontal=True
)

professional_divider()

# ══════════════════════════════════════════════════════════════════════════════
# STANDARD HITTER REPORT
# ══════════════════════════════════════════════════════════════════════════════
if view_mode == "Standard Hitter Report":
    section_header("Standard Hitter Report")
    
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    colB, colD = st.columns([1, 1])

    batter_key_std = colB.selectbox(
        "👤 Select Player",
        options=batters_keys,
        index=0,
        format_func=lambda k: display_name_by_key.get(k, k)
    )

    if batter_key_std:
        df_b_all = df_neb_bat[df_neb_bat["BatterKey"] == batter_key_std].copy()
        df_b_all["DateOnly"] = pd.to_datetime(df_b_all["Date"], errors="coerce").dt.date
        date_groups = df_b_all.groupby("DateOnly")["PitcherTeam"].agg(
            lambda s: sorted(set([TEAM_NAME_MAP.get(str(x), str(x)) for x in s if pd.notna(x)]))
        )
        date_opts, date_labels = [], {}
        for d, teams in date_groups.items():
            if pd.isna(d):
                continue
            label = f"{format_date_long(d)}"
            if teams:
                label += f" ({'/'.join(teams)})"
            date_opts.append(d)
            date_labels[d] = label
        date_opts = sorted(date_opts)
    else:
        df_b_all = df_neb_bat.iloc[0:0].copy()
        date_opts, date_labels = [], {}

    selected_date = colD.selectbox(
        "📅 Select Game Date",
        options=date_opts,
        format_func=lambda d: date_labels.get(d, format_date_long(d)),
        index=len(date_opts)-1 if date_opts else 0
    ) if date_opts else None
    
    st.markdown('</div>', unsafe_allow_html=True)

    if batter_key_std and selected_date:
        df_date = df_b_all[df_b_all["DateOnly"] == selected_date].copy()
    else:
        df_date = df_b_all.iloc[0:0].copy()

    batter_display = display_name_by_key.get(batter_key_std, batter_key_std)

    if df_date.empty:
        info_message("Select a player and game date to see the Standard Hitter Report.")
    else:
        professional_divider()
        
        # Player header card
        st.markdown(f"""
        <div class="player-header">
            <div class="player-name">{batter_display}</div>
            <div class="player-subtitle">{format_date_long(selected_date)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        subsection_header("Pitch-by-Pitch Analysis")
        fig_std = create_hitter_report(df_date, batter_display, ncols=3)
        if fig_std:
            st.pyplot(fig_std)
            plt.close(fig_std)
        
        professional_divider()
        
        subsection_header("Spray Chart")
        fig_spray = create_spray_chart(df_date, batter_display)
        if fig_spray:
            st.pyplot(fig_spray)
            plt.close(fig_spray)
        else:
            info_message("No balls in play with valid location data for this game.")

# ══════════════════════════════════════════════════════════════════════════════
# PROFILES & HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Profiles & Heatmaps":
    section_header("Profiles & Heatmaps")

    batter_key = st.selectbox(
        "👤 Select Player",
        options=batters_keys,
        index=0,
        format_func=lambda k: display_name_by_key.get(k, k)
    )

    subsection_header("Filters")
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    colM, colD2, colN, colH = st.columns([1.2, 1.2, 0.9, 1.9])

    if batter_key:
        df_b_all = df_neb_bat[df_neb_bat["BatterKey"] == batter_key].copy()
        df_player_all = df_b_all.copy()
        dates_all = pd.to_datetime(df_b_all["Date"], errors="coerce").dropna().dt.date
        present_months = sorted(pd.Series(dates_all).map(lambda d: d.month).unique().tolist())
    else:
        df_b_all = df_neb_bat.iloc[0:0].copy()
        df_player_all = df_b_all.copy()
        present_months = []

    sel_months = colM.multiselect(
        "📅 Months",
        options=present_months,
        format_func=lambda n: MONTH_NAME_BY_NUM.get(n, str(n)),
        default=[],
        key="prof_months",
    )

    if batter_key:
        dser = pd.to_datetime(df_player_all["Date"], errors="coerce").dt.date
        if sel_months:
            dser = dser[pd.Series(dser).map(lambda d: d.month if pd.notna(d) else None).isin(sel_months)]
        present_days = sorted(pd.Series(dser).dropna().map(lambda d: d.day).unique().tolist())
    else:
        present_days = []

    sel_days = colD2.multiselect("📆 Days", options=present_days, default=[], key="prof_days")

    lastN = int(colN.number_input("🎯 Last N games", min_value=0, max_value=50, step=1, value=0, format="%d", key="prof_lastn"))
    hand_choice = colH.radio("⚾ Pitcher Hand", ["Both","LHP","RHP"], index=0, horizontal=True, key="prof_hand")
    
    st.markdown('</div>', unsafe_allow_html=True)

    if sel_months:
        mask_m = pd.to_datetime(df_player_all["Date"], errors="coerce").dt.month.isin(sel_months)
    else:
        mask_m = pd.Series(True, index=df_player_all.index)
    if sel_days:
        mask_d = pd.to_datetime(df_player_all["Date"], errors="coerce").dt.day.isin(sel_days)
    else:
        mask_d = pd.Series(True, index=df_player_all.index)
    df_profiles = df_player_all[mask_m & mask_d].copy()

    if lastN and not df_profiles.empty:
        uniq_dates = pd.to_datetime(df_profiles["Date"], errors="coerce").dt.date.dropna().unique()
        uniq_dates = sorted(uniq_dates)
        last_dates = set(uniq_dates[-lastN:])
        df_profiles = df_profiles[pd.to_datetime(df_profiles["Date"], errors="coerce").dt.date.isin(last_dates)].copy()

    if hand_choice == "LHP":
        df_profiles = df_profiles[df_profiles.get('PitcherThrows').astype(str).str.upper().str.startswith('L')].copy()
    elif hand_choice == "RHP":
        df_profiles = df_profiles[df_profiles.get('PitcherThrows').astype(str).str.upper().str.startswith('R')].copy()

    if batter_key and df_profiles.empty:
        info_message("No rows for the selected filters.")
    elif batter_key:
        professional_divider()
        
        season_label = {
            "2025 season": "2025",
            "2025/26 Scrimmages": "2025/26 Scrimmages",
            "2026 season": "2026",
        }.get(period, "—")
        
        # Player header
        st.markdown(f"""
        <div class="player-header">
            <div class="player-name">{display_name_by_key.get(batter_key,batter_key)}</div>
            <div class="player-subtitle">Split Profiles — {season_label}</div>
        </div>
        """, unsafe_allow_html=True)

        t1_counts, t2_rates, t3_batted = build_profile_tables(df_profiles)

        subsection_header("Summary")
        st.dataframe(themed_styler(t1_counts, nowrap=True), use_container_width=True, hide_index=True)

        professional_divider()

        subsection_header("Plate Discipline")
        st.dataframe(themed_styler(t2_rates, nowrap=True), use_container_width=True, hide_index=True)

        professional_divider()

        subsection_header("Batted Ball Distribution")
        st.dataframe(themed_styler(t3_batted, nowrap=True), use_container_width=True, hide_index=True)

        professional_divider()

        subsection_header("Spray Chart")
        fig_spray = create_profile_spray_chart(df_profiles, display_name_by_key.get(batter_key, batter_key))
        if fig_spray:
            st.pyplot(fig_spray)
            plt.close(fig_spray)
        else:
            info_message("No balls in play with valid location data for the selected filters.")

        professional_divider()

        subsection_header("Hitter Heatmaps")
        st.markdown('<p class="caption-text">Density maps showing contact, whiffs, and hard-hit balls</p>', unsafe_allow_html=True)
        fig_hm = hitter_heatmaps(df_profiles, batter_key)
        if fig_hm:
            st.pyplot(fig_hm)
            plt.close(fig_hm)

# ══════════════════════════════════════════════════════════════════════════════
# RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Rankings":
    section_header("Team Rankings")

    subsection_header("Filters")
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    colM, colD2, colN, colH = st.columns([1.2, 1.2, 0.9, 1.9])

    df_scope = df_neb_bat.copy()

    dates_all = pd.to_datetime(df_scope["Date"], errors="coerce").dropna().dt.date
    present_months = sorted(pd.Series(dates_all).map(lambda d: d.month).unique().tolist())
    sel_months = colM.multiselect(
        "📅 Months",
        options=present_months,
        format_func=lambda n: MONTH_NAME_BY_NUM.get(n, str(n)),
        default=[],
        key="rk_months",
    )

    dser = pd.to_datetime(df_scope["Date"], errors="coerce").dt.date
    if sel_months:
        dser = dser[pd.Series(dser).map(lambda d: d.month if pd.notna(d) else None).isin(sel_months)]
    present_days = sorted(pd.Series(dser).dropna().map(lambda d: d.day).unique().tolist())
    sel_days = colD2.multiselect("📆 Days", options=present_days, default=[], key="rk_days")

    lastN = int(colN.number_input("🎯 Last N games", min_value=0, max_value=50, step=1, value=0, format="%d", key="rk_lastn"))
    hand_choice = colH.radio("⚾ Pitcher Hand", ["Both","LHP","RHP"], index=0, horizontal=True, key="rk_hand")
    
    st.markdown('</div>', unsafe_allow_html=True)

    if sel_months:
        mask_m = pd.to_datetime(df_scope["Date"], errors="coerce").dt.month.isin(sel_months)
    else:
        mask_m = pd.Series(True, index=df_scope.index)
    if sel_days:
        mask_d = pd.to_datetime(df_scope["Date"], errors="coerce").dt.day.isin(sel_days)
    else:
        mask_d = pd.Series(True, index=df_scope.index)
    df_scope = df_scope[mask_m & mask_d].copy()

    if lastN and not df_scope.empty:
        uniq_dates = pd.to_datetime(df_scope["Date"], errors="coerce").dt.date.dropna().unique()
        uniq_dates = sorted(uniq_dates)
        last_dates = set(uniq_dates[-lastN:])
        df_scope = df_scope[pd.to_datetime(df_scope["Date"], errors="coerce").dt.date.isin(last_dates)].copy()

    if hand_choice == "LHP":
        df_scope = df_scope[df_scope.get('PitcherThrows').astype(str).str.upper().str.startswith('L')].copy()
    elif hand_choice == "RHP":
        df_scope = df_scope[df_scope.get('PitcherThrows').astype(str).str.upper().str.startswith('R')].copy()

    if df_scope.empty:
        info_message("No rows for the selected filters.")
        st.stop()

    professional_divider()

    rankings_df = build_rankings_numeric(df_scope, display_name_by_key)
    min_pa = int(st.number_input("🎯 Minimum PA Filter", min_value=0, value=0, step=1, key="rk_min_pa"))
    if min_pa > 0:
        rankings_df = rankings_df[rankings_df["PA"] >= min_pa]

    styled = style_rankings(rankings_df)

    subsection_header("Player Rankings")
    st.markdown('<p class="caption-text">Green = team leader | Red = team last place</p>', unsafe_allow_html=True)
    
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=520
    )
    
    if period == "2025/26 Scrimmages":
        professional_divider()
        subsection_header("Complete Fall Scrimmage Statistics")
        st.markdown('<p class="caption-text">Full season stats for all players during fall scrimmages</p>', unsafe_allow_html=True)
        
        min_pa_complete = int(st.number_input("🎯 Min PA for Complete Table", min_value=0, value=10, step=1, key="complete_min_pa"))
        complete_rankings = rankings_df.copy()
        if min_pa_complete > 0:
            complete_rankings = complete_rankings[complete_rankings["PA"] >= min_pa_complete]
        
        styled_complete = style_rankings(complete_rankings)
        st.dataframe(
            styled_complete,
            use_container_width=True,
            hide_index=True,
            height=520
        )

# ══════════════════════════════════════════════════════════════════════════════
# FALL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
else:
    if period != "2025/26 Scrimmages":
        info_message("Please select '2025/26 Scrimmages' from the Time Period dropdown to view Fall Summary.")
        st.stop()
    
    section_header("Fall 2025 Performance Summary")
    
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        batter_key = st.selectbox(
            "👤 Select Player",
            options=batters_keys,
            index=0,
            format_func=lambda k: display_name_by_key.get(k, k),
            key="fall_player"
        )
    
    if not batter_key:
        info_message("Select a player to view their fall summary.")
        st.stop()
    
    df_player_fall = df_neb_bat[df_neb_bat["BatterKey"] == batter_key].copy()
    
    if df_player_fall.empty:
        warning_message(f"No fall scrimmage data found for {display_name_by_key.get(batter_key, batter_key)}.")
        st.stop()
    
    player_display = display_name_by_key.get(batter_key, batter_key)
    
    # Player header
    st.markdown(f"""
    <div class="player-header">
        <div class="player-name">{player_display}</div>
        <div class="player-subtitle">Fall 2025 Performance Summary</div>
    </div>
    """, unsafe_allow_html=True)
    
    professional_divider()
    
    player_stats = _compute_split_core(df_player_fall)
    
    # Color guide
    st.markdown("""
    <div class="color-guide">
        <p style='margin: 0 0 5px 0; font-size: 14px; color: #666;'>
            <strong>Color Guide:</strong> 
            <span style='background-color: #28a745; color: white; padding: 2px 8px; border-radius: 3px; margin: 0 4px;'>Green</span> 
            = Above D1 Average | 
            <span style='background-color: #ffffff; color: black; padding: 2px 8px; border-radius: 3px; border: 1px solid #ddd; margin: 0 4px;'>White</span> 
            = Near D1 Average | 
            <span style='background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 3px; margin: 0 4px;'>Red</span> 
            = Below D1 Average
        </p>
        <p style='margin: 5px 0 0 0; font-size: 12px; color: #888; font-style: italic;'>
            Note: PA, Hits, Doubles, Triples, Home Runs, Avg Launch Angle, Swing%, and Z-Swing% are not color-coded
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # SECTION 1: OVERALL PERFORMANCE
    subsection_header("Overall Performance")
    
    perf_data_1 = pd.DataFrame({
        'Metric': ['AVG', 'OBP', 'SLG', 'OPS'],
        'Value': [
            f"{player_stats['AVG']:.3f}",
            f"{player_stats['OBP']:.3f}",
            f"{player_stats['SLG']:.3f}",
            f"{player_stats['OPS']:.3f}"
        ]
    })
    
    perf_data_2 = pd.DataFrame({
        'Metric': ['wOBA', 'K%', 'BB%', 'PA'],
        'Value': [
            f"{player_stats['wOBA']:.3f}" if pd.notna(player_stats['wOBA']) else "—",
            f"{(player_stats['SO']/player_stats['PA']*100):.1f}%" if player_stats['PA'] > 0 else "—",
            f"{(player_stats['BB']/player_stats['PA']*100):.1f}%" if player_stats['PA'] > 0 else "—",
            f"{player_stats['PA']}"
        ]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Batting Statistics</div>', unsafe_allow_html=True)
        st.dataframe(
            style_performance_table(perf_data_1),
            use_container_width=True,
            height=200
        )
    
    with col2:
        st.markdown('<div class="subsection-header">Advanced Metrics</div>', unsafe_allow_html=True)
        st.dataframe(
            style_performance_table(perf_data_2),
            use_container_width=True,
            height=200
        )
    
    # Extra Base Hits
    st.markdown('<div class="subsection-header">Extra Base Hits</div>', unsafe_allow_html=True)
    
    xbh_data = pd.DataFrame({
        'Metric': ['Hits', 'Doubles', 'Triples', 'Home Runs'],
        'Value': [
            f"{player_stats['Hits']}",
            f"{player_stats['2B']}",
            f"{player_stats['3B']}",
            f"{player_stats['HR']}"
        ]
    })
    
    st.dataframe(
        themed_styler(xbh_data, nowrap=True),
        use_container_width=True,
        height=200
    )
    
    professional_divider()
    
    # SECTION 2: BATTED BALL QUALITY
    subsection_header("Batted Ball Quality")
    
    bb_data = pd.DataFrame({
        'Metric': ['Avg Exit Velocity', 'Max Exit Velocity', 'Avg Launch Angle', 'Hard Hit%', 'Barrel%'],
        'Value': [
            f"{player_stats['Avg EV']:.1f} mph" if pd.notna(player_stats['Avg EV']) else "—",
            f"{player_stats['Max EV']:.1f} mph" if pd.notna(player_stats['Max EV']) else "—",
            f"{player_stats['Avg LA']:.1f}°" if pd.notna(player_stats['Avg LA']) else "—",
            f"{player_stats['HardHit%']:.1f}%" if pd.notna(player_stats['HardHit%']) else "—",
            f"{player_stats['Barrel%']:.1f}%" if pd.notna(player_stats['Barrel%']) else "—"
        ]
    })
    
    st.dataframe(
        style_performance_table(bb_data),
        use_container_width=True,
        height=230
    )
    
    professional_divider()
    
    # SECTION 3: PLATE DISCIPLINE
    subsection_header("Plate Discipline")
    
    pd_data = pd.DataFrame({
        'Metric': ['Swing%', 'Whiff%', 'Chase%', 'Z-Swing%', 'Z-Contact%', 'Z-Whiff%'],
        'Value': [
            f"{player_stats['Swing%']:.1f}%" if pd.notna(player_stats['Swing%']) else "—",
            f"{player_stats['Whiff%']:.1f}%" if pd.notna(player_stats['Whiff%']) else "—",
            f"{player_stats['Chase%']:.1f}%" if pd.notna(player_stats['Chase%']) else "—",
            f"{player_stats['ZSwing%']:.1f}%" if pd.notna(player_stats['ZSwing%']) else "—",
            f"{player_stats['ZContact%']:.1f}%" if pd.notna(player_stats['ZContact%']) else "—",
            f"{player_stats['ZWhiff%']:.1f}%" if pd.notna(player_stats['ZWhiff%']) else "—"
        ]
    })
    
    st.dataframe(
        style_performance_table(pd_data),
        use_container_width=True,
        height=260
    )
    
    professional_divider()
    
    # SECTION 4: GAME-BY-GAME PERFORMANCE
    subsection_header("Game-by-Game Performance")
    
    game_table = create_game_by_game_table(df_player_fall)
    if not game_table.empty:
        st.dataframe(
            themed_styler(game_table, nowrap=False),
            use_container_width=True,
            height=400
        )
    else:
        info_message("No game-by-game data available.")
    
    professional_divider()
    
    # SECTION 5: PERFORMANCE SPLITS
    subsection_header("Performance Splits")
    
    split_type = st.selectbox(
        "Select Split Type",
        options=["vs Pitch Type", "vs Pitcher Handedness", "By Count Situation"],
        index=0
    )
    
    if split_type == "vs Pitch Type":
        splits_table = create_pitch_type_splits(df_player_fall)
    elif split_type == "vs Pitcher Handedness":
        splits_table = create_handedness_splits(df_player_fall)
    else:
        splits_table = create_count_splits(df_player_fall)
    
    if not splits_table.empty:
        st.dataframe(
            themed_styler(splits_table, nowrap=False),
            use_container_width=True,
            height=300
        )
    else:
        info_message("No split data available.")
    
    professional_divider()
    
    # SECTION 6: VISUALIZATIONS
    subsection_header("Visualizations")
    
    tab1, tab2 = st.tabs(["📊 Spray Chart", "🔥 Heatmaps"])
    
    with tab1:
        fig_spray = create_profile_spray_chart(df_player_fall, player_display)
        if fig_spray:
            st.pyplot(fig_spray)
            plt.close(fig_spray)
        else:
            info_message("No balls in play with valid location data for fall scrimmages.")
    
    with tab2:
        fig_hm = hitter_heatmaps(df_player_fall, batter_key)
        if fig_hm:
            st.pyplot(fig_hm)
            plt.close(fig_hm)
        else:
            info_message("Not enough data for heatmaps.")
    
    professional_divider()
    
    # SECTION 7: PROGRESS TRACKER
    subsection_header("Progress Tracker")
    st.markdown('<p class="caption-text">Track performance metrics across the fall season</p>', unsafe_allow_html=True)
    
    metric_choice = st.selectbox(
        "Select Metric to Track",
        options=['OPS', 'AVG', 'wOBA', 'HardHit%', 'Barrel%', 'Whiff%'],
        index=0
    )
    
    fig_progress = create_progress_chart(df_player_fall, metric=metric_choice)
    if fig_progress:
        st.pyplot(fig_progress)
        plt.close(fig_progress)
    else:
        info_message("Not enough games to show progression chart.")

# Footer
professional_divider()
st.markdown(f"""
<div style='text-align: center; color: #999; font-size: 12px; padding: 20px 0;'>
    Nebraska Baseball Hitter Analytics Platform • {period}<br>
    Powered by Trackman Data
</div>
""", unsafe_allow_html=True)
