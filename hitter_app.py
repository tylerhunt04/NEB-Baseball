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
# CONFIG & THEME
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nebraska Hitter Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Color Palette - Nebraska Brand Colors
HUSKER_RED = "#E41C38"
CREAM = "#FEFDFA"
DARK_GRAY = "#1C1C1C"
LIGHT_GRAY = "#F5F5F5"
ACCENT_GRAY = "#58595B"
WHITE = "#FFFFFF"

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES — clean, light, single accent color
# ──────────────────────────────────────────────────────────────────────────────
BORDER = "#E5E7EB"
MUTED  = "#6B7280"
INK    = "#111827"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: #FAFAFA; }
.block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1400px; }

/* Hide Streamlit chrome */
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Typography */
h1, h2, h3, h4 { color: #111827; font-weight: 700; letter-spacing: -0.01em; }
h1 { font-size: 1.9rem; }
h2 { font-size: 1.5rem; margin-top: 0.5rem; }
h3 { font-size: 1.1rem; margin-top: 1.5rem; }
h4 { font-size: 0.95rem; color: #374151; }

hr { margin: 28px 0; border: none; border-top: 1px solid #E5E7EB; }

/* Sidebar */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
[data-testid="stSidebar"] h3 {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: #6B7280; margin: 0.25rem 0 0.5rem 0; font-weight: 600;
}
[data-testid="stSidebar"] hr { margin: 18px 0; }

/* Buttons (generic) */
.stButton > button {
    background: #FFFFFF; color: #111827; border: 1px solid #D1D5DB;
    border-radius: 6px; font-weight: 500; box-shadow: none;
}
.stButton > button:hover { border-color: #E41C38; color: #E41C38; }

/* Top navigation (button row) */
.nav-row [data-testid="stHorizontalBlock"] { gap: 0.4rem; }
[data-testid="stHorizontalBlock"] .stButton > button {
    background: transparent; border: 1px solid transparent; color: #4B5563;
    font-size: 0.85rem; font-weight: 500; padding: 8px 6px; width: 100%;
    border-radius: 6px;
}
[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #F3F4F6; color: #111827; border-color: transparent;
}

/* Native tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E5E7EB; }
.stTabs [data-baseweb="tab"] {
    padding: 8px 16px; background: transparent; font-weight: 500; color: #6B7280;
}
.stTabs [aria-selected="true"] { color: #E41C38; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #E41C38; }

/* Alerts */
.stAlert { border-radius: 8px; }

/* st.table */
[data-testid="stTable"] table { border: 1px solid #E5E7EB; border-radius: 8px; overflow: hidden; }
[data-testid="stTable"] td { font-size: 0.88rem; padding: 8px 12px; border-color: #F3F4F6; }

/* ── Reusable components ───────────────────────────── */
.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 0 14px 0; margin-bottom: 8px; border-bottom: 3px solid #E41C38;
}
.app-header .brand { display: flex; align-items: center; gap: 14px; }
.app-header img { height: 44px; width: auto; border-radius: 4px; }
.app-header .title { font-size: 1.35rem; font-weight: 700; color: #111827; line-height: 1.1; }
.app-header .subtitle { font-size: 0.82rem; color: #6B7280; margin-top: 2px; }
.app-header .period {
    font-size: 0.78rem; font-weight: 600; color: #E41C38; background: #FDECEF;
    padding: 4px 10px; border-radius: 999px;
}

.page-title { font-size: 1.5rem; font-weight: 700; color: #111827; margin: 8px 0 2px 0; }
.page-sub { font-size: 0.9rem; color: #6B7280; margin-bottom: 18px; }

.section-title {
    font-size: 0.78rem; font-weight: 600; color: #374151; text-transform: uppercase;
    letter-spacing: 0.08em; margin: 26px 0 10px 0; padding-bottom: 6px;
    border-bottom: 1px solid #E5E7EB;
}

.player-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-left: 4px solid #E41C38;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 18px;
}
.player-card .name { font-size: 1.4rem; font-weight: 700; color: #111827; }
.player-card .meta { font-size: 0.85rem; color: #6B7280; margin-top: 2px; }

.stat-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px;
    padding: 12px 8px; text-align: center;
}
.stat-card .lbl {
    font-size: 0.68rem; color: #6B7280; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 4px;
}
.stat-card .val { font-size: 1.3rem; font-weight: 700; color: #111827; line-height: 1.1; }
.stat-card.hl { border-top: 3px solid #E41C38; }
.stat-card.hl .val { color: #E41C38; }

.leader-card {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px;
    padding: 12px 14px; margin-bottom: 8px; display: flex; align-items: center; gap: 14px;
}
.leader-card .rank {
    font-size: 0.75rem; font-weight: 700; color: #9CA3AF; width: 22px;
}
.leader-card .big { font-size: 1.35rem; font-weight: 700; color: #111827; min-width: 74px; }
.leader-card .big small { font-size: 0.75rem; color: #6B7280; font-weight: 500; margin-left: 2px; }
.leader-card .who { font-size: 0.9rem; font-weight: 600; color: #111827; }
.leader-card .ctx { font-size: 0.76rem; color: #6B7280; }

.legend-note {
    background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 18px; font-size: 0.85rem; color: #4B5563;
}
.legend-note .chip { display: inline-block; padding: 1px 8px; border-radius: 4px; margin: 0 4px; font-size: 0.78rem; }
.legend-note .fine { font-size: 0.76rem; color: #9CA3AF; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# Matplotlib defaults so every chart shares the same clean look
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#D1D5DB",
    "axes.labelcolor": "#374151",
    "axes.titleweight": "bold",
    "axes.titlecolor": "#111827",
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

TABLE_HEADER_PROPS = (
    "background-color: #F3F4F6; color: #374151; font-weight: 600; "
    "font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; "
    "white-space: nowrap; border-bottom: 1px solid #E5E7EB;"
)

# ── Small HTML helpers used throughout the app ───────────────────────────────
def section_title(text: str):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)

def page_title(title: str, subtitle: str = ""):
    sub = f"<div class='page-sub'>{subtitle}</div>" if subtitle else "<div style='height:12px'></div>"
    st.markdown(f"<div class='page-title'>{title}</div>{sub}", unsafe_allow_html=True)

def player_card(name: str, meta: str = ""):
    m = f"<div class='meta'>{meta}</div>" if meta else ""
    st.markdown(f"<div class='player-card'><div class='name'>{name}</div>{m}</div>",
                unsafe_allow_html=True)

def stat_card_html(label, value, highlight=False) -> str:
    cls = "stat-card hl" if highlight else "stat-card"
    return f"<div class='{cls}'><div class='lbl'>{label}</div><div class='val'>{value}</div></div>"

def stat_row(stats: dict, keys: list, highlight: set = frozenset()):
    cols = st.columns(len(keys))
    for c, k in zip(cols, keys):
        c.markdown(stat_card_html(k, stats.get(k, "—"), k in highlight), unsafe_allow_html=True)

def leader_card_html(rank, big, unit, who, ctx) -> str:
    return (f"<div class='leader-card'><div class='rank'>#{rank}</div>"
            f"<div class='big'>{big}<small>{unit}</small></div>"
            f"<div><div class='who'>{who}</div><div class='ctx'>{ctx}</div></div></div>")

# Default data paths per period
DATA_PATH_2025   = "B10C25_hitter_app_columns.csv"
DATA_PATH_SCRIM  = "Scrimmage(27).csv"
DATA_PATH_2026   = "_AH.csv"
DATA_PATH_SCRIM_2627 = "fall_26.csv"      # ← new: 2026/27 scrimmages
DATA_PATH_2027   = "_AH_2027.csv"                 # ← new: 2027 season
DATA_PATH_NCAA   = "2026_reg_season.parquet"  # ← swap in your actual filename here

PROB_LOOKUP_PATH = "EV_LA_probabilities.csv"

BANNER_CANDIDATES = [
    "NebraskaChampions.jpg",
    "/mnt/data/NebraskaChampions.jpg",
    "/mnt/user-data/uploads/1768490581164_image.png"
]

# Big Ten / opponents pretty names
TEAM_NAME_MAP = {
    # Big Ten
    "ILL_ILL": "Illinois",
    "MIC_SPA": "Michigan State",
    "UCLA": "UCLA",
    "IOWA_HAW": "Iowa",
    "IU": "Indiana",
    "IND_HOS": "Indiana",
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
    # Non-conference 2026 schedule
    "AUB_TIG": "Auburn",
    "AUB": "Auburn",
    "LOU_CAR": "Louisville",
    "LOU": "Louisville",
    "KAN_WIL": "Kansas State",
    "KAN_STA": "Kansas State",
    "KSU": "Kansas State",
    "FSU_SEM": "Florida State",
    "FLO_SEM": "Florida State",
    "FSU": "Florida State",
    "UNO_MAV": "Omaha",
    "OMA_MAV": "Omaha",
    "OMA": "Omaha",
    "SDU_JAC": "South Dakota State",
    "SDA_JAC": "South Dakota State",
    "SDSU": "South Dakota State",
    "NDU_BIS": "North Dakota State",
    "NDA_BIS": "North Dakota State",
    "NDSU": "North Dakota State",
    "MAI_BLA": "Maine",
    "UME_BLA": "Maine",
    "MAINE": "Maine",
    "CRE_JAY": "Creighton",
    "CRE": "Creighton",
    "KAN_JAY": "Kansas",
    "KU_JAY": "Kansas",
    "KAN": "Kansas",
    "WIC_SHO": "Wichita State",
    "WIC": "Wichita State",
    # Desert Invitational common opponents
    "ARI_CAT": "Arizona",
    "ARI_SUN": "Arizona State",
    "TCU_HOR": "TCU",
    "TCU": "TCU",
    "TEX_LON": "Texas",
    "TEX": "Texas",
    "OKL_SOO": "Oklahoma",
    "OKL": "Oklahoma",
    "OKS_COW": "Oklahoma State",
}

# Maps a frozenset of opponent codes to a tournament/event display name
TOURNAMENT_NAME_MAP = {
    frozenset({"LOU_CAR", "FSU_SEM", "KAN_WIL"}): "Globe Life Field",
    frozenset({"LOU_CAR", "FLO_SEM", "KAN_WIL"}): "Globe Life Field",
    frozenset({"LOU", "FSU", "KAN_WIL"}): "Globe Life Field",
}

# D1 AVERAGES FOR COMPARISON
D1_AVERAGES = {
    "ZWhiff%": 15.7,
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
    if pd.isna(value) or stat_name not in D1_AVERAGES:
        return "#ffffff"
    
    avg = D1_AVERAGES[stat_name]
    
    if stat_name in ["K%", "Whiff%", "Chase%", "ZWhiff%"]:
        diff_pct = (avg - value) / avg
    else:
        diff_pct = (value - avg) / avg
    
    if diff_pct >= 0.20:
        return "#BBF7D0"
    elif diff_pct >= 0.10:
        return "#D1FAE5"
    elif diff_pct >= 0.05:
        return "#ECFDF5"
    elif diff_pct >= -0.05:
        return "#ffffff"
    elif diff_pct >= -0.10:
        return "#FEF2F2"
    elif diff_pct >= -0.20:
        return "#FEE2E2"
    else:
        return "#FECACA"

def style_performance_table(df: pd.DataFrame, stat_name_col='Metric') -> pd.DataFrame:
    def apply_color(row):
        stat_name = row[stat_name_col]
        value_str = row['Value']
        
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
        
        d1_key = stat_mapping.get(stat_name, stat_name)
        
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
        
        text_color = "#111827"
        
        return [f'background-color: {bg_color}; color: {text_color}' if col == 'Value' else '' 
                for col in row.index]
    
    styled = df.style.apply(apply_color, axis=1)
    
    header_props = TABLE_HEADER_PROPS
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
        out = pd.to_datetime(s, errors="coerce", utc=False)
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

SOFT_RYG = colors.LinearSegmentedColormap.from_list(
    "soft_ryg", ["#FECACA", "#FFFFFF", "#BBF7D0"], N=256
)

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
    header_props = TABLE_HEADER_PROPS
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

def render_nb_banner(title="Nebraska Baseball", subtitle="Hitter Analytics", period_label=""):
    """Compact app bar: title on the left, active time period on the right."""
    pill = f"<span class='period'>{period_label}</span>" if period_label else ""
    st.markdown(
        f"""
        <div class="app-header">
          <div class="brand">
            <div>
              <div class="title">{title}</div>
              <div class="subtitle">{subtitle}</div>
            </div>
          </div>
          {pill}
        </div>
        """,
        unsafe_allow_html=True,
    )

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
WOBAC_2025 = {
    "wBB": 0.693, "wHBP": 0.723,
    "w1B": 0.883, "w2B": 1.253, "w3B": 1.585, "wHR": 2.037,
    "wOBAScale": 1.23,
    "lg_wOBA": 0.314,
}

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
    
    stats_by_date = []
    for i, date in enumerate(dates):
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
    
    fig, ax = plt.subplots(figsize=(11, 4.5))
    
    metric_values = df_stats[metric].values
    dates_plot = range(len(dates))
    
    ax.plot(dates_plot, metric_values, marker='o', linewidth=2.5, 
            markersize=8, color=HUSKER_RED, label=metric)
    
    final_val = metric_values[-1]
    ax.axhline(y=final_val, color='#9CA3AF', linestyle='--', linewidth=1, 
               alpha=0.7, label=f'Final: {final_val:.3f}')
    
    ax.set_xlabel('Game', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'{metric} — cumulative by game', fontsize=13, loc='left', pad=12)
    ax.set_xticks(dates_plot)
    ax.set_xticklabels([f"Game {i+1}" for i in dates_plot], rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, axis='y', color='#F3F4F6', linewidth=1)
    
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
        **{k: core_total[k] for k in ["Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]}
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
                **{k: np.nan for k in ["Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]}
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
            **{k: core[k] for k in ["Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]}
        })

    t1 = pd.DataFrame(
        t1_rows,
        columns=["Split","PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS","wOBA","xwOBA"]
    )
    t2 = pd.DataFrame(
        t2_rows,
        columns=["Split","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]
    )

    for c in ["AVG","OBP","SLG","OPS","wOBA","xwOBA"]:
        if c in t1.columns:
            t1[c] = t1[c].apply(lambda v: "—" if pd.isna(v) else (f"{float(v):.3f}"[1:] if float(v) < 1.0 else f"{float(v):.3f}"))

    for c in ["Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]:
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
    "ZWhiff%","Chase%" , "Whiff%"
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

    header_props = TABLE_HEADER_PROPS
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
        st.warning("No balls in play found for this game.")
        return None
    
    if 'Bearing' not in inplay.columns or 'Distance' not in inplay.columns:
        st.warning("Missing 'Bearing' or 'Distance' columns in data.")
        return None
    
    inplay['Bearing'] = pd.to_numeric(inplay['Bearing'], errors='coerce')
    inplay['Distance'] = pd.to_numeric(inplay['Distance'], errors='coerce')
    inplay = inplay.dropna(subset=['Bearing', 'Distance'])
    
    if inplay.empty:
        st.warning("No balls in play with valid Bearing and Distance data.")
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
    ax.set_title(f"{batter_display_name} - Spray Chart\n{date_str}", 
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
    
    ax.set_title(f"{batter_display_name} - Spray Chart (All Batted Balls)", 
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
            return {"label": "PA Result: Walk", "klass": "Walk", "k_type": None, "res": None, "ev": None, "tag": None}
        if kor == "Strikeout":
            last_call = str(s_call.iloc[-1]) if not s_call.empty else ""
            if last_call == "StrikeSwinging":
                k_type = "Swinging"
            elif last_call == "StrikeCalled":
                k_type = "Looking"
            else:
                k_type = "Unknown"
            return {"label": f"PA Result: Strikeout ({k_type})", "klass": "K", "k_type": k_type, "res": None, "ev": None, "tag": None}

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
                    return {"label": "PA Result: Walk", "klass": "Walk", "k_type": None, "res": None, "ev": None, "tag": None}
            elif call == "HitByPitch":
                return {"label": "PA Result: HBP", "klass": "Other", "k_type": None, "res": None, "ev": None, "tag": None}

        if strikes >= 3:
            return {"label": f"PA Result: Strikeout ({k_type or 'Unknown'})", "klass": "K", "k_type": k_type or "Unknown",
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
            return {"label": f"PA Result: {' '.join(bits)}", "klass": "InPlay", "k_type": None,
                    "res": res, "ev": float(es) if pd.notna(es) else None, "tag": tag}

        return {"label": "PA Result: —", "klass": "Other", "k_type": None, "res": None, "ev": None, "tag": None}

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
        _inn = int(inn) if pd.notna(inn) and float(inn).is_integer() else inn
        ax.set_title(f"PA {idx+1} | Inning {_inn} {tb}", fontsize=10, fontweight='bold', pad=6)
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
def _skip_title_row(path: str) -> int:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
        if first and not first.startswith("PitchNo") and "," not in first[:20]:
            return 1
    except Exception:
        pass
    return 0

def load_many_csv(paths: list) -> pd.DataFrame:
    dfs = []
    for p in paths:
        skip = _skip_title_row(p)
        try:
            df = pd.read_csv(p, low_memory=False, skiprows=skip)
        except UnicodeDecodeError:
            df = pd.read_csv(p, low_memory=False, encoding="latin-1", skiprows=skip)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    return ensure_date_column(out)

@st.cache_data(show_spinner=True)
def load_single_csv(path: str) -> pd.DataFrame:
    skip = _skip_title_row(path)
    try:
        df = pd.read_csv(path, low_memory=False, skiprows=skip)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1", skiprows=skip)
    return ensure_date_column(df)

def load_for_period(period_label: str, path_2025: str, path_scrim: str, path_2026: str,
                     path_scrim2627: str = "", path_2027: str = "") -> pd.DataFrame:
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
    elif period_label == "2026/27 Scrimmages":
        paths = _expand_paths(path_scrim2627)
        if not paths:
            return pd.DataFrame()
        return load_many_csv(paths)
    elif period_label == "2027 season":
        paths = _expand_paths(path_2027)
        if not paths:
            try:
                return load_single_csv(path_2027)
            except Exception:
                return pd.DataFrame()
        return load_many_csv(paths)
    else:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
st.sidebar.markdown(
    "<div style='font-size:1.05rem;font-weight:700;color:#111827;padding:4px 0 2px 0;'>Nebraska Baseball</div>"
    "<div style='font-size:0.8rem;color:#6B7280;margin-bottom:14px;'>Hitter Analytics</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Data")

# Time Period Selection
period = st.sidebar.selectbox(
    "Time Period",
    options=[
        "2025 season",
        "2025/26 Scrimmages",
        "2026 season",
        "2026/27 Scrimmages",
        "2027 season",
    ],
    index=0
)

# Data paths in sidebar expander
with st.sidebar.expander("Data Paths", expanded=False):
    st.caption("Paste a CSV path, a directory path, or a glob pattern.")
    path_2025      = st.text_input("2025 season path", value=DATA_PATH_2025,  key="path_2025")
    path_scrim     = st.text_input("2025/26 Scrimmages path/pattern", value=DATA_PATH_SCRIM, key="path_scrim")
    path_2026      = st.text_input("2026 season path/pattern", value=DATA_PATH_2026,  key="path_2026")
    path_scrim2627 = st.text_input("2026/27 Scrimmages path/pattern", value=DATA_PATH_SCRIM_2627, key="path_scrim2627")
    path_2027      = st.text_input("2027 season path/pattern", value=DATA_PATH_2027,  key="path_2027")

path_2025      = st.session_state.get("path_2025", DATA_PATH_2025)
path_scrim     = st.session_state.get("path_scrim", DATA_PATH_SCRIM)
path_2026      = st.session_state.get("path_2026", DATA_PATH_2026)
path_scrim2627 = st.session_state.get("path_scrim2627", DATA_PATH_SCRIM_2627)
path_2027      = st.session_state.get("path_2027", DATA_PATH_2027)

render_nb_banner(period_label=period.title())

df_all = load_for_period(period, path_2025, path_scrim, path_2026, path_scrim2627, path_2027)
if df_all.empty:
    st.error(f"No data loaded for '{period}'. Check the path(s) in the sidebar.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_ncaa_parquet(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        pass
    return pd.DataFrame()

df_ncaa = load_ncaa_parquet(DATA_PATH_NCAA)

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

# ══════════════════════════════════════════════════════════════════════════════
# VIEW MODE SELECTOR IN MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
_TABS = [
    "Standard Hitter Report",
    "Profiles & Heatmaps",
    "Season Summary",
    "Catcher Framing",
    "Rankings",
    "Weekend Series",
]
if period in ("2026 season", "2027 season"):
    _TABS.append("Regional Scouting")

# If user was on Regional Scouting and switches period, fall back to first tab
if st.session_state.get("view_mode") == "Regional Scouting" and "Regional Scouting" not in _TABS:
    st.session_state["view_mode"] = _TABS[0]

if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = _TABS[0]

_TAB_LABELS = {
    "Standard Hitter Report": "Game Report",
    "Profiles & Heatmaps":    "Profiles",
    "Season Summary":         "Season Summary",
    "Catcher Framing":        "Catcher Framing",
    "Rankings":               "Rankings",
    "Weekend Series":         "Series",
    "Regional Scouting":      "Scouting",
}

def _set_view(label: str):
    st.session_state["view_mode"] = label

st.markdown("""
<style>
/* active nav item */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: #E41C38 !important; color: #FFFFFF !important;
    border: 1px solid #E41C38 !important; font-weight: 600 !important;
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: #C41830 !important; border-color: #C41830 !important; color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

_tab_cols = st.columns(len(_TABS))
for _ti, _tab_label in enumerate(_TABS):
    with _tab_cols[_ti]:
        st.button(
            _TAB_LABELS.get(_tab_label, _tab_label),
            key=f"nav_{_ti}",
            use_container_width=True,
            type="primary" if st.session_state["view_mode"] == _tab_label else "secondary",
            on_click=_set_view,
            args=(_tab_label,),
        )

st.markdown("<div style='border-bottom:1px solid #E5E7EB;margin:6px 0 18px 0;'></div>",
            unsafe_allow_html=True)

view_mode = st.session_state["view_mode"]

# ══════════════════════════════════════════════════════════════════════════════
# STANDARD HITTER REPORT
# ══════════════════════════════════════════════════════════════════════════════
if view_mode == "Standard Hitter Report":
    page_title("Game Report", "Pitch-by-pitch breakdown and spray chart for a single game")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Player Selection")
    batter_key_std = st.sidebar.selectbox(
        "Select Player",
        options=batters_keys,
        index=0,
        format_func=lambda k: display_name_by_key.get(k, k),
        key="std_player"
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

    selected_date = st.sidebar.selectbox(
        "Game Date",
        options=date_opts,
        format_func=lambda d: date_labels.get(d, format_date_long(d)),
        index=len(date_opts)-1 if date_opts else 0,
        key="std_date"
    ) if date_opts else None

    if batter_key_std and selected_date:
        df_date = df_b_all[df_b_all["DateOnly"] == selected_date].copy()
    else:
        df_date = df_b_all.iloc[0:0].copy()

    batter_display = display_name_by_key.get(batter_key_std, batter_key_std)

    if df_date.empty:
        st.info("Select a player and game date from the sidebar to view the Standard Hitter Report.")
    else:
        fig_std = create_hitter_report(df_date, batter_display, ncols=3)
        if fig_std:
            st.pyplot(fig_std)
        
        section_title("Spray Chart")
        fig_spray = create_spray_chart(df_date, batter_display)
        if fig_spray:
            st.pyplot(fig_spray)
        else:
            st.info("No balls in play with valid location data for this game.")

# ══════════════════════════════════════════════════════════════════════════════
# PROFILES & HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Profiles & Heatmaps":
    page_title("Player Profile", "Filtered production, plate discipline and batted-ball profile")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Player Selection")
    batter_key = st.sidebar.selectbox(
        "Select Player",
        options=batters_keys,
        index=0,
        format_func=lambda k: display_name_by_key.get(k, k),
        key="prof_player"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    if batter_key:
        df_b_all = df_neb_bat[df_neb_bat["BatterKey"] == batter_key].copy()
        df_player_all = df_b_all.copy()
        dates_all = pd.to_datetime(df_b_all["Date"], errors="coerce").dropna().dt.date
        present_months = sorted(pd.Series(dates_all).map(lambda d: d.month).unique().tolist())
    else:
        df_b_all = df_neb_bat.iloc[0:0].copy()
        df_player_all = df_b_all.copy()
        present_months = []

    sel_months = st.sidebar.multiselect(
        "Months",
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

    sel_days = st.sidebar.multiselect("Days", options=present_days, default=[], key="prof_days")
    lastN = int(st.sidebar.number_input("Last N games", min_value=0, max_value=50, step=1, value=0, format="%d", key="prof_lastn"))
    hand_choice = st.sidebar.radio("Pitcher Hand", ["Both","LHP","RHP"], index=0, key="prof_hand")

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
        st.info("No rows for the selected filters. Adjust filters in the sidebar.")
    elif batter_key:
        season_label = {
            "2025 season": "2025",
            "2025/26 Scrimmages": "2025/26 Scrimmages",
            "2026 season": "2026",
            "2026/27 Scrimmages": "2026/27 Scrimmages",
            "2027 season": "2027",
        }.get(period, "—")
        
        _filters = []
        if sel_months: _filters.append(", ".join(MONTH_NAME_BY_NUM.get(m, str(m)) for m in sel_months))
        if sel_days:   _filters.append("Days " + ", ".join(str(d) for d in sel_days))
        if lastN:      _filters.append(f"Last {lastN} games")
        if hand_choice != "Both": _filters.append(f"vs {hand_choice}")
        _meta = season_label + (" · " + " · ".join(_filters) if _filters else " · All games")
        player_card(display_name_by_key.get(batter_key, batter_key), _meta)

        t1_counts, t2_rates, t3_batted = build_profile_tables(df_profiles)

        section_title("Summary Statistics")
        st.table(themed_styler(t1_counts, nowrap=True))

        section_title("Plate Discipline")
        st.table(themed_styler(t2_rates, nowrap=True))

        section_title("Exit Velocity & Barrel by Pitch Type")

        def _ev_barrel_row(grp: pd.DataFrame) -> dict:
            inplay = grp.get("PitchCall", pd.Series(dtype=object)).astype(str).eq("InPlay")
            ev_bip  = pd.to_numeric(grp.loc[inplay, "ExitSpeed"] if inplay.any() else pd.Series(dtype=float), errors="coerce").dropna()
            la_bip  = pd.to_numeric(grp.loc[inplay, "Angle"]     if inplay.any() else pd.Series(dtype=float), errors="coerce").dropna()
            common  = ev_bip.index.intersection(la_bip.index)
            barrels = int(college_barrel_mask(ev_bip.loc[common], la_bip.loc[common]).sum()) if len(common) else 0
            n_bip   = len(common)
            return {
                "HardHit%": round(float((ev_bip >= 95).mean() * 100), 1) if len(ev_bip) else None,
                "Barrel%":  round(barrels / n_bip * 100, 1)              if n_bip   else None,
                "Avg EV":   round(float(ev_bip.mean()), 1)               if len(ev_bip) else None,
                "Max EV":   round(float(ev_bip.max()),  1)               if len(ev_bip) else None,
                "Avg LA":   round(float(la_bip.mean()), 1)               if len(la_bip) else None,
            }

        _pt_col_ev = _pitch_type_col(df_profiles)
        df_profiles["_PG_EV"] = df_profiles[_pt_col_ev].astype(str).map(pitch_group_of)
        _batter_ev_df = df_profiles[df_profiles.get("BatterKey") == batter_key].copy()

        _ev_rows = []
        _overall = _ev_barrel_row(_batter_ev_df)
        _overall["Split"] = "Overall"
        _ev_rows.append(_overall)

        for _grp_name in ["Fastball", "Breaking", "Offspeed"]:
            _sub = _batter_ev_df[_batter_ev_df["_PG_EV"] == _grp_name]
            if not _sub.empty:
                _r = _ev_barrel_row(_sub)
                _r["Split"] = _grp_name
                _ev_rows.append(_r)

        if _ev_rows:
            _ev_tbl = pd.DataFrame(_ev_rows)[["Split","HardHit%","Barrel%","Avg EV","Max EV","Avg LA"]]

            def _fmt_pct(v):
                return f"{v:.1f}%" if v is not None and pd.notna(v) else "—"
            def _fmt_ev(v):
                return f"{v:.1f}" if v is not None and pd.notna(v) else "—"

            _ev_tbl["HardHit%"] = _ev_tbl["HardHit%"].apply(_fmt_pct)
            _ev_tbl["Barrel%"]  = _ev_tbl["Barrel%"].apply(_fmt_pct)
            _ev_tbl["Avg EV"]   = _ev_tbl["Avg EV"].apply(_fmt_ev)
            _ev_tbl["Max EV"]   = _ev_tbl["Max EV"].apply(_fmt_ev)
            _ev_tbl["Avg LA"]   = _ev_tbl["Avg LA"].apply(_fmt_ev)

            st.table(themed_styler(_ev_tbl.rename(columns={"Split": ""}), nowrap=True))

        section_title("Batted Ball Distribution")
        st.table(themed_styler(t3_batted, nowrap=True))

        
        tab1, tab2 = st.tabs(["Spray Chart", "Heatmaps"])
        
        with tab1:
            fig_spray = create_profile_spray_chart(df_profiles, display_name_by_key.get(batter_key, batter_key))
            if fig_spray:
                st.pyplot(fig_spray)
            else:
                st.info("No balls in play with valid location data for the selected filters.")

        with tab2:
            _pt_col = _pitch_type_col(df_profiles)
            df_profiles["_PitchGroup"] = df_profiles[_pt_col].astype(str).map(pitch_group_of)
            _avail_groups = [g for g in ["Fastball", "Offspeed", "Breaking"] if g in df_profiles["_PitchGroup"].values]

            _pitch_opts = ["All Pitch Types"] + _avail_groups
            _sel_pt = st.selectbox(
                "Filter by pitch type",
                options=_pitch_opts,
                index=0,
                key="hm_pitch_type_profiles",
            )

            _df_hm = df_profiles if _sel_pt == "All Pitch Types" else df_profiles[df_profiles["_PitchGroup"].eq(_sel_pt)]

            if _df_hm.empty:
                st.info("No pitches of that type in the current filter.")
            else:
                fig_hm = hitter_heatmaps(_df_hm, batter_key)
                if fig_hm:
                    st.pyplot(fig_hm)
                else:
                    st.info("Not enough data for heatmaps.")

# ══════════════════════════════════════════════════════════════════════════════
# RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Rankings":
    page_title("Team Rankings", "Team totals and individual leaderboard")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    df_scope = df_neb_bat.copy()

    dates_all = pd.to_datetime(df_scope["Date"], errors="coerce").dropna().dt.date
    present_months = sorted(pd.Series(dates_all).map(lambda d: d.month).unique().tolist())
    sel_months = st.sidebar.multiselect(
        "Months",
        options=present_months,
        format_func=lambda n: MONTH_NAME_BY_NUM.get(n, str(n)),
        default=[],
        key="rk_months",
    )

    dser = pd.to_datetime(df_scope["Date"], errors="coerce").dt.date
    if sel_months:
        dser = dser[pd.Series(dser).map(lambda d: d.month if pd.notna(d) else None).isin(sel_months)]
    present_days = sorted(pd.Series(dser).dropna().map(lambda d: d.day).unique().tolist())
    sel_days = st.sidebar.multiselect("Days", options=present_days, default=[], key="rk_days")

    lastN = int(st.sidebar.number_input("Last N games", min_value=0, max_value=50, step=1, value=0, format="%d", key="rk_lastn"))
    hand_choice = st.sidebar.radio("Pitcher Hand", ["Both","LHP","RHP"], index=0, key="rk_hand")
    min_pa = int(st.sidebar.number_input("Minimum PA", min_value=0, value=0, step=1, key="rk_min_pa"))

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
        st.info("No rows for the selected filters. Adjust filters in the sidebar.")
        st.stop()

    rankings_df = build_rankings_numeric(df_scope, display_name_by_key)
    if min_pa > 0:
        rankings_df = rankings_df[rankings_df["PA"] >= min_pa]

    if not df_scope.empty:
        def _team_totals_from_scope(df: pd.DataFrame) -> dict:
            pa_key = [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in df.columns]
            if not pa_key:
                return {}
            if "PitchofPA" in df.columns:
                df_s = df.sort_values(pa_key + ["PitchofPA"])
            else:
                df_s = df.copy()
            pa_last = df_s.groupby(pa_key, dropna=False).last().reset_index()
            play  = pa_last.get("PlayResult", pd.Series(dtype=object)).astype(str)
            korbb = pa_last.get("KorBB",      pd.Series(dtype=object)).astype(str)
            s_call= pa_last.get("PitchCall",  pd.Series(dtype=object)).astype(str)
            is_hit = play.isin(["Single","Double","Triple","HomeRun"])
            is_so  = korbb.eq("Strikeout")
            is_bb  = korbb.eq("Walk")
            is_hbp = s_call.eq("HitByPitch")
            is_sf  = play.str.contains(r"SacrificeFly|SacFly", case=False, regex=True)
            is_sac = play.str.contains(r"SacrificeBunt|SacBunt", case=False, regex=True)
            H  = int(is_hit.sum()); SO = int(is_so.sum())
            BB = int(is_bb.sum()); HBP = int(is_hbp.sum())
            SF = int(is_sf.sum()); HR  = int(play.eq("HomeRun").sum())
            PA = len(pa_last)
            AB = max(PA - BB - HBP - SF - int(is_sac.sum()), 0)
            bases = (play.eq("Single").sum() + 2*play.eq("Double").sum()
                     + 3*play.eq("Triple").sum() + 4*play.eq("HomeRun").sum())
            AVG = H / AB if AB > 0 else np.nan
            OBP = (H+BB+HBP)/(AB+BB+HBP+SF) if (AB+BB+HBP+SF) > 0 else np.nan
            SLG = bases / AB if AB > 0 else np.nan
            OPS = OBP + SLG if pd.notna(OBP) and pd.notna(SLG) else np.nan
            exitv  = pd.to_numeric(df.get("ExitSpeed", pd.Series(dtype=float)), errors="coerce")
            inplay = df.get("PitchCall", pd.Series(dtype=object)).astype(str).eq("InPlay")
            ev_bip = exitv[inplay].dropna()
            hh_pct = float((ev_bip >= 95).mean() * 100) if len(ev_bip) else np.nan
            avg_ev = float(ev_bip.mean()) if len(ev_bip) else np.nan
            return {
                "PA": PA, "AB": AB, "H": H,
                "2B": int(play.eq("Double").sum()),
                "3B": int(play.eq("Triple").sum()),
                "HR": HR, "BB": BB, "SO": SO,
                "AVG": f"{AVG:.3f}" if pd.notna(AVG) else "—",
                "OBP": f"{OBP:.3f}" if pd.notna(OBP) else "—",
                "SLG": f"{SLG:.3f}" if pd.notna(SLG) else "—",
                "OPS": f"{OPS:.3f}" if pd.notna(OPS) else "—",
                "K%": f"{SO/PA*100:.1f}%" if PA > 0 else "—",
                "BB%": f"{BB/PA*100:.1f}%" if PA > 0 else "—",
                "Avg EV": f"{avg_ev:.1f}" if pd.notna(avg_ev) else "—",
                "HardHit%": f"{hh_pct:.1f}%" if pd.notna(hh_pct) else "—",
            }

        t = _team_totals_from_scope(df_scope)
        if t:
            section_title("Team Totals")

            row1 = ["PA","AB","H","2B","3B","HR","BB","SO"]
            row2 = ["AVG","OBP","SLG","OPS","K%","BB%","Avg EV","HardHit%"]
            stat_row(t, row1)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            stat_row(t, row2, highlight={"OPS"})

            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    section_title("Individual Rankings")

    styled = style_rankings(rankings_df)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=600
    )

# ══════════════════════════════════════════════════════════════════════════════
# FALL SUMMARY / SEASON REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Season Summary":
    if period not in ("2025/26 Scrimmages", "2026 season", "2026/27 Scrimmages", "2027 season"):
        st.info("Please select a scrimmage or season period from the Time Period dropdown to view this report.")
        st.stop()

    _report_title = {
        "2025/26 Scrimmages": "Fall 2025 Performance Summary",
        "2026 season":        "2026 Season Performance Report",
        "2026/27 Scrimmages": "Fall 2026 Performance Summary",
        "2027 season":        "2027 Season Performance Report",
    }.get(period, "Season Performance Report")
    page_title(_report_title)
    
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Player Selection")
    batter_key = st.sidebar.selectbox(
        "Select Player",
        options=batters_keys,
        index=0,
        format_func=lambda k: display_name_by_key.get(k, k),
        key="fall_player"
    )
    
    if not batter_key:
        st.info("Select a player from the sidebar to view their report.")
        st.stop()
    
    df_player_fall = df_neb_bat[df_neb_bat["BatterKey"] == batter_key].copy()
    
    if df_player_fall.empty:
        st.warning(f"No data found for {display_name_by_key.get(batter_key, batter_key)}.")
        st.stop()
    
    player_display = display_name_by_key.get(batter_key, batter_key)
    
    player_stats = _compute_split_core(df_player_fall)
    _n_games = pd.to_datetime(df_player_fall["Date"], errors="coerce").dt.date.nunique()
    player_card(player_display, f"{_n_games} games · {player_stats['PA']} PA")

    _pa = player_stats["PA"]
    _headline = {
        "PA":   f"{_pa}",
        "AVG":  fmt_avg3(player_stats["AVG"]),
        "OBP":  fmt_avg3(player_stats["OBP"]),
        "SLG":  fmt_avg3(player_stats["SLG"]),
        "OPS":  fmt_avg3(player_stats["OPS"]),
        "wOBA": fmt_avg3(player_stats["wOBA"]),
        "K%":   f"{player_stats['SO']/_pa*100:.1f}%" if _pa else "—",
        "BB%":  f"{player_stats['BB']/_pa*100:.1f}%" if _pa else "—",
    }
    stat_row(_headline, list(_headline.keys()), highlight={"OPS"})
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='legend-note'>
      <b>Shading vs. D1 average:</b>
      <span class='chip' style='background:#BBF7D0;'>Well above</span>
      <span class='chip' style='background:#ECFDF5;'>Above</span>
      <span class='chip' style='background:#FFFFFF;border:1px solid #E5E7EB;'>Near</span>
      <span class='chip' style='background:#FEE2E2;'>Below</span>
      <span class='chip' style='background:#FECACA;'>Well below</span>
      <div class='fine'>PA, hit counts, launch angle, Swing% and Z-Swing% are not shaded.</div>
    </div>
    """, unsafe_allow_html=True)

    section_title("Overall Performance")
    
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
        section_title("Batting Statistics")
        st.dataframe(
            style_performance_table(perf_data_1),
            use_container_width=True
        )
    
    with col2:
        section_title("Advanced Metrics")
        st.dataframe(
            style_performance_table(perf_data_2),
            use_container_width=True
        )
    
    section_title("Extra Base Hits")
    xbh_data = pd.DataFrame({
        'Metric': ['Hits', 'Doubles', 'Triples', 'Home Runs'],
        'Value': [
            f"{player_stats['Hits']}",
            f"{player_stats['2B']}",
            f"{player_stats['3B']}",
            f"{player_stats['HR']}"
        ]
    })
    
    header_props = TABLE_HEADER_PROPS
    xbh_styled = xbh_data.style.set_table_styles([
        {'selector': 'thead th', 'props': header_props},
        {'selector': 'th.col_heading', 'props': header_props},
        {'selector': 'th', 'props': header_props},
        {'selector': 'td', 'props': 'white-space: nowrap;'}
    ]).hide(axis="index")
    
    st.dataframe(
        xbh_styled,
        use_container_width=True
    )
    
    
    section_title("Batted Ball Quality")
    
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
        use_container_width=True
    )
    
    
    section_title("Plate Discipline")
    
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
        use_container_width=True
    )
    
    
    section_title("Game-by-Game Performance")
    
    game_table = create_game_by_game_table(df_player_fall)
    if not game_table.empty:
        st.dataframe(
            themed_styler(game_table, nowrap=False),
            use_container_width=True
        )
    else:
        st.info("No game-by-game data available.")
    
    
    section_title("Performance Splits")
    
    split_type = st.selectbox(
        "Select Split Type",
        options=["vs Pitch Type", "vs Pitcher Handedness", "By Count Situation"],
        index=0,
        key="season_split_type"
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
            use_container_width=True
        )
    else:
        st.info("No split data available.")
    
    
    section_title("Visualizations")
    
    tab1, tab2 = st.tabs(["Spray Chart", "Heatmaps"])
    
    with tab1:
        fig_spray = create_profile_spray_chart(df_player_fall, player_display)
        if fig_spray:
            st.pyplot(fig_spray)
        else:
            st.info("No balls in play with valid location data.")
    
    with tab2:
        _pt_col2 = _pitch_type_col(df_player_fall)
        df_player_fall["_PitchGroup"] = df_player_fall[_pt_col2].astype(str).map(pitch_group_of)
        _avail_groups2 = [g for g in ["Fastball", "Offspeed", "Breaking"] if g in df_player_fall["_PitchGroup"].values]

        _pitch_opts2 = ["All Pitch Types"] + _avail_groups2
        _sel_pt2 = st.selectbox(
            "Filter by pitch type",
            options=_pitch_opts2,
            index=0,
            key="hm_pitch_type_season",
        )

        _df_hm2 = df_player_fall if _sel_pt2 == "All Pitch Types" else df_player_fall[df_player_fall["_PitchGroup"].eq(_sel_pt2)]

        if _df_hm2.empty:
            st.info("No pitches of that type in the current filter.")
        else:
            fig_hm = hitter_heatmaps(_df_hm2, batter_key)
            if fig_hm:
                st.pyplot(fig_hm)
            else:
                st.info("Not enough data for heatmaps.")
    
    
    section_title("Progress Tracker")
    
    metric_choice = st.selectbox(
        "Select Metric to Track",
        options=['OPS', 'AVG', 'wOBA', 'HardHit%', 'Barrel%', 'Whiff%'],
        index=0,
        key="season_progress_metric"
    )
    
    fig_progress = create_progress_chart(df_player_fall, metric=metric_choice)
    if fig_progress:
        st.pyplot(fig_progress)
    else:
        st.info("Not enough games to show progression chart.")


# ══════════════════════════════════════════════════════════════════════════════
# CATCHER FRAMING
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Catcher Framing":
    import matplotlib.patches as patches

    FRAME_ZONE_X_MIN, FRAME_ZONE_X_MAX = -0.83, 0.83
    FRAME_ZONE_Y_MIN, FRAME_ZONE_Y_MAX = 1.50, 3.50
    FRAME_ZONE_WIDTH  = FRAME_ZONE_X_MAX - FRAME_ZONE_X_MIN
    FRAME_ZONE_HEIGHT = FRAME_ZONE_Y_MAX - FRAME_ZONE_Y_MIN
    FRAME_PLOT_X_LIM  = (-2.2, 2.2)
    FRAME_PLOT_Y_LIM  = (0.8, 4.5)
    FRAME_POS_COLOR   = "#DC2626"
    FRAME_NEG_COLOR   = "#3498db"

    for _col in ["Catcher", "CatcherTeam", "PlateLocSide", "PlateLocHeight", "PitchCall", "Date"]:
        if _col not in df_all.columns:
            df_all[_col] = pd.NA

    neb_catch_df = df_all[df_all["CatcherTeam"].astype(str).str.upper().eq("NEB")].copy()
    neb_catch_df["Catcher"] = neb_catch_df["Catcher"].astype(str).str.strip()
    neb_catch_df = ensure_date_column(neb_catch_df)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Catcher Framing Filters")

    neb_catch_df["DateOnly"] = pd.to_datetime(neb_catch_df["Date"], errors="coerce").dt.date

    all_catch_dates = sorted(neb_catch_df["DateOnly"].dropna().unique())
    all_catch_names = sorted(neb_catch_df["Catcher"].dropna().unique())

    catch_scope = st.sidebar.radio("Report Scope", ["Single Game", "Full Season"], key="catch_scope")

    if catch_scope == "Single Game":
        date_options = [str(d) for d in all_catch_dates]
        if date_options:
            sel_date_str = st.sidebar.selectbox("Select Game Date", date_options,
                                                index=len(date_options) - 1, key="catch_date")
            sel_date = pd.to_datetime(sel_date_str).date()
            catch_view_df = neb_catch_df[neb_catch_df["DateOnly"] == sel_date]
        else:
            catch_view_df = neb_catch_df.copy()
    else:
        catch_view_df = neb_catch_df.copy()

    sel_catchers = st.sidebar.multiselect(
        "Filter Catcher(s)", all_catch_names, default=all_catch_names, key="catch_names"
    )
    if sel_catchers:
        catch_view_df = catch_view_df[catch_view_df["Catcher"].isin(sel_catchers)]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Shadow Zone")
    st.sidebar.caption(
        "Pitches within this buffer outside the strike zone are considered 'frameable'. "
        "Pitches beyond the buffer are excluded from rate calculations."
    )
    FRAME_SHADOW = st.sidebar.slider(
        "Shadow Zone Buffer (ft)", min_value=0.10, max_value=0.60,
        value=0.33, step=0.05, key="catch_shadow",
        help="0.33 ft ≈ 4 inches outside the rulebook zone"
    )

    def _classify_framing(sub, shadow=None):
        if shadow is None:
            shadow = FRAME_SHADOW
        called = sub[sub["PitchCall"].isin(["BallCalled", "StrikeCalled"])].copy()
        side   = pd.to_numeric(called["PlateLocSide"],   errors="coerce")
        height = pd.to_numeric(called["PlateLocHeight"], errors="coerce")
        in_zone = (
            side.between(FRAME_ZONE_X_MIN, FRAME_ZONE_X_MAX) &
            height.between(FRAME_ZONE_Y_MIN, FRAME_ZONE_Y_MAX)
        )
        in_shadow_outside = (
            side.between(FRAME_ZONE_X_MIN - shadow, FRAME_ZONE_X_MAX + shadow) &
            height.between(FRAME_ZONE_Y_MIN - shadow, FRAME_ZONE_Y_MAX + shadow) &
            ~in_zone
        )
        in_shadow_inside = (
            in_zone &
            (
                (side   < FRAME_ZONE_X_MIN + shadow) | (side   > FRAME_ZONE_X_MAX - shadow) |
                (height < FRAME_ZONE_Y_MIN + shadow) | (height > FRAME_ZONE_Y_MAX - shadow)
            )
        )
        called["InZone"]          = in_zone
        called["InShadowOutside"] = in_shadow_outside
        called["InShadowInside"]  = in_shadow_inside
        pos = called[in_shadow_outside & (called["PitchCall"] == "StrikeCalled")]
        neg = called[in_shadow_inside  & (called["PitchCall"] == "BallCalled")]
        frameable_pos_opps = int(in_shadow_outside.sum())
        frameable_neg_opps = int(in_shadow_inside.sum())
        steal_rate    = round(len(pos) / max(frameable_pos_opps, 1) * 100, 1)
        giveback_rate = round(len(neg) / max(frameable_neg_opps, 1) * 100, 1)
        totals = {
            "total_called":       len(called),
            "pos_frames":         len(pos),
            "neg_frames":         len(neg),
            "net_frames":         len(pos) - len(neg),
            "frameable_pos_opps": frameable_pos_opps,
            "frameable_neg_opps": frameable_neg_opps,
            "steal_rate":         steal_rate,
            "giveback_rate":      giveback_rate,
        }
        return pos, neg, totals

    def _draw_zone_plot(pos, neg, ax, shadow=None):
        if shadow is None:
            shadow = FRAME_SHADOW
        ax.set_facecolor("white")
        ax.figure.set_facecolor("white")
        ax.set_xlim(*FRAME_PLOT_X_LIM)
        ax.set_ylim(*FRAME_PLOT_Y_LIM)
        shadow_rect = patches.Rectangle(
            (FRAME_ZONE_X_MIN - shadow, FRAME_ZONE_Y_MIN - shadow),
            FRAME_ZONE_WIDTH + 2 * shadow, FRAME_ZONE_HEIGHT + 2 * shadow,
            linewidth=1.2, edgecolor="#999999", facecolor="#eeeeee", alpha=0.5,
            linestyle="--", zorder=1
        )
        ax.add_patch(shadow_rect)
        zone_rect = patches.Rectangle(
            (FRAME_ZONE_X_MIN, FRAME_ZONE_Y_MIN), FRAME_ZONE_WIDTH, FRAME_ZONE_HEIGHT,
            linewidth=2, edgecolor="black", facecolor="none", zorder=2
        )
        ax.add_patch(zone_rect)
        plate_x = [-0.71, 0, 0.71, 0.71, 0, -0.71]
        plate_y = [0.15, 0.0, 0.15, 0.30, 0.45, 0.30]
        ax.plot(plate_x + [plate_x[0]], plate_y + [plate_y[0]], color="#555555", lw=1.2, alpha=0.6)
        if len(pos):
            ax.scatter(pd.to_numeric(pos["PlateLocSide"], errors="coerce"),
                       pd.to_numeric(pos["PlateLocHeight"], errors="coerce"),
                       c=FRAME_POS_COLOR, s=55, alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.4,
                       label=f"+ Frame ({len(pos)})")
        if len(neg):
            ax.scatter(pd.to_numeric(neg["PlateLocSide"], errors="coerce"),
                       pd.to_numeric(neg["PlateLocHeight"], errors="coerce"),
                       c=FRAME_NEG_COLOR, s=55, alpha=0.85, zorder=3,
                       edgecolors="white", linewidths=0.4,
                       label=f"− Frame ({len(neg)})")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc="upper right", fontsize=7.5, facecolor="white",
                  labelcolor="black", framealpha=0.9, edgecolor="#cccccc")

    page_title("Catcher Framing", "Strikes stolen and given back in the shadow zone")

    if catch_view_df.empty or not sel_catchers:
        st.info("No catcher data available for the selected filters.")
    else:
        section_title("Framing Summary by Catcher")

        summary_rows = []
        for _catcher, _grp in catch_view_df.groupby("Catcher"):
            _pos, _neg, _totals = _classify_framing(_grp)
            summary_rows.append({
                "Catcher":              _catcher,
                "Called Pitches":       _totals["total_called"],
                "+ Frames":             _totals["pos_frames"],
                "− Frames":              _totals["neg_frames"],
                "Net Frames":           _totals["net_frames"],
                "Shadow Opps (+)":      _totals["frameable_pos_opps"],
                "Shadow Opps (−)":      _totals["frameable_neg_opps"],
                "Steal Rate %":         _totals["steal_rate"],
                "Give-back Rate %":     _totals["giveback_rate"],
            })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values("Net Frames", ascending=False)

            def _color_net(val):
                if val > 0:   return "color: #16A34A; font-weight:bold"
                elif val < 0: return "color: #DC2626; font-weight:bold"
                return ""

            def _color_steal(val):
                if val >= 50:  return "color: #16A34A; font-weight:bold"
                elif val < 30: return "color: #DC2626"
                return "color: #D97706"
            def _color_giveback(val):
                if val <= 15:  return "color: #16A34A; font-weight:bold"
                elif val > 30: return "color: #DC2626"
                return "color: #D97706"
            styled_catch = (
                summary_df.style
                .map(_color_net,      subset=["Net Frames"])
                .map(lambda v: "color: #DC2626", subset=["+ Frames"])
                .map(lambda v: "color: #3498db", subset=["− Frames"])
                .map(_color_steal,    subset=["Steal Rate %"])
                .map(_color_giveback, subset=["Give-back Rate %"])
                .format({"Steal Rate %": "{:.1f}", "Give-back Rate %": "{:.1f}"})
                .set_properties(**{"background-color": "white", "color": "black", "border-color": "#dddddd"})
                .set_table_styles([
                    {"selector": "th", "props": [("background-color", "#F3F4F6"),
                                                  ("color", "black"),
                                                  ("font-weight", "bold"),
                                                  ("text-align", "center"),
                                                  ("border-bottom", "2px solid #dddddd")]},
                    {"selector": "td", "props": [("text-align", "center")]},
                ])
                .hide(axis="index")
            )
            st.dataframe(styled_catch, use_container_width=True)

        section_title("Framing Zone Plots")

        catchers_in_view = [c for c in sel_catchers if c in catch_view_df["Catcher"].values]

        if catchers_in_view:
            n_cols = 3
            plot_cols = st.columns(n_cols)

            for _i, _catcher in enumerate(catchers_in_view):
                _cdf = catch_view_df[catch_view_df["Catcher"] == _catcher]
                _pos, _neg, _totals = _classify_framing(_cdf)
                _net = _totals["net_frames"]
                _net_color = "#16A34A" if _net >= 0 else "#DC2626"
                _net_sign  = "+" if _net >= 0 else ""

                with plot_cols[_i % n_cols]:
                    _steal_color = "#16A34A" if _totals["steal_rate"] >= 50 else ("#DC2626" if _totals["steal_rate"] < 30 else "#D97706")
                    _gb_color    = "#16A34A" if _totals["giveback_rate"] <= 15 else ("#DC2626" if _totals["giveback_rate"] > 30 else "#D97706")
                    def _fm(v, lbl, c="#111827"):
                        return (f"<div style='text-align:center;min-width:52px'>"
                                f"<div style='color:{c};font-size:1.15rem;font-weight:700'>{v}</div>"
                                f"<div style='color:#6B7280;font-size:0.68rem;text-transform:uppercase;"
                                f"letter-spacing:0.04em'>{lbl}</div></div>")
                    st.markdown(
                        f"<div class='player-card' style='padding:12px 14px;margin-bottom:8px'>"
                        f"<div class='name' style='font-size:1.05rem'>{_catcher}</div>"
                        f"<div style='display:flex;gap:10px;margin-top:8px;flex-wrap:wrap'>"
                        + _fm(_totals["pos_frames"], "+ Frames", FRAME_POS_COLOR)
                        + _fm(_totals["neg_frames"], "− Frames", FRAME_NEG_COLOR)
                        + _fm(f"{_net_sign}{_net}", "Net", _net_color)
                        + _fm(f"{_totals['steal_rate']}%", "Steal", _steal_color)
                        + _fm(f"{_totals['giveback_rate']}%", "Give-back", _gb_color)
                        + "</div></div>",
                        unsafe_allow_html=True,
                    )

                    _fig, _ax = plt.subplots(figsize=(3.8, 4.2), facecolor="white")
                    _draw_zone_plot(_pos, _neg, _ax, shadow=FRAME_SHADOW)
                    _fig.tight_layout()
                    st.pyplot(_fig)
                    plt.close(_fig)

        with st.expander("How to read this report"):
            st.markdown(f"""
- <span style='color:{FRAME_POS_COLOR}'>●</span> **Positive frame (steal)** — pitch outside the rulebook zone but inside the shadow band, called a strike.
- <span style='color:{FRAME_NEG_COLOR}'>●</span> **Negative frame (give-back)** — pitch inside the zone near the edge, called a ball.
- **Steal rate** — share of outside shadow-band pitches called strikes. Higher is better.
- **Give-back rate** — share of inside shadow-band pitches called balls. Lower is better.

The dashed box on each plot is the shadow zone ({FRAME_SHADOW:.2f} ft buffer). Pitches beyond it are excluded from the rates.
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WEEKEND SERIES SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Weekend Series":
    page_title("Series Summary", "Team and individual production by series")

    def build_series_index(df: pd.DataFrame) -> list[dict]:
        """Return list of dicts: {label, opponents (list), dates}."""
        if df.empty or "DateOnly" not in df.columns:
            return []
        opp_col = "PitcherTeam"
        if opp_col not in df.columns:
            return []

        dated = (
            df[df["BatterTeam"].astype(str).str.upper().eq("NEB")]
            .groupby("DateOnly")[opp_col]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "UNK")
            .reset_index()
            .sort_values("DateOnly")
        )
        if dated.empty:
            return []

        clusters = []
        cur_dates = [dated.iloc[0]["DateOnly"]]
        cur_opps  = [dated.iloc[0][opp_col]]

        for _, row in dated.iloc[1:].iterrows():
            gap = (row["DateOnly"] - cur_dates[-1]).days
            if gap <= 1:
                cur_dates.append(row["DateOnly"])
                cur_opps.append(row[opp_col])
            elif gap == 2:
                unique_so_far = len(set(cur_opps))
                if unique_so_far >= 2 or row[opp_col] not in cur_opps:
                    candidate = set(cur_opps) | {row[opp_col]}
                    is_tournament = any(
                        key <= candidate or candidate <= key or key == candidate
                        for key in TOURNAMENT_NAME_MAP
                    )
                    if is_tournament:
                        cur_dates.append(row["DateOnly"])
                        cur_opps.append(row[opp_col])
                        continue
                clusters.append((cur_dates[:], cur_opps[:]))
                cur_dates = [row["DateOnly"]]
                cur_opps  = [row[opp_col]]
            else:
                clusters.append((cur_dates[:], cur_opps[:]))
                cur_dates = [row["DateOnly"]]
                cur_opps  = [row[opp_col]]
        clusters.append((cur_dates, cur_opps))

        result = []
        for dates, opps in clusters:
            d0, d1 = min(dates), max(dates)
            seen = set()
            opp_names = []
            for code in opps:
                name = TEAM_NAME_MAP.get(str(code), str(code))
                if name not in seen:
                    seen.add(name)
                    opp_names.append(name)
            raw_code_set = frozenset(opps)
            tournament = None
            for key, name in TOURNAMENT_NAME_MAP.items():
                if key <= raw_code_set or raw_code_set <= key or key == raw_code_set:
                    tournament = name
                    break
            display_name = tournament if tournament else " / ".join(opp_names)
            if d0 == d1:
                label = f"{display_name}  –  {format_date_long(d0)}"
            else:
                label = f"{display_name}  –  {d0.strftime('%b %d')} – {format_date_long(d1)}"
            result.append({"label": label, "opponent": display_name, "opponents": opp_names, "dates": dates})
        result.reverse()
        return result

    def build_opponent_combined_entries(series_index: list) -> list:
        """
        For opponents appearing in 2+ separate date-based clusters,
        build a combined 'All Games' entry to prepend to the series list.
        """
        from collections import defaultdict
        opp_to_series = defaultdict(list)

        for s in series_index:
            for opp_name in s["opponents"]:
                opp_to_series[opp_name].append(s)

        combined = []
        for opp_name, clusters in opp_to_series.items():
            if len(clusters) < 2:
                continue
            all_dates = sorted(set(d for c in clusters for d in c["dates"]))
            n_games = len(all_dates)
            label = f"{opp_name} – All {n_games} Games"
            combined.append({
                "label": label,
                "opponent": opp_name,
                "opponents": [opp_name],
                "dates": all_dates,
            })

        combined.sort(key=lambda x: x["opponent"])
        return combined

    if "DateOnly" not in df_neb_bat.columns:
        df_neb_bat["DateOnly"] = pd.to_datetime(df_neb_bat["Date"], errors="coerce").dt.date

    if "DateOnly" not in df_all.columns:
        df_all["DateOnly"] = pd.to_datetime(df_all["Date"], errors="coerce").dt.date

    # Build date-based series, then prepend any combined opponent entries
    series_index = build_series_index(df_neb_bat)
    combined_entries = build_opponent_combined_entries(series_index)
    series_index = combined_entries + series_index

    if not series_index:
        st.info("No series data found. Make sure Nebraska game data is loaded.")
    else:
        series_labels = [s["label"] for s in series_index]
        sel_label = st.selectbox(
            "Select Series",
            options=series_labels,
            index=0,
            key="weekend_series_picker",
        )
        sel_series    = next(s for s in series_index if s["label"] == sel_label)
        sel_dates     = sel_series["dates"]
        sel_opp       = sel_series["opponent"]
        sel_opp_str   = sel_series["opponent"]

        neb_ser = df_neb_bat[df_neb_bat["DateOnly"].isin(sel_dates)].copy()
        all_ser = df_all[df_all["DateOnly"].isin(sel_dates)].copy()

        if neb_ser.empty:
            st.warning("No Nebraska batting data found for this series.")
        else:
            n_games = len(sel_dates)
            game_word = "game" if n_games == 1 else "games"

            date_range = ", ".join(format_date_long(d) for d in sorted(sel_dates))
            player_card(f"Nebraska vs. {sel_opp_str}", f"{n_games} {game_word} · {date_range}")

            section_title("Team Batting")

            def _series_team_stats(df: pd.DataFrame) -> dict:
                pa_key = [c for c in ["GameID","Inning","Top/Bottom","PAofInning"] if c in df.columns]
                if not pa_key:
                    return {}

                if "PitchofPA" in df.columns:
                    df_s = df.sort_values(pa_key + ["PitchofPA"])
                else:
                    df_s = df.copy()

                pa_last = df_s.groupby(pa_key, dropna=False).last().reset_index()

                play  = pa_last.get("PlayResult", pd.Series(dtype=object)).astype(str)
                korbb = pa_last.get("KorBB",      pd.Series(dtype=object)).astype(str)
                s_call= pa_last.get("PitchCall",  pd.Series(dtype=object)).astype(str)

                is_hit = play.isin(["Single","Double","Triple","HomeRun"])
                is_so  = korbb.eq("Strikeout")
                is_bb  = korbb.eq("Walk")
                is_hbp = s_call.eq("HitByPitch")
                is_sf  = play.str.contains(r"SacrificeFly|SacFly", case=False, regex=True)
                is_sac = play.str.contains(r"SacrificeBunt|SacBunt", case=False, regex=True)

                H  = int(is_hit.sum())
                SO = int(is_so.sum())
                BB = int(is_bb.sum())
                HBP= int(is_hbp.sum())
                SF = int(is_sf.sum())
                HR = int(play.eq("HomeRun").sum())
                doubles = int(play.eq("Double").sum())
                triples = int(play.eq("Triple").sum())
                PA = len(pa_last)
                AB = max(PA - BB - HBP - SF - int(is_sac.sum()), 0)

                bases = (play.eq("Single").sum()
                         + 2*play.eq("Double").sum()
                         + 3*play.eq("Triple").sum()
                         + 4*play.eq("HomeRun").sum())

                AVG = H / AB if AB > 0 else np.nan
                OBP = (H+BB+HBP) / (AB+BB+HBP+SF) if (AB+BB+HBP+SF) > 0 else np.nan
                SLG = bases / AB if AB > 0 else np.nan
                OPS = OBP + SLG if pd.notna(OBP) and pd.notna(SLG) else np.nan

                exitv = pd.to_numeric(df.get("ExitSpeed", pd.Series(dtype=float)), errors="coerce")
                inplay = df.get("PitchCall", pd.Series(dtype=object)).astype(str).eq("InPlay")
                ev_bip = exitv[inplay].dropna()
                hh_pct = float((ev_bip >= 95).mean() * 100) if len(ev_bip) else np.nan

                return {
                    "PA": PA, "AB": AB, "H": H, "2B": doubles,
                    "3B": triples, "HR": HR, "BB": BB, "SO": SO,
                    "AVG": f"{AVG:.3f}" if pd.notna(AVG) else "—",
                    "OBP": f"{OBP:.3f}" if pd.notna(OBP) else "—",
                    "SLG": f"{SLG:.3f}" if pd.notna(SLG) else "—",
                    "OPS": f"{OPS:.3f}" if pd.notna(OPS) else "—",
                    "K%": f"{SO/PA*100:.1f}%" if PA > 0 else "—",
                    "BB%": f"{BB/PA*100:.1f}%" if PA > 0 else "—",
                    "HardHit%": f"{hh_pct:.1f}%" if pd.notna(hh_pct) else "—",
                }
            team_stats = _series_team_stats(neb_ser)

            counting = ["PA","AB","H","2B","3B","HR","BB","SO"]
            rate_stats = ["AVG","OBP","SLG","OPS","K%","BB%","HardHit%"]

            stat_row(team_stats, counting)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            stat_row(team_stats, rate_stats, highlight={"OPS"})

            section_title("Series Leaders")
            left_col, mid_col, right_col = st.columns(3, gap="large")

            with left_col:
                st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:10px;'>Top Exit Velocities</div>", unsafe_allow_html=True)
                ev_df = neb_ser[
                    neb_ser["PitchCall"].astype(str).eq("InPlay") &
                    pd.to_numeric(neb_ser["ExitSpeed"], errors="coerce").notna()
                ].copy()
                ev_df["EV"] = pd.to_numeric(ev_df["ExitSpeed"], errors="coerce")
                ev_top3 = (
                    ev_df.nlargest(3, "EV")
                    [["BatterKey", "EV", "DateOnly", "PlayResult"]]
                    .reset_index(drop=True)
                )
                if ev_top3.empty:
                    st.info("No exit velocity data.")
                else:
                    for rank, row in ev_top3.iterrows():
                        result = str(row.get("PlayResult", "")).strip()
                        if not result or result.lower() in ("", "nan", "undefined"):
                            result = "—"
                        st.markdown(
                            leader_card_html(rank + 1, f"{row['EV']:.1f}", "mph", row["BatterKey"],
                                             f"{result} · {format_date_long(row['DateOnly'])}"),
                            unsafe_allow_html=True,
                        )

            with mid_col:
                st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:10px;'>Top Hit Distances</div>", unsafe_allow_html=True)
                dist_df = neb_ser[
                    neb_ser["PitchCall"].astype(str).eq("InPlay") &
                    pd.to_numeric(neb_ser["Distance"], errors="coerce").notna()
                ].copy()
                dist_df["Dist"] = pd.to_numeric(dist_df["Distance"], errors="coerce")
                dist_top3 = (
                    dist_df.nlargest(3, "Dist")
                    [["BatterKey", "Dist", "DateOnly", "PlayResult"]]
                    .reset_index(drop=True)
                )
                if dist_top3.empty:
                    st.info("No distance data.")
                else:
                    for rank, row in dist_top3.iterrows():
                        result = str(row.get("PlayResult", "")).strip()
                        if not result or result.lower() in ("", "nan", "undefined"):
                            result = "—"
                        st.markdown(
                            leader_card_html(rank + 1, f"{row['Dist']:.0f}", "ft", row["BatterKey"],
                                             f"{result} · {format_date_long(row['DateOnly'])}"),
                            unsafe_allow_html=True,
                        )

            with right_col:
                st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#374151;margin-bottom:10px;'>Nebraska Pitch Velocities</div>", unsafe_allow_html=True)
                velo_src = all_ser.copy()
                velo_src["RV"] = pd.to_numeric(velo_src.get("RelSpeed", pd.Series(dtype=float)), errors="coerce")
                velo_src = velo_src[velo_src["RV"].notna()].copy()

                neb_pitch_mask = velo_src.get("PitcherTeam", pd.Series(dtype=str)).astype(str).str.upper().eq("NEB")
                velo_neb = velo_src[neb_pitch_mask]
                if velo_neb.empty:
                    velo_neb = velo_src

                pitch_name_col = next(
                    (c for c in ["Pitcher"] if c in velo_neb.columns), None
                )
                if pitch_name_col:
                    velo_neb = velo_neb.copy()
                    velo_neb["_PitcherDisp"] = velo_neb[pitch_name_col].map(
                        lambda x: normalize_name(str(x)) if pd.notna(x) else "Unknown"
                    )
                else:
                    velo_neb = velo_neb.copy()
                    velo_neb["_PitcherDisp"] = "Unknown"

                pitcher_bests = (
                    velo_neb.groupby("_PitcherDisp")["RV"]
                    .max()
                    .reset_index()
                    .nlargest(3, "RV")
                    .reset_index(drop=True)
                )

                if pitcher_bests.empty:
                    st.info("No pitch velocity data.")
                else:
                    for rank, row in pitcher_bests.iterrows():
                        p_rows = velo_neb[velo_neb["_PitcherDisp"] == row["_PitcherDisp"]].nlargest(3, "RV")
                        top3_str = "  ·  ".join(f"{v:.1f}" for v in p_rows["RV"].tolist())
                        st.markdown(
                            leader_card_html(rank + 1, f"{row['RV']:.1f}", "mph", row["_PitcherDisp"],
                                             f"Top 3: {top3_str}"),
                            unsafe_allow_html=True,
                        )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            section_title("Individual Batter Breakdown")

            def _batter_series_line(df: pd.DataFrame) -> pd.DataFrame:
                pa_key = ["GameID","Inning","Top/Bottom","PAofInning"]
                pa_key = [c for c in pa_key if c in df.columns]
                if not pa_key:
                    return pd.DataFrame()

                if "PitchofPA" in df.columns:
                    df_s = df.sort_values(["BatterKey"] + pa_key + ["PitchofPA"])
                else:
                    df_s = df.copy()

                rows = []
                for bk, grp in df_s.groupby("BatterKey", dropna=False):
                    if not bk:
                        continue

                    pa_last = grp.groupby(pa_key, dropna=False).last().reset_index()
                    PA = len(pa_last)
                    if PA == 0:
                        continue

                    play  = pa_last.get("PlayResult", pd.Series(dtype=object)).astype(str)
                    korbb = pa_last.get("KorBB",      pd.Series(dtype=object)).astype(str)
                    s_call= pa_last.get("PitchCall",  pd.Series(dtype=object)).astype(str)

                    is_hit = play.isin(["Single","Double","Triple","HomeRun"])
                    is_so  = korbb.eq("Strikeout")
                    is_bb  = korbb.eq("Walk")
                    is_hbp = s_call.eq("HitByPitch")
                    is_sf  = play.str.contains(r"SacrificeFly|SacFly", case=False, regex=True)
                    is_sac_bunt = play.str.contains(r"SacrificeBunt|SacBunt", case=False, regex=True)

                    H   = int(is_hit.sum())
                    SO  = int(is_so.sum())
                    BB  = int(is_bb.sum())
                    HBP = int(is_hbp.sum())
                    SF  = int(is_sf.sum())
                    HR  = int(play.eq("HomeRun").sum())
                    AB  = PA - BB - HBP - SF - int(is_sac_bunt.sum())
                    AB  = max(AB, 0)

                    bases = (play.eq("Single").sum()
                             + 2*play.eq("Double").sum()
                             + 3*play.eq("Triple").sum()
                             + 4*play.eq("HomeRun").sum())

                    AVG = H / AB if AB > 0 else np.nan
                    OBP = (H+BB+HBP) / (AB+BB+HBP+SF) if (AB+BB+HBP+SF) > 0 else np.nan
                    SLG = bases / AB if AB > 0 else np.nan
                    OPS = OBP + SLG if pd.notna(OBP) and pd.notna(SLG) else np.nan

                    exitv  = pd.to_numeric(grp.get("ExitSpeed", pd.Series(dtype=float)), errors="coerce")
                    inplay = grp.get("PitchCall", pd.Series(dtype=object)).astype(str).eq("InPlay")
                    ev_bip = exitv[inplay].dropna()
                    avg_ev = float(ev_bip.mean()) if len(ev_bip) else np.nan
                    max_ev = float(ev_bip.max())  if len(ev_bip) else np.nan

                    rows.append({
                        "Batter": bk,
                        "PA": PA, "AB": AB, "H": H, "HR": HR,
                        "BB": BB, "SO": SO,
                        "AVG": round(AVG, 3) if pd.notna(AVG) else np.nan,
                        "OBP": round(OBP, 3) if pd.notna(OBP) else np.nan,
                        "SLG": round(SLG, 3) if pd.notna(SLG) else np.nan,
                        "OPS": round(OPS, 3) if pd.notna(OPS) else np.nan,
                        "Avg EV": round(avg_ev, 1) if pd.notna(avg_ev) else np.nan,
                        "Max EV": round(max_ev, 1) if pd.notna(max_ev) else np.nan,
                    })

                out = pd.DataFrame(rows).sort_values("OPS", ascending=False, na_position="last")
                return out

            batter_breakdown = _batter_series_line(neb_ser)
            if not batter_breakdown.empty:
                fmt_map = {
                    "AVG": "{:.3f}", "OBP": "{:.3f}", "SLG": "{:.3f}", "OPS": "{:.3f}",
                    "Avg EV": "{:.1f}", "Max EV": "{:.1f}",
                }
                st.dataframe(
                    batter_breakdown.style
                    .hide(axis="index")
                    .format(fmt_map, na_rep="—")
                    .background_gradient(subset=["OPS"], cmap=SOFT_RYG, vmin=0.5, vmax=1.2)
                    .background_gradient(subset=["Max EV"], cmap=SOFT_RYG, vmin=75, vmax=105),
                    use_container_width=True,
                )
            else:
                st.info("No batter data available.")


# ══════════════════════════════════════════════════════════════════════════════
# REGIONAL SCOUTING — THEIR PITCHERS vs OUR HITTERS
# ══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Regional Scouting":

    import requests as _requests

    # ── Color constants for this page (blue accent = hitter-side framing) ────
    SC_BLU   = "#2563EB"
    SC_RED   = "#DC2626"
    SC_GRN   = "#16A34A"
    SC_AMB   = "#D97706"
    SC_DARK  = "#0f0f0f"
    SC_D2    = "#161616"
    SC_D3    = "#1d1d1d"
    SC_D4    = "#252525"

    # ── Scouting page CSS (light, matches the rest of the app) ───────────────
    st.markdown("""
    <style>
    .sc-header-card {
        background:#FFFFFF;border:1px solid #E5E7EB;border-left:4px solid #E41C38;
        border-radius:8px;padding:16px 20px;margin-bottom:16px;
    }
    .sc-tag-nb, .sc-tag-opp {
        display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px;margin-right:6px;
    }
    .sc-tag-nb  { background:#FDECEF;color:#E41C38; }
    .sc-tag-opp { background:#F3F4F6;color:#374151; }
    .sc-matchup-names { font-size:1.35rem;font-weight:700;color:#111827;line-height:1.3;margin:8px 0 4px 0; }
    .sc-matchup-sub { font-size:0.82rem;color:#6B7280;margin-bottom:12px; }
    .sc-grade-segs { display:flex;gap:3px;height:6px;margin-bottom:6px; }
    .sc-grade-seg { flex:1;border-radius:2px; }
    .sc-grade-labels { display:flex;justify-content:space-between;align-items:center; }
    .sc-grade-verdict { font-size:0.85rem;font-weight:600;text-align:center;flex:1; }
    .sc-grade-lbl { font-size:0.72rem; }
    .sc-section-title {
        font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#374151;
        margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #E5E7EB;
    }
    .sc-pitch-card {
        background:#FFFFFF;border:1px solid #E5E7EB;border-radius:8px;padding:12px 14px;margin-bottom:10px;
    }
    .sc-pitch-head { display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:9px; }
    .sc-pitch-name { font-size:0.95rem;font-weight:600;color:#111827; }
    .sc-pitch-usage {
        font-size:0.72rem;color:#374151;background:#F3F4F6;padding:2px 8px;border-radius:999px;
        margin-left:8px;vertical-align:middle;
    }
    .sc-pitch-det { font-size:0.78rem;color:#6B7280;margin-top:3px; }
    .sc-mrow { display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px;margin-bottom:8px; }
    .sc-mbox { background:#F9FAFB;border:1px solid #F3F4F6;border-radius:6px;padding:7px 8px; }
    .sc-mlbl { font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px; }
    .sc-mval { font-size:1.1rem;font-weight:700;line-height:1; }
    .sc-mbar { height:3px;border-radius:2px;margin-top:6px; }
    .sc-chip { font-size:0.65rem;padding:1px 6px;border-radius:4px;margin-top:6px;display:inline-block;font-weight:600; }
    .sc-chip-bad  { background:#FEE2E2;color:#B91C1C; }
    .sc-chip-ok   { background:#FEF3C7;color:#92400E; }
    .sc-chip-good { background:#DCFCE7;color:#166534; }
    .sc-pnote { font-size:0.8rem;color:#4B5563;line-height:1.5; }
    .sc-right-card {
        background:#FFFFFF;border:1px solid #E5E7EB;border-radius:8px;padding:14px 16px;margin-bottom:12px;
    }
    .sc-zone-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:3px;width:150px; }
    .sc-zone-cell {
        aspect-ratio:1;border-radius:4px;display:flex;align-items:center;justify-content:center;
        font-size:0.78rem;font-weight:600;border:1px solid #F3F4F6;
    }
    .sc-plt-grid { display:grid;grid-template-columns:1fr 1fr;gap:8px; }
    .sc-plt-card { background:#F9FAFB;border:1px solid #F3F4F6;border-radius:6px;padding:10px 12px; }
    .sc-plt-hdr { font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#6B7280;margin-bottom:6px; }
    .sc-plt-row { display:flex;justify-content:space-between;margin-bottom:4px; }
    .sc-plt-lbl { font-size:0.8rem;color:#6B7280; }
    .sc-plt-val { font-size:0.85rem;font-weight:600;color:#111827; }
    .sc-note-box {
        background:#FFFFFF;border:1px solid #E5E7EB;border-radius:8px;padding:14px 16px;margin-top:12px;
    }
    .sc-note-body {
        background:#F9FAFB;border-left:3px solid #E41C38;border-radius:0 6px 6px 0;
        padding:11px 13px;font-size:0.88rem;color:#374151;line-height:1.65;
    }
    .cr { color:#DC2626; } .ca { color:#D97706; } .cg { color:#16A34A; } .cb { color:#2563EB; }
    </style>
    """, unsafe_allow_html=True)

    # ── Build opponent pitcher profiles from NCAA parquet ────────────────────
    def _ivb_col(df):
        for c in ["InducedVertBreak", "VertBreak", "pfx_z"]:
            if c in df.columns:
                return c
        return None

    def _hb_col(df):
        for c in ["HorzBreak", "pfx_x"]:
            if c in df.columns:
                return c
        return None

    def _build_scout_pitchers(df_ncaa: pd.DataFrame) -> dict:
        if df_ncaa.empty:
            return {}

        pt_col  = next((c for c in ["AutoPitchType","TaggedPitchType","PitchType"] if c in df_ncaa.columns), None)
        ivb_c   = _ivb_col(df_ncaa)
        hb_c    = _hb_col(df_ncaa)

        # Exclude Nebraska pitchers — we want opponents facing our hitters
        team_col = "PitcherTeam" if "PitcherTeam" in df_ncaa.columns else None
        if team_col:
            opp = df_ncaa[~df_ncaa[team_col].astype(str).str.upper().eq("NEB")].copy()
        else:
            opp = df_ncaa.copy()

        if opp.empty or pt_col is None:
            return {}

        result = {}
        group_cols = [c for c in ["Pitcher","PitcherTeam","PitcherThrows"] if c in opp.columns]
        if not group_cols:
            return {}

        for keys, grp in opp.groupby(group_cols, dropna=False):
            if len(grp) < 50:
                continue

            if len(group_cols) == 3:
                p_name, p_team, p_throws = keys
            elif len(group_cols) == 2:
                p_name, p_team = keys; p_throws = "R"
            else:
                p_name = keys; p_team = "UNK"; p_throws = "R"

            p_name   = str(p_name)
            p_team   = str(p_team)
            p_throws = str(p_throws).strip().upper()
            p_hand   = "LHP" if p_throws.startswith("L") else "RHP"

            # Arsenal
            ar = []
            for ptype, pg in grp.groupby(pt_col, dropna=False):
                ptype = str(ptype)
                if ptype.lower() in ("", "nan", "undefined", "other", "unknown"):
                    continue
                if len(pg) < 8:
                    continue
                velo = pd.to_numeric(pg.get("RelSpeed", pd.Series(dtype=float)), errors="coerce").mean()
                ivb  = pd.to_numeric(pg.get(ivb_c, pd.Series(dtype=float)), errors="coerce").mean() if ivb_c else 0.0
                hb   = pd.to_numeric(pg.get(hb_c,  pd.Series(dtype=float)), errors="coerce").mean() if hb_c else 0.0
                if pd.isna(velo):
                    continue
                usage = f"{len(pg)/len(grp)*100:.0f}%"
                display = _pretty_pitch_name(ptype)
                ar.append({
                    "t":   display,
                    "v":   round(float(velo), 1),
                    "ivb": round(float(ivb) if pd.notna(ivb) else 0.0, 1),
                    "hb":  round(float(hb)  if pd.notna(hb)  else 0.0, 1),
                    "u":   usage,
                })

            if not ar:
                continue

            ar.sort(key=lambda x: float(x["u"].replace("%", "")), reverse=True)

            # Zone tendency — where this pitcher locates
            lside = pd.to_numeric(grp.get("PlateLocSide",   pd.Series(dtype=float)), errors="coerce")
            lht   = pd.to_numeric(grp.get("PlateLocHeight", pd.Series(dtype=float)), errors="coerce")
            x_bounds = [(-0.83, -0.277), (-0.277, 0.277), (0.277, 0.83)]
            y_bounds = [(2.83, 3.5), (1.83, 2.83), (1.5, 1.83)]
            zone_counts = []
            for yr, yb in y_bounds:
                for xr in x_bounds:
                    cell = ((lside >= xr[0]) & (lside <= xr[1]) & (lht >= yr) & (lht <= yb))
                    zone_counts.append(int(cell.sum()))
            mx = max(zone_counts) if any(zone_counts) else 1
            zone = [round(c / max(mx, 1) * 3) for c in zone_counts]

            # K/BB proxy
            korbb_s = grp.get("KorBB", pd.Series(dtype=object)).astype(str)
            n_k  = int(korbb_s.eq("Strikeout").sum())
            n_bb = int(korbb_s.eq("Walk").sum())
            kbb  = round(n_k / max(n_bb, 1), 1)

            key = f"{normalize_name(p_name)}_{p_team}".lower().replace(" ", "_").replace(",", "")
            result[key] = {
                "name":     normalize_name(p_name),
                "tm":       p_team,
                "tm_label": TEAM_NAME_MAP.get(p_team, p_team),
                "hand":     p_hand,
                "era":      "—",
                "ip":       "—",
                "kbb":      kbb,
                "ar":       ar,
                "zone":     zone,
            }

        return result

    SCOUT_PITCHERS = _build_scout_pitchers(df_ncaa)

    if not SCOUT_PITCHERS:
        st.warning("No opponent pitcher data found. Make sure the NCAA parquet file is loaded and the path is set correctly in DATA_PATH_NCAA.")
        st.stop()

    PITCH_COLORS = {
        "4-Seam":   "#4a9eff",
        "2-Seam":   "#2ec4b6",
        "Sinker":   "#2ec4b6",
        "Slider":   "#f5a623",
        "Curveball":"#a78bfa",
        "Changeup": "#f97316",
        "Cutter":   "#3adb76",
        "Splitter": "#e879f9",
    }

    # ── Helper: compute Nebraska hitter profile from CSV data ─────────────────
    def _sc_hitter_profile(df_h: "pd.DataFrame", pitcher_hand: str) -> dict:
        """Extract scouting profile stats for one Nebraska hitter."""
        import numpy as _np

        s_call  = df_h.get("PitchCall",    pd.Series(dtype=object)).astype(str)
        play    = df_h.get("PlayResult",   pd.Series(dtype=object)).astype(str)
        korbb   = df_h.get("KorBB",        pd.Series(dtype=object)).astype(str)
        exitv   = pd.to_numeric(df_h.get("ExitSpeed",   pd.Series(dtype=float)), errors="coerce")
        lside   = pd.to_numeric(df_h.get("PlateLocSide",  pd.Series(dtype=float)), errors="coerce")
        lht     = pd.to_numeric(df_h.get("PlateLocHeight",pd.Series(dtype=float)), errors="coerce")
        balls_c = pd.to_numeric(df_h.get("Balls",   pd.Series(dtype=float)), errors="coerce")
        strikes_c = pd.to_numeric(df_h.get("Strikes", pd.Series(dtype=float)), errors="coerce")
        ht_col  = df_h.get("TaggedHitType", pd.Series(dtype=object)).astype(str)
        bearing = pd.to_numeric(df_h.get("Bearing", pd.Series(dtype=float)), errors="coerce")
        pthrows = df_h.get("PitcherThrows", pd.Series(dtype=object)).astype(str)
        bside   = df_h.get("BatterSide", pd.Series(dtype=object)).astype(str)
        pitchofpa = pd.to_numeric(df_h.get("PitchofPA", pd.Series(dtype=float)), errors="coerce")

        is_swing   = s_call.isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"])
        is_whiff   = s_call.eq("StrikeSwinging")
        is_inzone  = lside.between(-0.83, 0.83) & lht.between(1.5, 3.5)
        is_inplay  = s_call.eq("InPlay")

        n_pa   = int((pitchofpa == 1).sum())
        n_so   = int(korbb.eq("Strikeout").sum())
        n_bb   = int(korbb.eq("Walk").sum())
        k_pct  = round(n_so / n_pa * 100, 1) if n_pa else 0.0
        bb_pct = round(n_bb / n_pa * 100, 1) if n_pa else 0.0

        # Pull %
        def _is_pull(row_side, row_bear):
            if not _np.isfinite(row_bear):
                return False
            side = str(row_side).strip().upper()[:1]
            if side == "R":
                return row_bear < -15
            elif side == "L":
                return row_bear > 15
            return False

        inplay_mask = is_inplay.values
        if inplay_mask.sum() > 0:
            inplay_df = df_h[inplay_mask].copy()
            bear_vals = pd.to_numeric(inplay_df.get("Bearing", pd.Series(dtype=float)), errors="coerce")
            side_vals = inplay_df.get("BatterSide", pd.Series(dtype=object)).astype(str)
            pull_flags = [_is_pull(s, b) for s, b in zip(side_vals, bear_vals)]
            pull_pct = round(sum(pull_flags) / len(pull_flags) * 100, 1) if pull_flags else 0.0
            gb_pct   = round(ht_col[is_inplay].str.contains("GroundBall", case=False, na=False).mean() * 100, 1)
        else:
            pull_pct = 0.0
            gb_pct   = 0.0

        # First pitch swing %
        fps_mask = (balls_c == 0) & (strikes_c == 0)
        fps_pct  = round(is_swing[fps_mask].mean() * 100, 1) if fps_mask.sum() else 0.0

        # Two-strike chase %
        ts_mask  = (strikes_c == 2)
        ts_oos   = ts_mask & ~is_inzone
        ts_chase = round(is_swing[ts_oos].mean() * 100, 1) if ts_oos.sum() else 0.0

        # Batter hand
        bside_vals = df_h.get("BatterSide", pd.Series(dtype=object)).astype(str)
        bside_counts = bside_vals.value_counts()
        raw_side = bside_counts.index[0] if not bside_counts.empty else "R"
        if str(raw_side).strip().upper().startswith("L"):
            bhand = "LHH"
        elif str(raw_side).strip().upper() in ("B","S","SW"):
            bhand = "SWB"
        else:
            bhand = "RHH"

        # Season batting line
        is_hit  = is_inplay & play.isin(["Single","Double","Triple","HomeRun"])
        bbout   = is_inplay & play.eq("Out")
        fc_mask = play.eq("FieldersChoice")
        err_m   = play.eq("Error")
        hbp_m   = s_call.eq("HitByPitch")
        n_hits  = int(is_hit.sum())
        n_ab    = int(n_hits + n_so + bbout.sum() + fc_mask.sum() + err_m.sum())
        n_hbp   = int(hbp_m.sum())
        bases   = (play.eq("Single").sum() + 2*play.eq("Double").sum()
                   + 3*play.eq("Triple").sum() + 4*play.eq("HomeRun").sum())
        _avg = n_hits/n_ab if n_ab else 0.0
        _obp = (n_hits + n_bb + n_hbp) / n_pa if n_pa else 0.0
        _slg = bases/n_ab if n_ab else 0.0
        avg_str = f".{int(round(_avg*1000)):03d}"
        obp_str = f".{int(round(_obp*1000)):03d}"
        ops_str = f".{int(round((_obp+_slg)*1000)):03d}"

        # Platoon splits (basic)
        def _plt_stats(hand_filter):
            sub = df_h[pthrows.str.strip().str.upper().str.startswith(hand_filter[0].upper())]
            if sub.empty:
                return {"avg": "—", "ops": "—", "k": "—", "bb": "—"}
            _sc = sub.get("PitchCall", pd.Series(dtype=object)).astype(str)
            _pl = sub.get("PlayResult", pd.Series(dtype=object)).astype(str)
            _kb = sub.get("KorBB",     pd.Series(dtype=object)).astype(str)
            _pp = pd.to_numeric(sub.get("PitchofPA", pd.Series(dtype=float)), errors="coerce")
            _pa = int((_pp == 1).sum())
            if not _pa:
                return {"avg": "—", "ops": "—", "k": "—", "bb": "—"}
            _so  = int(_kb.eq("Strikeout").sum())
            _bb2 = int(_kb.eq("Walk").sum())
            _ip  = _sc.eq("InPlay")
            _hit = _ip & _pl.isin(["Single","Double","Triple","HomeRun"])
            _bbo = _ip & _pl.eq("Out")
            _fc  = _pl.eq("FieldersChoice")
            _err = _pl.eq("Error")
            _hbp2= _sc.eq("HitByPitch")
            _h   = int(_hit.sum())
            _ab2 = int(_h + _so + _bbo.sum() + _fc.sum() + _err.sum())
            _bas = (_pl.eq("Single").sum() + 2*_pl.eq("Double").sum()
                    + 3*_pl.eq("Triple").sum() + 4*_pl.eq("HomeRun").sum())
            _a   = _h/_ab2 if _ab2 else 0.0
            _o   = (_h+_bb2+_hbp2.sum())/_pa if _pa else 0.0
            _s   = _bas/_ab2 if _ab2 else 0.0
            return {
                "avg": f".{int(round(_a*1000)):03d}",
                "ops": f".{int(round((_o+_s)*1000)):03d}",
                "k":   f"{round(_so/_pa*100,1)}%",
                "bb":  f"{round(_bb2/_pa*100,1)}%",
            }

        plt_R = _plt_stats("R")
        plt_L = _plt_stats("L")

        # Zone vulnerability (3x3) — whiff or weak contact rate per zone
        x_bounds = [(-0.83, -0.277), (-0.277, 0.277), (0.277, 0.83)]
        y_bounds = [(2.83, 3.5), (1.83, 2.83), (1.5, 1.83)]
        zone_scores = []
        for yr, yb in y_bounds:
            for xr in x_bounds:
                cell_mask = (lside.between(xr[0], xr[1]) & lht.between(yr, yb))
                if cell_mask.sum() < 3:
                    zone_scores.append(0)
                    continue
                cell_whiff = is_whiff[cell_mask].mean()
                cell_hard  = (exitv[is_inplay & cell_mask] >= 95).mean() if (is_inplay & cell_mask).sum() else 0.0
                vuln = cell_whiff * 0.6 + (1 - min(cell_hard, 1.0)) * 0.4
                zone_scores.append(round(vuln * 3))
        # Normalise to 0-3
        mx = max(zone_scores) if any(zone_scores) else 1
        zone_scores = [min(3, round(z / max(mx, 1) * 3)) for z in zone_scores]

        return {
            "k": k_pct, "bb": bb_pct, "pull": pull_pct,
            "fps": fps_pct, "tkc": ts_chase, "gb": gb_pct,
            "hand": bhand, "avg": avg_str, "obp": obp_str, "ops": ops_str,
            "zone": zone_scores,
            "plt": {"R": plt_R, "L": plt_L},
        }

    # ── Pitch metric computation (hitter framing: green = good for hitter) ───
    def _sc_calc(h: dict, pitch: dict, p_hand: str) -> dict:
        k, bb, pull = h["k"], h["bb"], h["pull"]
        fps, tkc, gb = h["fps"], h["tkc"], h["gb"]
        hh = h["hand"]
        t, v, ivb, hb = pitch["t"], pitch["v"], pitch["ivb"], pitch["hb"]
        same = (p_hand == "RHP" and hh == "RHH") or (p_hand == "LHP" and hh == "LHH")
        glove = (p_hand == "RHP" and hb < -4) or (p_hand == "LHP" and hb > 4)

        w = k * 1.0 + 9
        if t in ("4-Seam", "Sinker"):
            if ivb > 14: w += (k - 18) * 0.15
            if v > 93:   w += 2.5
        if t == "2-Seam":
            if gb > 42: w -= 3
        if t in ("Slider", "Cutter"):
            if glove and pull > 46: w += 5
            w += tkc * 0.08
        if t == "Changeup":
            w += tkc * 0.14
            if not same: w += 3
        if t == "Curveball":
            w += (k - 18) * 0.12 + 3
        w = int(round(min(max(w, 12), 53)))

        hrd = max(9.0, 41 - w * 0.32 + (fps - 38) * 0.1)
        if t == "4-Seam" and (100 - gb) > 36: hrd += 3
        if t in ("2-Seam", "Sinker") and gb > 42: hrd += 4
        hrd = int(round(min(max(hrd, 9), 50)))

        ev = round(min(max(82 + (hrd - 20) * 0.33, 79), 96), 1)

        ch = tkc * 0.5 + ((-2) if bb > 11 else 3) + 11
        if t in ("Curveball", "Slider"): ch += 4
        ch = int(round(min(max(ch, 12), 51)))

        return {"w": w, "hrd": hrd, "ev": ev, "ch": ch}

    def _sc_verdict(val, metric):
        """Hitter framing: green = good for our hitter."""
        if metric == "w":   return "good" if val < 22 else ("ok" if val < 34 else "bad")
        if metric == "hrd": return "good" if val > 30 else ("ok" if val > 21 else "bad")
        if metric == "ev":  return "good" if val > 88 else ("ok" if val > 84 else "bad")
        if metric == "ch":  return "good" if val < 22 else ("ok" if val < 33 else "bad")
        return "ok"

    def _sc_chip_label(metric, verdict):
        g, b = verdict == "good", verdict == "bad"
        if metric == "w":
            return "Low whiff" if g else ("High whiff" if b else "Average")
        if metric == "hrd":
            return "Hard contact" if g else ("Weak contact" if b else "Average")
        if metric == "ev":
            return "High EV" if g else ("Low EV" if b else "Avg EV")
        if metric == "ch":
            return "Disciplined" if g else ("Chase risk" if b else "Moderate")
        return "—"

    def _sc_color(verdict):
        return {"good": SC_GRN, "ok": SC_AMB, "bad": SC_RED}.get(verdict, SC_AMB)

    def _sc_chip_cls(verdict):
        return {"good": "sc-chip-good", "ok": "sc-chip-ok", "bad": "sc-chip-bad"}.get(verdict, "sc-chip-ok")

    def _sc_val_color(verdict):
        return {"good": "cg", "ok": "ca", "bad": "cr"}.get(verdict, "ca")

    def _sc_grade(results: list) -> int:
        """Hitter framing: high score = favorable matchup."""
        n = len(results)
        if not n: return 50
        aw  = sum(r["w"]   for r in results) / n
        ah  = sum(r["hrd"] for r in results) / n
        ae  = sum(r["ev"]  for r in results) / n
        ac  = sum(r["ch"]  for r in results) / n
        s = ((53 - aw)/53*35) + (ah/50*25) + ((ae - 79)/17*20) + ((51 - ac)/51*20)
        return int(min(max(round(s), 5), 95))

    def _sc_movement_svg(ivb: float, hb: float, color: str) -> str:
        cx = round(8 + ((hb + 16) / 32) * 64, 1)
        cy = round(72 - ((ivb + 14) / 34) * 64, 1)
        return (
            f'<svg viewBox="0 0 80 80" width="70" height="70" style="display:block;flex-shrink:0;">'
            f'<rect width="80" height="80" fill="#F9FAFB" stroke="#E5E7EB" stroke-width="0.5" rx="5"/>'
            f'<line x1="40" y1="3" x2="40" y2="77" stroke="#E5E7EB" stroke-width="0.5"/>'
            f'<line x1="3" y1="40" x2="77" y2="40" stroke="#E5E7EB" stroke-width="0.5"/>'
            f'<circle cx="{cx}" cy="{cy}" r="5.5" fill="{color}" opacity="0.9"/>'
            f'<circle cx="{cx}" cy="{cy}" r="9.5" fill="{color}" opacity="0.14"/>'
            f'<text x="40" y="79" text-anchor="middle" font-size="6.5" fill="#9CA3AF" font-family="sans-serif">HB</text>'
            f'<text x="3" y="44" text-anchor="middle" font-size="6.5" fill="#9CA3AF" '
            f'font-family="sans-serif" transform="rotate(-90,3,42)">IVB</text>'
            f'</svg>'
        )

    def _sc_pitch_note(pitch: dict, hitter: dict, p_hand: str) -> str:
        t, v, ivb, hb = pitch["t"], pitch["v"], pitch["ivb"], pitch["hb"]
        pull, k, tkc, gb, hh = hitter["pull"], hitter["k"], hitter["tkc"], hitter["gb"], hitter["hand"]
        same  = (p_hand == "RHP" and hh == "RHH") or (p_hand == "LHP" and hh == "LHH")
        glove = (p_hand == "RHP" and hb < -4) or (p_hand == "LHP" and hb > 4)

        if t in ("4-Seam", "Sinker"):
            if ivb > 15 and k < 18:
                return "Elite ride — stay in the zone and attack contact early in the count."
            if ivb < 13:
                return "Below-average ride — attack this fastball aggressively in hitter-friendly counts."
            return "Handle the ride — squared-up contact is achievable when it's located middle."
        if t == "2-Seam":
            return "Heavy arm-side run. Stay up the middle — don't over-commit to the pull side."
        if t in ("Slider", "Cutter"):
            if glove and pull > 46:
                return "Glove-side break is dangerous for pull-heavy hitters. Stay patient and take it deep if it's away."
            return "Breaking ball off the fastball tunnel. Don't commit early — let it travel."
        if t == "Changeup":
            return "Arm-side fade — keep your weight back. Don't expand down and away in two-strike counts."
        if t == "Curveball":
            return "12-6 break — look for it early and attack it in the zone. Don't chase below the knees."
        return "Secondary offering. Stay disciplined."

    def _sc_zone_html(zone_arr: list) -> str:
        mx = max(zone_arr) if any(zone_arr) else 1
        cells = []
        for v in zone_arr:
            if not v:
                cells.append('<div class="sc-zone-cell" style="background:#F9FAFB;"></div>')
                continue
            i = v / mx
            bg = (f"rgba(228,28,56,{0.75 if i > 0.7 else (0.35 if i > 0.4 else 0.12)})")
            tc = "#FFFFFF" if i > 0.7 else "#7F1D1D"
            cells.append(f'<div class="sc-zone-cell" style="background:{bg};color:{tc};">{v}</div>')
        return "".join(cells)

    # ── Radar chart (matplotlib, dark bg, blue) ───────────────────────────────
    def _sc_radar_pitcher(pitcher: dict):
        import matplotlib.pyplot as _plt
        import numpy as _np
        labels = ["Velocity", "Vert. break", "Horiz. break", "ERA quality", "Arsenal depth", "K/BB ratio"]
        av = sum(p["v"] for p in pitcher["ar"]) / len(pitcher["ar"])
        ai = sum(abs(p["ivb"]) for p in pitcher["ar"]) / len(pitcher["ar"])
        ah = sum(abs(p["hb"]) for p in pitcher["ar"]) / len(pitcher["ar"])
        try:
            er = float(pitcher["era"])
        except (TypeError, ValueError):
            er = 4.5  # ERA not in TrackMan data — use a neutral midpoint
        kb = float(pitcher["kbb"])
        values = [
            min((av - 73) / 25 * 100, 100),
            min(ai / 16 * 100, 100),
            min(ah / 13 * 100, 100),
            max(0, (5.5 - er) / 4 * 100),
            min(len(pitcher["ar"]) / 4 * 100, 100),
            min(kb / 5 * 100, 100),
        ]
        N = len(labels)
        angles = [n / float(N) * 2 * _np.pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]

        fig, ax = _plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True),
                                facecolor="white")
        ax.set_facecolor("white")
        ax.spines["polar"].set_color("#E5E7EB")
        ax.set_thetagrids([a * 180 / _np.pi for a in angles[:-1]], labels,
                          fontsize=8, color="#6B7280")
        ax.set_ylim(0, 100)
        ax.yaxis.set_visible(False)
        for gl in ax.yaxis.get_gridlines():
            gl.set_color("#E5E7EB")
        for gl in ax.xaxis.get_gridlines():
            gl.set_color("#E5E7EB")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=7.5, color="#4B5563")

        ax.plot(angles, values, color=SC_BLU, linewidth=1.5, linestyle="solid")
        ax.fill(angles, values, color=SC_BLU, alpha=0.10)
        ax.scatter(angles[:-1], values[:-1], color=SC_BLU, s=20, zorder=5)

        fig.tight_layout(pad=0.5)
        return fig

    # ── Selectors ─────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Scouting Selectors")

    hitter_opts  = batters_keys
    pitcher_opts = list(SCOUT_PITCHERS.keys())

    sel_hitter_key = st.sidebar.selectbox(
        "Nebraska hitter",
        options=hitter_opts,
        format_func=lambda k: display_name_by_key.get(k, k),
        key="scout_hitter",
    )
    sel_pitcher_key = st.sidebar.selectbox(
        "Opponent pitcher",
        options=pitcher_opts,
        format_func=lambda k: f"{SCOUT_PITCHERS[k]['name']} — {SCOUT_PITCHERS[k]['tm_label']} {SCOUT_PITCHERS[k]['hand']}",
        key="scout_pitcher",
    )

    # ── Compute matchup ───────────────────────────────────────────────────────
    pitcher = SCOUT_PITCHERS[sel_pitcher_key]
    df_h    = df_neb_bat[df_neb_bat["BatterKey"] == sel_hitter_key].copy()
    hitter_name = display_name_by_key.get(sel_hitter_key, sel_hitter_key)
    hitter_pos  = ""  # position not reliably in CSV; can wire up later

    if df_h.empty:
        st.warning("No data found for selected hitter. Try a different selection.")
    else:
        hitter = _sc_hitter_profile(df_h, pitcher["hand"])
        results = [_sc_calc(hitter, p, pitcher["hand"]) for p in pitcher["ar"]]
        score   = _sc_grade(results)

        # ── Matchup header card ───────────────────────────────────────────────
        grade_segs_html = ""
        col   = SC_GRN if score > 62 else (SC_AMB if score > 40 else SC_RED)
        label = "Favorable matchup" if score > 62 else ("Neutral" if score > 40 else "High threat")
        segs  = "".join(
            f'<div class="sc-grade-seg" style="background:{col if score >= (i+1)*10 else "#F3F4F6"}"></div>'
            for i in range(10)
        )
        lbl_left_col  = col if score < 40   else "#9CA3AF"
        lbl_right_col = col if score > 62   else "#9CA3AF"

        st.markdown(f"""
        <div class="sc-header-card">
          <div>
            <span class="sc-tag-nb">Nebraska</span>
            <span style="color:#9CA3AF;font-size:11px;margin-right:6px;">vs</span>
            <span class="sc-tag-opp">{pitcher['tm_label']}</span>
          </div>
          <div class="sc-matchup-names">
            {hitter_name}
            <span style="color:#9CA3AF;font-weight:400;font-size:1rem;"> vs </span>
            {pitcher['name']}
          </div>
          <div class="sc-matchup-sub">
            Nebraska {hitter['hand']} &nbsp;&middot;&nbsp;
            {hitter['avg']} AVG &nbsp;&middot;&nbsp; {hitter['obp']} OBP &nbsp;&middot;&nbsp; {hitter['ops']} OPS
            &nbsp;&nbsp;|&nbsp;&nbsp;
            {pitcher['hand']} &nbsp;&middot;&nbsp; {pitcher['era']} ERA &nbsp;&middot;&nbsp; {pitcher['ip']} IP &nbsp;&middot;&nbsp; K/BB {pitcher['kbb']}
          </div>
          <div style="margin-top:12px;">
            <div class="sc-grade-segs">{segs}</div>
            <div class="sc-grade-labels">
              <span class="sc-grade-lbl" style="color:{lbl_left_col};">High threat</span>
              <span class="sc-grade-verdict" style="color:{col};">{label}</span>
              <span class="sc-grade-lbl" style="color:{lbl_right_col};">Favorable</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-column layout ─────────────────────────────────────────────────
        lcol, rcol = st.columns([1.0, 0.85])

        # LEFT — pitch breakdown cards
        with lcol:
            st.markdown(
                f'<div class="sc-section-title"><span>{hitter_name} vs this arsenal</span></div>',
                unsafe_allow_html=True,
            )
            for i, pitch in enumerate(pitcher["ar"]):
                m    = results[i]
                vw   = _sc_verdict(m["w"],   "w")
                vh   = _sc_verdict(m["hrd"], "hrd")
                ve   = _sc_verdict(m["ev"],  "ev")
                vc   = _sc_verdict(m["ch"],  "ch")
                pcol = PITCH_COLORS.get(pitch["t"], SC_BLU)
                note = _sc_pitch_note(pitch, hitter, pitcher["hand"])
                fm   = lambda x: (f"+{x:.1f}" if x >= 0 else f"{x:.1f}")
                svg  = _sc_movement_svg(pitch["ivb"], pitch["hb"], pcol)

                st.markdown(f"""
                <div class="sc-pitch-card">
                  <div class="sc-pitch-head">
                    <div>
                      <span class="sc-pitch-name">{pitch['t']}</span>
                      <span class="sc-pitch-usage">{pitch['u']}</span>
                      <div class="sc-pitch-det">{pitch['v']} mph &nbsp;&middot;&nbsp;
                        IVB {fm(pitch['ivb'])}&quot; &nbsp;&middot;&nbsp; HB {fm(pitch['hb'])}&quot;</div>
                    </div>
                    {svg}
                  </div>
                  <div class="sc-mrow">
                    <div class="sc-mbox">
                      <div class="sc-mlbl">Whiff %</div>
                      <div class="sc-mval {_sc_val_color(vw)}">{m['w']}%</div>
                      <div class="sc-mbar" style="background:{_sc_color(vw)};width:{m['w']}%;max-width:100%;"></div>
                      <span class="sc-chip {_sc_chip_cls(vw)}">{_sc_chip_label('w', vw)}</span>
                    </div>
                    <div class="sc-mbox">
                      <div class="sc-mlbl">Hard hit %</div>
                      <div class="sc-mval {_sc_val_color(vh)}">{m['hrd']}%</div>
                      <div class="sc-mbar" style="background:{_sc_color(vh)};width:{m['hrd']}%;max-width:100%;"></div>
                      <span class="sc-chip {_sc_chip_cls(vh)}">{_sc_chip_label('hrd', vh)}</span>
                    </div>
                    <div class="sc-mbox">
                      <div class="sc-mlbl">Avg EV</div>
                      <div class="sc-mval {_sc_val_color(ve)}">{m['ev']}</div>
                      <div class="sc-mbar" style="background:{_sc_color(ve)};width:{min((m['ev']-79)/17*100,100):.0f}%;max-width:100%;"></div>
                      <span class="sc-chip {_sc_chip_cls(ve)}">{_sc_chip_label('ev', ve)}</span>
                    </div>
                    <div class="sc-mbox">
                      <div class="sc-mlbl">Chase %</div>
                      <div class="sc-mval {_sc_val_color(vc)}">{m['ch']}%</div>
                      <div class="sc-mbar" style="background:{_sc_color(vc)};width:{m['ch']}%;max-width:100%;"></div>
                      <span class="sc-chip {_sc_chip_cls(vc)}">{_sc_chip_label('ch', vc)}</span>
                    </div>
                  </div>
                  <div class="sc-pnote">{note}</div>
                </div>
                """, unsafe_allow_html=True)

        # RIGHT — radar, zone, platoon splits
        with rcol:
            # Pitcher threat radar
            st.markdown(
                '<div class="sc-section-title"><span>Pitcher threat profile</span></div>',
                unsafe_allow_html=True,
            )
            fig_r = _sc_radar_pitcher(pitcher)
            st.pyplot(fig_r, use_container_width=True)
            plt.close(fig_r)

            # Zone grid — pitcher location tendency (red)
            zone_cells_html = _sc_zone_html(pitcher["zone"])
            st.markdown(f"""
            <div class="sc-right-card">
              <div class="sc-section-title"><span>Pitcher location tendency</span></div>
              <div class="sc-zone-grid">{zone_cells_html}</div>
              <div style="display:flex;gap:10px;margin-top:8px;">
                <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#6B7280;">
                  <div style="width:8px;height:8px;border-radius:2px;background:rgba(228,28,56,0.75);"></div>
                  Most frequent
                </div>
                <div style="display:flex;align-items:center;gap:5px;font-size:11px;color:#6B7280;">
                  <div style="width:8px;height:8px;border-radius:2px;background:rgba(228,28,56,0.12);"></div>
                  Least frequent
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Platoon splits
            plt_R = hitter["plt"]["R"]
            plt_L = hitter["plt"]["L"]
            st.markdown(f"""
            <div class="sc-right-card">
              <div class="sc-section-title"><span>Platoon splits</span></div>
              <div class="sc-plt-grid">
                <div class="sc-plt-card">
                  <div class="sc-plt-hdr">vs RHP</div>
                  <div class="sc-plt-row"><span class="sc-plt-lbl">AVG</span><span class="sc-plt-val">{plt_R['avg']}</span></div>
                  <div class="sc-plt-row"><span class="sc-plt-lbl">OPS</span><span class="sc-plt-val">{plt_R['ops']}</span></div>
                  <div class="sc-plt-row"><span class="sc-plt-lbl">K%</span><span class="sc-plt-val">{plt_R['k']}</span></div>
                  <div class="sc-plt-row"><span class="sc-plt-lbl">BB%</span><span class="sc-plt-val">{plt_R['bb']}</span></div>
                </div>
                <div class="sc-plt-card">
                  <div class="sc-plt-hdr">vs LHP</div>
                  <div class="sc-plt-row"><span class="sc-plt-lbl">AVG</span><span class="sc-plt-val">{plt_L['avg']}</span></div>
                  <div class="sc-plt-row"><span class="sc-plt-lbl">OPS</span><span class="sc-plt-val">{plt_L['ops']}</span></div>
                  <div class="sc-plt-row"><span class="sc-plt-lbl">K%</span><span class="sc-plt-val">{plt_L['k']}</span></div>
                  <div class="sc-plt-row"><span class="sc-plt-lbl">BB%</span><span class="sc-plt-val">{plt_L['bb']}</span></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── AI Scout note ─────────────────────────────────────────────────────

        @st.cache_data(show_spinner=False)
        def _sc_ai_note(hitter_name: str, hitter_dict: str, pitcher_name: str,
                        pitcher_hand: str, pitcher_team: str, era: str,
                        arsenal_summary: str) -> str:
            prompt = (
                f"You are a college baseball analyst. Write a pregame hitting plan for "
                f"{hitter_name} (Nebraska hitter) facing {pitcher_name} "
                f"({pitcher_team} {pitcher_hand}, {era} ERA). "
                f"Hitter profile — K%: {hitter_dict}. "
                f"How the hitter performs vs this arsenal: {arsenal_summary}. "
                f"Write 2-3 sentences spoken directly to the hitter. "
                f"Name the pitch to attack and the pitch to respect. "
                f"Be specific and concrete like a hitting coach in a pregame meeting."
            )
            try:
                _api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if not _api_key:
                    try:
                        _api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
                    except Exception:
                        _api_key = ""
                if not _api_key:
                    return "Scout note unavailable — add ANTHROPIC_API_KEY to your Streamlit secrets to enable."
                resp = _requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": _api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 250,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=20,
                )
                data = resp.json()
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        return block["text"].strip()
            except Exception:
                pass
            return "Scout note unavailable."

        h_summary = (
            f"K%={hitter['k']}, BB%={hitter['bb']}, pull={hitter['pull']}%, "
            f"FPS={hitter['fps']}%, 2K-chase={hitter['tkc']}%, GB%={hitter['gb']}%"
        )
        fm2 = lambda x: (f"+{x:.1f}" if x >= 0 else f"{x:.1f}")
        ar_summary = "; ".join(
            f"{p['t']}({p['u']}, IVB {fm2(p['ivb'])}\", HB {fm2(p['hb'])}\", {p['v']} mph): "
            f"whiff {m['w']}%, hardHit {m['hrd']}%, EV {m['ev']}, chase {m['ch']}%"
            for p, m in zip(pitcher["ar"], results)
        )

        note_text = _sc_ai_note(
            hitter_name, h_summary, pitcher["name"],
            pitcher["hand"], pitcher["tm_label"], pitcher["era"], ar_summary,
        )
        st.markdown(
            f'<div class="sc-note-box"><div class="sc-section-title"><span>Scout note</span></div>'
            f'<div class="sc-note-body">{note_text}</div></div>',
            unsafe_allow_html=True,
        )
