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
st.set_page_config(page_title="Nebraska Hitter Reports", layout="centered")  # wide OFF

# Default data paths per period (you can change these in the expander)
DATA_PATH_2025   = "B10C25_hitter_app_columns.csv"
DATA_PATH_SCRIM  = "Scrimmage(27).csv"  # file, directory, or glob pattern
DATA_PATH_2026   = "B10C26_hitter_app_columns.csv"  # placeholder; update when ready

# Optional EV×LA probability lookup (used for xwOBA).
# Build this once with the helper script; if absent, xwOBA will display as —.
PROB_LOOKUP_PATH = "EV_LA_probabilities.csv"

BANNER_CANDIDATES = [
    "NebraskaChampions.jpg",
    "/mnt/data/NebraskaChampions.jpg",
]

HUSKER_RED = "#E60026"

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

# ──────────────────────────────────────────────────────────────────────────────
# wOBA CONSTANTS (FanGraphs Guts! 2025)
# ──────────────────────────────────────────────────────────────────────────────
WOBAC_2025 = {
    "wBB": 0.693, "wHBP": 0.723,
    "w1B": 0.883, "w2B": 1.253, "w3B": 1.585, "wHR": 2.037,
    # Handy for wRAA/wRC+ if you want later:
    "wOBAScale": 1.23,
    "lg_wOBA": 0.314,
}

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
    """
    Normalize names like 'Buettenback,  Max\\u200b' -> 'Buettenback, Max'
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)      # zero-widths
    s = re.sub(r"\s+", " ", s).strip()               # collapse spaces
    s = re.sub(r"\s*,\s*", ", ", s)                  # ', ' consistently
    parts = [p.strip() for p in s.split(",") if p.strip()]
    parts = [p.title() for p in parts]
    return ", ".join(parts)

def parse_date_robust(df: pd.DataFrame) -> pd.Series:
    """
    Build a reliable Date series:
      1) try common date-like columns with multiple parse attempts
      2) fallback: try to extract YYYYMMDD from GameID (e.g., '20250908_NEB_xxx')
    Returns a datetime64[ns] normalized to date (no time).
    """
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

    # Fallback from GameID like '20250908_xxx'
    if "GameID" in df.columns:
        gid = df["GameID"].astype(str)
        ymd = gid.str.extract(r"(20\d{6})", expand=False)  # 20YYYYMMDD
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

# Keep panel size fixed
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
    custom_wall_distances: list = None  # NEW: list of (angle, distance) tuples
):
    """Draw a baseball field with dirt infield and grass outfield"""
    home = np.array(origin)

    # Use custom wall distances if provided
    if custom_wall_distances is not None:
        # Create a polygon for the custom outfield shape
        angles = [item[0] for item in custom_wall_distances]
        distances = [item[1] for item in custom_wall_distances]
        
        # Generate points along the custom outfield boundary
        outfield_points = []
        for angle, dist in zip(angles, distances):
            rad = math.radians(angle)
            x = home[0] + dist * math.cos(rad)
            y = home[1] + dist * math.sin(rad)
            outfield_points.append([x, y])
        
        # Close the polygon by adding the origin
        outfield_points.append(home.tolist())
        
        # Draw custom outfield grass
        ax.add_patch(Polygon(outfield_points, closed=True, facecolor='#228B22', edgecolor='black', linewidth=2))
        
        # Set outfield_radius to max distance for other calculations
        outfield_radius = max(distances)
        
        # Get the corner distances for basepaths (at 45° and 135°)
        left_corner_dist = distances[0]  # First point at 45°
        right_corner_dist = distances[-1]  # Last point at 135°
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
    
    # Draw dirt infield
    ax.add_patch(Wedge(home, size, 45, 135, facecolor='#ED8B00', edgecolor='black', linewidth=2))
    
    # Basepaths - extend to the actual corner distances
    for angle, corner_dist in [(45, left_corner_dist), (135, right_corner_dist)]:
        rad = math.radians(angle)
        end = home + np.array([corner_dist * math.cos(rad), corner_dist * math.sin(rad)])
        perp = np.array([-math.sin(rad), math.cos(rad)])
        off = perp * (path_width / 2)
        corners = [home + off, home - off, end - off, end + off]
        ax.add_patch(Polygon(corners, closed=True, facecolor='#ED8B00', edgecolor='black', linewidth=1))

    # Infield grass with rounded arc behind bases (like Baseball Savant)
    gsize = size * grass_scale
    gfirst = home + np.array((gsize, gsize))
    gsecond = home + np.array((0.0, 2 * gsize))
    gthird = home + np.array((-gsize, gsize))
    
    # Create a rounded arc for the back of the infield grass instead of sharp triangle
    arc_angles = np.linspace(45, 135, 50)
    arc_radius = gsize * 1.8  # Extend the arc out further
    arc_points = []
    for angle in arc_angles:
        rad = math.radians(angle)
        x = home[0] + arc_radius * math.cos(rad)
        y = home[1] + arc_radius * math.sin(rad)
        arc_points.append([x, y])
    
    # Create polygon with rounded arc
    grass_polygon = [gfirst.tolist()] + arc_points + [gthird.tolist(), home.tolist()]
    ax.add_patch(Polygon(grass_polygon, closed=True, facecolor='#228B22', edgecolor='none'))
    
    for pos in [gfirst, gsecond, gthird]:
        ax.add_patch(Rectangle((pos[0] - base_size/2, pos[1] - base_size/2), base_size, base_size,
                               facecolor='white', edgecolor='black', linewidth=1))

    # Home plate
    half = base_size / 2
    plate = Polygon([
        (home[0] - half, home[1]),
        (home[0] + half, home[1]),
        (home[0] + half * 0.6, home[1] - half * 0.8),
        (home[0], home[1] - base_size),
        (home[0] - half * 0.6, home[1] - half * 0.8)
    ], closed=True, facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(plate)

    # Foul lines
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
    """
    Convert bearing (degrees, 0=straight away center) and distance to x,y coordinates
    Bearing: negative = pull side for RHH (left field), positive = opposite field
    Distance is in feet
    """
    # Convert bearing to radians (add 90 to make 0° point up the middle)
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
        return out[1:] if val < 1 else out  # show .382
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

def render_nb_banner(image_candidates=BANNER_CANDIDATES, title="Nebraska Baseball", height_px=180):
    b64 = None
    for p in image_candidates:
        b64 = _img_to_b64(p)
        if b64:
            break
    if not b64:
        return
    st.markdown(
        f"""
        <div style="position: relative; width: 100%; height: {height_px}px; border-radius: 12px; overflow: hidden; margin-bottom: 10px;">
          <img src="data:image/jpeg;base64,{b64}" style="width:100%; height:100%; object-fit:cover; filter: brightness(0.6);" />
          <div style="position:absolute; inset:0; background: rgba(0,0,0,0.35);"></div>
          <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center;">
            <div style="font-size:40px; font-weight:800; color:white; text-shadow: 0 2px 12px rgba(0,0,0,.9);">
              {title}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# BATTED BALL / DISCIPLINE / STATS
# ──────────────────────────────────────────────────────────────────────────────
def assign_spray_category(row):
    ang  = row.get('Bearing', np.nan)
    side = str(row.get('BatterSide', "")).upper()[:1]  # 'L' or 'R'
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
# COLLEGE BARREL MASK (BIP denominator): EV ≥ 95 mph AND 10° ≤ LA ≤ 35°
# ──────────────────────────────────────────────────────────────────────────────
def college_barrel_mask(ev_series: pd.Series, la_series: pd.Series) -> pd.Series:
    """
    Vectorized college barrel mask:
    - Exit velocity ≥ 95 mph
    - Launch angle between 10° and 35° (inclusive)
    Returns a boolean Series aligned to the input index.
    """
    ev = pd.to_numeric(ev_series, errors="coerce")
    la = pd.to_numeric(la_series, errors="coerce")
    return (ev >= 95.0) & la.between(10.0, 35.0, inclusive="both")
# ──────────────────────────────────────────────────────────────────────────────
# xwOBA SUPPORT: EV×LA probability merge + wOBA/xwOBA calculators
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_prob_lookup(path: str):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Expect columns: EV_bin, LA_bin, 1B, 2B, 3B, HR, Out (probabilities 0-1)
            return df
    except Exception:
        pass
    return None

PROB_LOOKUP = load_prob_lookup(PROB_LOOKUP_PATH)

def _bin_ev_la(df: pd.DataFrame):
    """Add EV_bin/LA_bin (categorical intervals) used to join with probability lookup."""
    df = df.copy()
    df["ExitSpeed"] = pd.to_numeric(df.get("ExitSpeed"), errors="coerce")
    df["Angle"]     = pd.to_numeric(df.get("Angle"), errors="coerce")
    # Use the same binning as you used when you created EV_LA_probabilities.csv
    ev_bins = np.arange(40, 115+3, 3)   # 40–115 mph, 3 mph bins
    la_bins = np.arange(-30, 50+5, 5)   # -30–50°, 5° bins
    df["EV_bin"] = pd.cut(df["ExitSpeed"], bins=ev_bins, right=False)
    df["LA_bin"] = pd.cut(df["Angle"],     bins=la_bins, right=False)
    return df

def merge_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge p1B/p2B/p3B/pHR into df based on EV_bin & LA_bin using PROB_LOOKUP.
    If lookup is missing or EV/LA NaN, leaves probabilities as NaN.
    """
    if PROB_LOOKUP is None or df.empty:
        return df
    if "EV_bin" not in df.columns or "LA_bin" not in df.columns:
        df = _bin_ev_la(df)

    lk = PROB_LOOKUP.copy()
    # Ensure columns present
    for c in ["1B","2B","3B","HR","Out"]:
        if c not in lk.columns:
            lk[c] = 0.0

    # Merge on the string representation of intervals to avoid category dtype mismatch
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
    # Rename to probability column names used by _compute_xwoba
    merged = merged.rename(columns={"1B":"p1B","2B":"p2B","3B":"p3B","HR":"pHR"})
    return merged

def _derive_ev_counts(df: pd.DataFrame):
    """Return dict with counts needed for wOBA: 1B,2B,3B,HR,BB,IBB,HBP,SF,AB,PA."""
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

    # Walks/HBP/IBB/SF
    bb      = int((korbb == 'Walk').sum())
    ibb     = 0  # set if you track IBB separately
    hbp     = int((s_call == 'HitByPitch').sum())
    sf_mask = play.astype(str).str.contains(r'(?i)sacrifice\s*fly|^SF$', na=False)
    sf      = int(sf_mask.sum())

    # AB and PA (same AB logic as elsewhere for consistency)
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
    """
    Expected wOBA using EV×LA probabilities if present (p1B,p2B,p3B,pHR).
    Falls back to NaN if probability columns are missing.
    """
    need = ["p1B","p2B","p3B","pHR"]
    if not all(c in df.columns for c in need):
        return float('nan')

    p1 = pd.to_numeric(df["p1B"], errors="coerce").fillna(0.0)
    p2 = pd.to_numeric(df["p2B"], errors="coerce").fillna(0.0)
    p3 = pd.to_numeric(df["p3B"], errors="coerce").fillna(0.0)
    pH = pd.to_numeric(df["pHR"], errors="coerce").fillna(0.0)

    # Expected value on contact
    exp_contact_num = (
        WOBAC_2025["w1B"]*p1 + WOBAC_2025["w2B"]*p2
      + WOBAC_2025["w3B"]*p3 + WOBAC_2025["wHR"]*pH
    ).sum()

    # Keep BB/HBP as actual events (standard)
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

# ── Broad pitch groups for Profiles ───────────────────────────────────────────
def _pitch_type_col(df: pd.DataFrame) -> str:
    """Best-effort column for pitch labels."""
    for c in ["AutoPitchType", "TaggedPitchType", "PitchType", "Pitch_Name"]:
        if c in df.columns:
            return c
    # create a blank so downstream code never KeyErrors
    df["AutoPitchType"] = ""
    return "AutoPitchType"

def _norm_text(x) -> str:
    try:
        return str(x).strip().lower()
    except Exception:
        return ""

def pitch_group_of(label: str) -> str:
    """Map raw label → {'Fastball','Offspeed','Breaking','Unknown'}"""
    t = _norm_text(label)

    # quick contains first (handles "FourSeamFastBall", "Two Seam", etc.)
    if any(k in t for k in ["fast", "four", "2-seam", "two seam", "sinker", "cutter", "cut"]):
        return "Fastball"
    if any(k in t for k in ["change", "split", "fork", "vulcan", "palm"]):
        return "Offspeed"
    if any(k in t for k in ["slider", "sweeper", "curve", "slurve", "knuck"]):
        return "Breaking"

    # common short codes
    if t in {"ff","fa","fb","ft","si","fc"}: return "Fastball"
    if t in {"ch","sp"}:                      return "Offspeed"
    if t in {"sl","cu","kc"}:                 return "Breaking"

    return "Unknown"

# ──────────────────────────────────────────────────────────────────────────────
# SPLIT METRICS (Totals & by pitch) + RANKINGS BASE
# ──────────────────────────────────────────────────────────────────────────────
def _compute_split_core(df: pd.DataFrame) -> dict:
    """Return a dict containing all fields we need for splits & rankings."""
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

    # Counts
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

    # Bases for SLG
    bases = (play.eq('Single').sum()
             + 2*play.eq('Double').sum()
             + 3*play.eq('Triple').sum()
             + 4*play.eq('HomeRun').sum())

    # Batting rates
    ba  = hits/ab if ab else 0.0
    obp = (hits + bb + hbp_mask.sum())/pa if pa else 0.0
    slg = bases/ab if ab else 0.0
    ops = obp + slg

    # EV/LA and batted-ball qualities
    avg_ev = exitv[inplay].mean()
    max_ev = exitv[inplay].max()
    avg_la = angle[inplay].mean()
    hard   = (exitv[inplay] >= 95).mean()*100 if inplay.any() else 0.0
    
    # College Barrel% (BIP denominator): EV ≥ 95 and 10°–35°
    if inplay.any():
        barrel_mask = college_barrel_mask(exitv[inplay], angle[inplay])
        barrel = float(barrel_mask.mean()) * 100.0
    else:
        barrel = 0.0

    # Discipline
    swing = isswing.mean()*100 if len(isswing) else 0.0
    whiff = (iswhiff.sum() / max(isswing.sum(),1) * 100) if len(isswing) else 0.0
    chase = (isswing[~isinzone].mean()*100) if (~isinzone).sum() else 0.0
    z_swing = (isswing[isinzone].mean()*100) if isinzone.sum() else 0.0
    z_contact = ((iscontact & isinzone).sum() / max(isswing[isinzone].sum(),1) * 100) if isinzone.sum() else 0.0
    z_whiff = ((iswhiff & isinzone).sum() / max(isswing[isinzone].sum(),1) * 100) if isinzone.sum() else 0.0

    # NEW: wOBA & xwOBA
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
    """
    Returns three DataFrames:
      t1_counts  -> Total + by pitch: PA AB SO BB Hits 2B 3B HR AVG OBP SLG OPS wOBA xwOBA
      t2_rates   -> Total + by pitch: Avg EV Max EV Avg LA HardHit% Barrel% Swing% Whiff% Chase% ZSwing% ZContact% ZWhiff%
      t3_batted  -> Totals only: LD% GB% FB% Pull% Middle% Oppo%
    """
    if df_profiles.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Total rows
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

      # By broad pitch group: Fastball, Offspeed, Breaking
    col = _pitch_type_col(df_profiles)
    groups_series = df_profiles[col].map(pitch_group_of)

    for g in ["Fastball", "Offspeed", "Breaking"]:
        sub = df_profiles[groups_series == g]
        if sub.empty:
            # keep row for consistency even if no pitches in that group
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


    # DataFrames (NOTE: include the new columns)
    t1 = pd.DataFrame(
        t1_rows,
        columns=["Split","PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS","wOBA","xwOBA"]
    )
    t2 = pd.DataFrame(
        t2_rows,
        columns=["Split","Avg EV","Max EV","Avg LA","HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]
    )

    # Format t1 batting rates as .xxx (for tables)
    for c in ["AVG","OBP","SLG","OPS","wOBA","xwOBA"]:
        if c in t1.columns:
            t1[c] = t1[c].apply(lambda v: "—" if pd.isna(v) else (f"{float(v):.3f}"[1:] if float(v) < 1.0 else f"{float(v):.3f}"))

    # Format t2: EV/LA with 2 decimals, percents 1 decimal
    for c in ["Avg EV","Max EV","Avg LA"]:
        if c in t2.columns:
            t2[c] = t2[c].apply(lambda v: "—" if pd.isna(v) else f"{float(v):.2f}")
    for c in ["HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]:
        if c in t2.columns:
            t2[c] = t2[c].apply(lambda v: "—" if pd.isna(v) else f"{round(float(v),1)}%")

    # T3 batted ball totals only
    t3 = create_batted_ball_profile(df_profiles).copy()
    for c in t3.columns:
        t3[c] = t3[c].apply(lambda v: "—" if pd.isna(v) else f"{float(v):.1f}%")

    return t1, t2, t3

# ──────────────────────────────────────────────────────────────────────────────
# RANKINGS HELPERS
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
    """
    Husker red header + conditional fill:
      • For most columns: leader (max) = green, last (min) = red
      • Special-case SO, ZWhiff%, Chase%: higher = red, lower = green
    """
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
# SPRAY CHART FUNCTION (NEW)
# ──────────────────────────────────────────────────────────────────────────────
def create_spray_chart(df_game: pd.DataFrame, batter_display_name: str):
    """
    Create a spray chart for balls in play from the selected game.
    Color-coded by PA number. Uses actual field dimensions: 335-395-325.
    """
    # Filter to balls in play with valid Bearing and Distance
    inplay = df_game[df_game.get('PitchCall') == 'InPlay'].copy()
    
    if inplay.empty:
        st.warning("No balls in play found for this game.")
        return None
    
    # Ensure we have the necessary columns
    if 'Bearing' not in inplay.columns or 'Distance' not in inplay.columns:
        st.warning("Missing 'Bearing' or 'Distance' columns in data.")
        return None
    
    inplay['Bearing'] = pd.to_numeric(inplay['Bearing'], errors='coerce')
    inplay['Distance'] = pd.to_numeric(inplay['Distance'], errors='coerce')
    
    # Remove rows with missing Bearing or Distance
    inplay = inplay.dropna(subset=['Bearing', 'Distance'])
    
    if inplay.empty:
        st.warning("No balls in play with valid Bearing and Distance data.")
        return None
    
    # Create PA number column based on GameID, Inning, Top/Bottom, PAofInning
    inplay = inplay.sort_values(['GameID', 'Inning', 'Top/Bottom', 'PAofInning', 'PitchofPA'])
    inplay['PA_num'] = inplay.groupby(['GameID', 'Inning', 'Top/Bottom', 'PAofInning']).ngroup() + 1
    
    # Convert bearing/distance to x,y coordinates
    coords = [bearing_distance_to_xy(row['Bearing'], row['Distance']) 
              for _, row in inplay.iterrows()]
    inplay['x'] = [c[0] for c in coords]
    inplay['y'] = [c[1] for c in coords]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create the custom wall distances matching Nebraska's fence
    # Generate many points along the fence for a smooth curve
    angles = np.linspace(45, 135, 100)
    wall_data = []
    for angle in angles:
        if angle <= 90:  # Left field to center
            t = (angle - 45) / (90 - 45)
            dist = 335 + t * (395 - 335)
        else:  # Center to right field
            t = (angle - 90) / (135 - 90)
            dist = 395 + t * (325 - 395)
        wall_data.append((angle, dist))
    
    # Draw the dirt diamond field with custom outfield shape
    draw_dirt_diamond(ax, origin=(0.0, 0.0), size=140, custom_wall_distances=wall_data)
    
    # Draw outfield wall with actual dimensions: LF=335, CF=395, RF=325
    angles = np.linspace(45, 135, 100)
    wall_distances = []
    for angle in angles:
        if angle <= 90:  # Left field to center
            t = (angle - 45) / (90 - 45)
            dist = 335 + t * (395 - 335)
        else:  # Center to right field
            t = (angle - 90) / (135 - 90)
            dist = 395 + t * (325 - 395)
        wall_distances.append(dist)
    
    # Plot outfield wall
    wall_x = [dist * np.cos(np.radians(ang)) for ang, dist in zip(angles, wall_distances)]
    wall_y = [dist * np.sin(np.radians(ang)) for ang, dist in zip(angles, wall_distances)]
    ax.plot(wall_x, wall_y, 'k-', linewidth=3, zorder=10, label='Outfield Wall')
    
    # Add distance markers
    for angle, dist, label in [(45, 335, '335'), (90, 395, '395'), (135, 325, '325')]:
        rad = np.radians(angle)
        x = dist * np.cos(rad)
        y = dist * np.sin(rad)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', 
                facecolor='yellow', edgecolor='black', linewidth=2, alpha=0.9), zorder=11)
    
    # Color palette for PAs
    n_pas = inplay['PA_num'].nunique()
    colors_list = plt.cm.tab20(np.linspace(0, 1, min(n_pas, 20)))
    pa_colors = {pa: colors_list[i % 20] for i, pa in enumerate(sorted(inplay['PA_num'].unique()))}
    
    # Plot each ball in play
    for idx, row in inplay.iterrows():
        pa_num = row['PA_num']
        play_result = str(row.get('PlayResult', ''))
        exit_speed = row.get('ExitSpeed', np.nan)
        
        # Marker size based on exit velocity
        if pd.notna(exit_speed):
            marker_size = max(150, min(exit_speed * 5, 600))
        else:
            marker_size = 250
        
        # Different markers for different outcomes
        if play_result in ['Single', 'Double', 'Triple', 'HomeRun']:
            marker = 'o'
            edgecolor = 'darkgreen'
            linewidth = 3
        else:
            marker = 'X'
            edgecolor = 'darkred'
            linewidth = 3
        
        # Plot the marker
        ax.scatter(row['x'], row['y'], 
                  c=[pa_colors[pa_num]], 
                  s=marker_size, 
                  marker=marker,
                  edgecolors=edgecolor, 
                  linewidths=linewidth,
                  alpha=0.95,
                  zorder=20)
        
        # Label with PA number (larger font, with background)
        ax.text(row['x'], row['y'], str(pa_num), 
               ha='center', va='center', 
               fontsize=11, fontweight='bold',
               color='white',
               bbox=dict(boxstyle='circle,pad=0.1', facecolor='black', alpha=0.5),
               zorder=21)
    
    # Create legend for PA numbers with detailed info
    legend_elements = []
    for pa in sorted(pa_colors.keys()):
        # Get all rows for this PA
        pa_rows = inplay[inplay['PA_num'] == pa]
        if pa_rows.empty:
            continue
        
        # Use the last row (the actual batted ball)
        row = pa_rows.iloc[-1]
        
        # Gather info
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
        
        # Create label with all info
        label = f"PA {pa}: {outcome_str} | {ev_str} mph, {la_str}, {dist_str} | {ht}"
        
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=pa_colors[pa], markersize=10,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=label)
        )
    
    # Add outcome type legend
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
    
    # Set axis limits to ensure all data is visible
    max_dist = max(inplay['Distance'].max(), 400)
    ax.set_xlim(-max_dist * 0.85, max_dist * 0.85)
    ax.set_ylim(-30, max_dist * 1.1)
    ax.set_aspect('equal')
    
    # Title
    date_str = format_date_long(inplay['Date'].iloc[0]) if 'Date' in inplay.columns else ""
    ax.set_title(f"{batter_display_name} - Spray Chart\n{date_str}", 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD HITTER REPORT (single game) — with boxed legends bottom
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
        """Return {'label': str, 'klass': 'Walk'|'K'|'InPlay'|'Other', 'k_type': 'Swinging'|'Looking'|'Unknown'|None,
                   'res': PlayResult or None, 'ev': float or None, 'tag': str or None}"""
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

# ──────────────────────────────────────────────────────────────────────────────
# PROFILE SPRAY CHART (color by hit type)
# ──────────────────────────────────────────────────────────────────────────────
def create_profile_spray_chart(df_profiles: pd.DataFrame, batter_display_name: str):
    """
    Create a spray chart for all balls in play from the filtered data.
    Color-coded by hit type (GB, FB, LD, PU).
    """
    # Filter to balls in play with valid Bearing and Distance
    inplay = df_profiles[df_profiles.get('PitchCall') == 'InPlay'].copy()
    
    if inplay.empty:
        return None
    
    # Ensure we have the necessary columns
    if 'Bearing' not in inplay.columns or 'Distance' not in inplay.columns:
        return None
    
    inplay['Bearing'] = pd.to_numeric(inplay['Bearing'], errors='coerce')
    inplay['Distance'] = pd.to_numeric(inplay['Distance'], errors='coerce')
    
    # Remove rows with missing Bearing or Distance
    inplay = inplay.dropna(subset=['Bearing', 'Distance'])
    
    if inplay.empty:
        return None
    
    # Convert bearing/distance to x,y coordinates
    coords = [bearing_distance_to_xy(row['Bearing'], row['Distance']) 
              for _, row in inplay.iterrows()]
    inplay['x'] = [c[0] for c in coords]
    inplay['y'] = [c[1] for c in coords]
    
    # Categorize hit types
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create the custom wall distances matching Nebraska's fence
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
    
    # Draw the dirt diamond field with custom outfield shape
    draw_dirt_diamond(ax, origin=(0.0, 0.0), size=100, custom_wall_distances=wall_data)
    
    # Draw outfield wall
    wall_x = [dist * np.cos(np.radians(ang)) for ang, dist in wall_data]
    wall_y = [dist * np.sin(np.radians(ang)) for ang, dist in wall_data]
    ax.plot(wall_x, wall_y, 'k-', linewidth=3, zorder=10)
    
    # Add distance markers
    for angle, dist, label in [(45, 335, '335'), (90, 395, '395'), (135, 325, '325')]:
        rad = np.radians(angle)
        x = dist * np.cos(rad)
        y = dist * np.sin(rad)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', 
                facecolor='yellow', edgecolor='black', linewidth=2, alpha=0.9), zorder=11)
    
    # Color mapping for hit types - bold, vibrant colors like Baseball Savant
    hit_type_colors = {
        'GroundBall': '#DC143C',  # Crimson red
        'LineDrive': '#FFD700',   # Gold
        'FlyBall': '#1E90FF',     # Dodger blue
        'Popup': '#FF69B4',       # Hot pink
        'Other': '#A9A9A9'        # Dark gray
    }
    
    # Plot each ball in play
    for idx, row in inplay.iterrows():
        hit_cat = row['HitCategory']
        play_result = str(row.get('PlayResult', ''))
        exit_speed = row.get('ExitSpeed', np.nan)
        
        # Smaller, more uniform marker sizes
        marker_size = 120  # Fixed smaller size
        
        # Different edge colors for hits vs outs
        if play_result in ['Single', 'Double', 'Triple', 'HomeRun']:
            edgecolor = 'black'
            linewidth = 2
        else:
            edgecolor = 'black'
            linewidth = 1.5
        
        # Plot the marker
        ax.scatter(row['x'], row['y'], 
                  c=hit_type_colors.get(hit_cat, '#A9A9A9'), 
                  s=marker_size, 
                  marker='o',
                  edgecolors=edgecolor, 
                  linewidths=linewidth,
                  alpha=0.85,
                  zorder=20)
    
    # Create legend for hit types
    legend_elements = []
    
    # Count each hit type
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
    
    # Add hit vs out legend
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
    
    # Set axis limits
    max_dist = max(inplay['Distance'].max(), 400)
    ax.set_xlim(-max_dist * 0.85, max_dist * 0.85)
    ax.set_ylim(-30, max_dist * 1.1)
    ax.set_aspect('equal')
    
    # Title
    ax.set_title(f"{batter_display_name} - Spray Chart (All Batted Balls)", 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
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
# LOADERS (single CSV or multi-CSV via directory/glob)
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

# ──────────────────────────────────────────────────────────────────────────────
# UI: BANNER + PERIOD SELECTOR + PATHS
# ──────────────────────────────────────────────────────────────────────────────
render_nb_banner(title="Nebraska Baseball")

period = st.selectbox(
    "Time Period",
    options=["2025 season", "2025/26 Scrimmages", "2026 season"],
    index=0
)

with st.expander("Data paths (optional quick edit)"):
    st.caption("Paste a CSV path, a directory path, or a glob pattern (e.g., `/mnt/data/scrims/*.csv`).")
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

# ──────────────────────────────────────────────────────────────────────────────
# BUILD HITTER KEYSPACE (NEB hitters with real PAs; dedup names)
# ──────────────────────────────────────────────────────────────────────────────
for col in ["Batter", "BatterTeam", "PitchofPA", "PitcherThrows", "PitcherTeam",
            "PlayResult", "KorBB", "PitchCall", "AutoPitchType", "ExitSpeed", "Angle",
            "PlateLocSide", "PlateLocHeight", "TaggedHitType", "Bearing", "BatterSide",
            "Distance"]:
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

# ──────────────────────────────────────────────────────────────────────────────
# Top section selector
# ──────────────────────────────────────────────────────────────────────────────
view_mode = st.radio("View", ["Standard Hitter Report", "Profiles & Heatmaps", "Rankings"], horizontal=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODE: STANDARD HITTER REPORT
# ──────────────────────────────────────────────────────────────────────────────
if view_mode == "Standard Hitter Report":
    st.markdown("### Nebraska Hitter Reports")
    colB, colD = st.columns([1, 1])

    batter_key_std = colB.selectbox(
        "Player",
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
        "Game Date",
        options=date_opts,
        format_func=lambda d: date_labels.get(d, format_date_long(d)),
        index=len(date_opts)-1 if date_opts else 0
    ) if date_opts else None

    if batter_key_std and selected_date:
        df_date = df_b_all[df_b_all["DateOnly"] == selected_date].copy()
    else:
        df_date = df_b_all.iloc[0:0].copy()

    batter_display = display_name_by_key.get(batter_key_std, batter_key_std)

    if df_date.empty:
        st.info("Select a player and game date to see the Standard Hitter Report.")
    else:
        st.markdown("### Standard Hitter Report")
        fig_std = create_hitter_report(df_date, batter_display, ncols=3)
        if fig_std:
            st.pyplot(fig_std)
        
        # Add spray chart
        st.markdown("### Spray Chart")
        fig_spray = create_spray_chart(df_date, batter_display)
        if fig_spray:
            st.pyplot(fig_spray)
        else:
            st.info("No balls in play with valid location data for this game.")

# ──────────────────────────────────────────────────────────────────────────────
# MODE: PROFILES & HEATMAPS (3 tables + heatmaps)
# ──────────────────────────────────────────────────────────────────────────────
elif view_mode == "Profiles & Heatmaps":
    st.markdown("### Profiles & Heatmaps")

    batter_key = st.selectbox(
        "Player",
        options=batters_keys,
        index=0,
        format_func=lambda k: display_name_by_key.get(k, k)
    )

    st.markdown("#### Filters")
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

    sel_days = colD2.multiselect("Days", options=present_days, default=[], key="prof_days")

    lastN = int(colN.number_input("Last N games", min_value=0, max_value=50, step=1, value=0, format="%d", key="prof_lastn"))
    hand_choice = colH.radio("Pitcher Hand", ["Both","LHP","RHP"], index=0, horizontal=True, key="prof_hand")

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
        st.info("No rows for the selected filters.")
    elif batter_key:
        season_label = {
            "2025 season": "2025",
            "2025/26 Scrimmages": "2025/26 Scrimmages",
            "2026 season": "2026",
        }.get(period, "—")
        st.markdown(f"#### Split Profiles — {display_name_by_key.get(batter_key,batter_key)} ({season_label})")

        t1_counts, t2_rates, t3_batted = build_profile_tables(df_profiles)

        st.markdown("**Summary**")
        st.table(themed_styler(t1_counts, nowrap=True))

        st.markdown("**Plate Discipline**")
        st.table(themed_styler(t2_rates, nowrap=True))

        st.markdown("**Batted Ball Distribution**")
        st.table(themed_styler(t3_batted, nowrap=True))

        st.markdown("#### Spray Chart")
        fig_spray = create_profile_spray_chart(df_profiles, display_name_by_key.get(batter_key, batter_key))
        if fig_spray:
            st.pyplot(fig_spray)
        else:
            st.info("No balls in play with valid location data for the selected filters.")

        st.markdown("#### Hitter Heatmaps")
        fig_hm = hitter_heatmaps(df_profiles, batter_key)
        if fig_hm:
            st.pyplot(fig_hm)

# ──────────────────────────────────────────────────────────────────────────────
# MODE: RANKINGS (team-wide, click-to-sort, red headers, leader/last coloring)
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("### Rankings")

    st.markdown("#### Filters")
    colM, colD2, colN, colH = st.columns([1.2, 1.2, 0.9, 1.9])

    df_scope = df_neb_bat.copy()

    dates_all = pd.to_datetime(df_scope["Date"], errors="coerce").dropna().dt.date
    present_months = sorted(pd.Series(dates_all).map(lambda d: d.month).unique().tolist())
    sel_months = colM.multiselect(
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
    sel_days = colD2.multiselect("Days", options=present_days, default=[], key="rk_days")

    lastN = int(colN.number_input("Last N games", min_value=0, max_value=50, step=1, value=0, format="%d", key="rk_lastn"))
    hand_choice = colH.radio("Pitcher Hand", ["Both","LHP","RHP"], index=0, horizontal=True, key="rk_hand")

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
        st.info("No rows for the selected filters.")
        st.stop()

    rankings_df = build_rankings_numeric(df_scope, display_name_by_key)
    min_pa = int(st.number_input("Min PA", min_value=0, value=0, step=1, key="rk_min_pa"))
    if min_pa > 0:
        rankings_df = rankings_df[rankings_df["PA"] >= min_pa]

    styled = style_rankings(rankings_df)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=520
    )
    
    # Show complete fall stats table if in scrimmage period
    if period == "2025/26 Scrimmages":
        st.markdown("---")
        st.markdown("#### Complete Fall Scrimmage Statistics")
        st.markdown("*Full season stats for all players during fall scrimmages*")
        
        min_pa_complete = int(st.number_input("Min PA for Complete Table", min_value=0, value=10, step=1, key="complete_min_pa"))
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

# ──────────────────────────────────────────────────────────────────────────────
# MODE: FALL SUMMARY (individual player fall scrimmage performance)
# ──────────────────────────────────────────────────────────────────────────────
else:  # Fall Summary
    st.markdown("### Fall Scrimmages Summary")
    
    # Only show if we're in the scrimmages period
    if period != "2025/26 Scrimmages":
        st.info("Please select '2025/26 Scrimmages' from the Time Period dropdown to view Fall Summary.")
        st.stop()
    
    # Player selector
    fall_player = st.selectbox(
        "Select Player",
        options=batters_keys,
        index=0,
        format_func=lambda k: display_name_by_key.get(k, k),
        key="fall_summary_player"
    )
    
    if not fall_player:
        st.info("Select a player to view their fall summary.")
        st.stop()
    
    # Get player's fall data
    df_fall = df_neb_bat[df_neb_bat["BatterKey"] == fall_player].copy()
    
    if df_fall.empty:
        st.warning(f"No fall scrimmage data found for {display_name_by_key.get(fall_player, fall_player)}.")
        st.stop()
    
    player_name = display_name_by_key.get(fall_player, fall_player)
    fall_stats = _compute_split_core(df_fall)
    
    # Player name banner
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, {HUSKER_RED} 0%, #a00018 100%); 
                    padding: 40px 30px; border-radius: 12px; margin-bottom: 30px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 48px; 
                       font-weight: 700; letter-spacing: -1px;">
                {player_name}
            </h1>
            <p style="color: rgba(255,255,255,0.9); text-align: center; margin-top: 12px; 
                      font-size: 20px; font-weight: 400; letter-spacing: 0.5px;">
                Fall 2025 Performance Report
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Slash line
    woba_value = fall_stats['wOBA'] if pd.notna(fall_stats['wOBA']) else 0.0
    st.markdown(f"""
        <div style="background: white; padding: 30px; border-radius: 12px; margin-bottom: 30px;
                    border: 2px solid {HUSKER_RED}; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div style="flex: 1;">
                    <p style="color: #6c757d; font-size: 13px; margin: 0; font-weight: 600; 
                              text-transform: uppercase; letter-spacing: 1px;">Batting Avg</p>
                    <p style="font-size: 48px; font-weight: 700; margin: 8px 0; color: {HUSKER_RED};">
                        {fall_stats['AVG']:.3f}
                    </p>
                </div>
                <div style="flex: 1;">
                    <p style="color: #6c757d; font-size: 13px; margin: 0; font-weight: 600; 
                              text-transform: uppercase; letter-spacing: 1px;">On-Base %</p>
                    <p style="font-size: 48px; font-weight: 700; margin: 8px 0; color: {HUSKER_RED};">
                        {fall_stats['OBP']:.3f}
                    </p>
                </div>
                <div style="flex: 1;">
                    <p style="color: #6c757d; font-size: 13px; margin: 0; font-weight: 600; 
                              text-transform: uppercase; letter-spacing: 1px;">Slugging %</p>
                    <p style="font-size: 48px; font-weight: 700; margin: 8px 0; color: {HUSKER_RED};">
                        {fall_stats['SLG']:.3f}
                    </p>
                </div>
                <div style="flex: 1;">
                    <p style="color: #6c757d; font-size: 13px; margin: 0; font-weight: 600; 
                              text-transform: uppercase; letter-spacing: 1px;">wOBA</p>
                    <p style="font-size: 48px; font-weight: 700; margin: 8px 0; color: {HUSKER_RED};">
                        {woba_value:.3f}
                    </p>
                </div>
                <div style="flex: 1;">
                    <p style="color: #6c757d; font-size: 13px; margin: 0; font-weight: 600; 
                              text-transform: uppercase; letter-spacing: 1px;">OPS</p>
                    <p style="font-size: 48px; font-weight: 700; margin: 8px 0; color: #333;">
                        {fall_stats['OPS']:.3f}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Counting stats
    st.markdown("#### Counting Stats")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    for col, label, value in [(c1, "Plate App", fall_stats['PA']), 
                               (c2, "Hits", fall_stats['Hits']),
                               (c3, "Doubles", fall_stats['2B']),
                               (c4, "Triples", fall_stats['3B']),
                               (c5, "Home Runs", fall_stats['HR'])]:
        with col:
            st.markdown(f"""
                <div style="background: white; padding: 24px; border-radius: 10px; text-align: center; 
                            border: 2px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <p style="margin: 0; font-size: 13px; color: #6c757d; font-weight: 600; 
                              text-transform: uppercase;">{label}</p>
                    <p style="margin: 8px 0 0 0; font-size: 40px; font-weight: 700; color: {HUSKER_RED};">
                        {value}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Batted Ball Quality
    st.markdown("#### Batted Ball Quality")
    c6, c7 = st.columns(2)
    
    avg_ev = fall_stats['Avg EV'] if pd.notna(fall_stats['Avg EV']) else 0
    max_ev = fall_stats['Max EV'] if pd.notna(fall_stats['Max EV']) else 0
    hard_hit = fall_stats['HardHit%'] if pd.notna(fall_stats['HardHit%']) else 0
    barrel = fall_stats['Barrel%'] if pd.notna(fall_stats['Barrel%']) else 0
    
    with c6:
        st.markdown(f"""
            <div style="background: white; padding: 24px; border-radius: 10px; 
                        border: 2px solid #e9ecef; margin-bottom: 15px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="font-weight: 600; color: #495057; margin-bottom: 12px; font-size: 15px;">
                    Avg Exit Velocity
                </p>
                <p style="font-size: 36px; font-weight: 700; color: {HUSKER_RED}; margin: 0;">
                    {avg_ev:.1f} <span style="font-size: 20px; color: #6c757d;">mph</span>
                </p>
                <div style="background: #f1f3f5; border-radius: 8px; height: 8px; margin-top: 12px; overflow: hidden;">
                    <div style="background: {HUSKER_RED}; height: 100%; width: {min(avg_ev/110*100, 100)}%; 
                                border-radius: 8px;"></div>
                </div>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 10px; 
                        border: 2px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="font-weight: 600; color: #495057; margin-bottom: 12px; font-size: 15px;">
                    Max Exit Velocity
                </p>
                <div style="background: #f1f3f5; border-radius: 8px; height: 8px; margin-top: 12px; overflow: hidden;">
                    <div style="background: {HUSKER_RED}; height: 100%; width: {min(max_ev/115*100, 100)}%; 
                                border-radius: 8px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with c7:
        st.markdown(f"""
            <div style="background: white; padding: 24px; border-radius: 10px; 
                        border: 2px solid #e9ecef; margin-bottom: 15px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="font-weight: 600; color: #495057; margin-bottom: 12px; font-size: 15px;">
                    Hard Hit %
                </p>
                <p style="font-size: 36px; font-weight: 700; color: {HUSKER_RED}; margin: 0;">
                    {hard_hit:.1f}<span style="font-size: 20px;">%</span>
                </p>
                <div style="background: #f1f3f5; border-radius: 8px; height: 8px; margin-top: 12px; overflow: hidden;">
                    <div style="background: {HUSKER_RED}; height: 100%; width: {min(hard_hit, 100)}%; 
                                border-radius: 8px;"></div>
                </div>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 10px; 
                        border: 2px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="font-weight: 600; color: #495057; margin-bottom: 12px; font-size: 15px;">
                    Barrel %
                </p>
                <p style="font-size: 36px; font-weight: 700; color: {HUSKER_RED}; margin: 0;">
                    {barrel:.1f}<span style="font-size: 20px;">%</span>
                </p>
                <div style="background: #f1f3f5; border-radius: 8px; height: 8px; margin-top: 12px; overflow: hidden;">
                    <div style="background: {HUSKER_RED}; height: 100%; width: {min(barrel, 100)}%; 
                                border-radius: 8px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Plate Discipline & Batted Ball Type
    c8, c9 = st.columns(2)
    
    k_pct = (fall_stats['SO']/fall_stats['PA']*100) if fall_stats['PA'] > 0 else 0
    bb_pct = (fall_stats['BB']/fall_stats['PA']*100) if fall_stats['PA'] > 0 else 0
    whiff = fall_stats['Whiff%'] if pd.notna(fall_stats['Whiff%']) else 0
    chase = fall_stats['Chase%'] if pd.notna(fall_stats['Chase%']) else 0
    
    with c8:
        st.markdown("#### Plate Discipline")
        st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; 
                        border: 2px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">K%</p>
                        <p style="color: {HUSKER_RED}; font-size: 28px; font-weight: 700; margin: 4px 0;">
                            {k_pct:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">BB%</p>
                        <p style="color: {HUSKER_RED}; font-size: 28px; font-weight: 700; margin: 4px 0;">
                            {bb_pct:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Whiff%</p>
                        <p style="color: {HUSKER_RED}; font-size: 28px; font-weight: 700; margin: 4px 0;">
                            {whiff:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Chase%</p>
                        <p style="color: {HUSKER_RED}; font-size: 28px; font-weight: 700; margin: 4px 0;">
                            {chase:.1f}%
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with c9:
        st.markdown("#### Batted Ball Type")
        bb_profile = create_batted_ball_profile(df_fall)
        ld_pct = bb_profile['LD%'].iloc[0] if not bb_profile.empty else 0
        gb_pct = bb_profile['GB%'].iloc[0] if not bb_profile.empty else 0
        fb_pct = bb_profile['FB%'].iloc[0] if not bb_profile.empty else 0
        
        st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; 
                        border: 2px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; text-align: center;">
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Line Drive</p>
                        <p style="color: {HUSKER_RED}; font-size: 32px; font-weight: 700; margin: 4px 0;">
                            {ld_pct:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Ground Ball</p>
                        <p style="color: {HUSKER_RED}; font-size: 32px; font-weight: 700; margin: 4px 0;">
                            {gb_pct:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Fly Ball</p>
                        <p style="color: {HUSKER_RED}; font-size: 32px; font-weight: 700; margin: 4px 0;">
                            {fb_pct:.1f}%
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Spray Chart
    st.markdown("---")
    st.markdown("#### Fall Spray Chart")
    spray_fig = create_profile_spray_chart(df_fall, player_name)
    if spray_fig:
        st.pyplot(spray_fig)
    else:
        st.info("No balls in play with valid location data.")
    
    # Heatmaps
    st.markdown("---")
    st.markdown("#### Fall Heatmaps")
    heatmap_fig = hitter_heatmaps(df_fall, fall_player)
    if heatmap_fig:
        st.pyplot(heatmap_fig)
    else:
        st.info("Not enough data for heatmaps.")
                <div style="background: #f1f3f5; border-radius: 8px; height: 8px; margin-top: 12px; overflow: hidden;">
                    <div style="background: {HUSKER_RED}; height: 100%; width: {min(max_ev/115*100, 100)}%; 
                                border-radius: 8px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col7:
        hard_hit = player_stats['HardHit%'] if pd.notna(player_stats['HardHit%']) else 0
        barrel = player_stats['Barrel%'] if pd.notna(player_stats['Barrel%']) else 0
        
        st.markdown(f"""
            <div style="background: white; padding: 24px; border-radius: 10px; 
                        border: 2px solid #e9ecef; margin-bottom: 15px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="font-weight: 600; color: #495057; margin-bottom: 12px; font-size: 15px;">
                    Hard Hit %
                </p>
                <p style="font-size: 36px; font-weight: 700; color: {HUSKER_RED}; margin: 0;">
                    {hard_hit:.1f}<span style="font-size: 20px;">%</span>
                </p>
                <div style="background: #f1f3f5; border-radius: 8px; height: 8px; margin-top: 12px; overflow: hidden;">
                    <div style="background: {HUSKER_RED}; height: 100%; width: {min(hard_hit, 100)}%; 
                                border-radius: 8px;"></div>
                </div>
            </div>
            
            <div style="background: white; padding: 24px; border-radius: 10px; 
                        border: 2px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <p style="font-weight: 600; color: #495057; margin-bottom: 12px; font-size: 15px;">
                    Barrel %
                </p>
                <p style="font-size: 36px; font-weight: 700; color: {HUSKER_RED}; margin: 0;">
                    {barrel:.1f}<span style="font-size: 20px;">%</span>
                </p>
                <div style="background: #f1f3f5; border-radius: 8px; height: 8px; margin-top: 12px; overflow: hidden;">
                    <div style="background: {HUSKER_RED}; height: 100%; width: {min(barrel, 100)}%; 
                                border-radius: 8px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Plate Discipline & Batted Ball Distribution side by side
    col8, col9 = st.columns(2)
    
    with col8:
        st.markdown("#### Plate Discipline")
        
        k_pct = (player_stats['SO']/player_stats['PA']*100) if player_stats['PA'] > 0 else 0
        bb_pct = (player_stats['BB']/player_stats['PA']*100) if player_stats['PA'] > 0 else 0
        whiff = player_stats['Whiff%'] if pd.notna(player_stats['Whiff%']) else 0
        chase = player_stats['Chase%'] if pd.notna(player_stats['Chase%']) else 0
        
        st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; 
                        border: 2px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">K%</p>
                        <p style="color: {HUSKER_RED}; font-size: 28px; font-weight: 700; margin: 4px 0;">
                            {k_pct:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">BB%</p>
                        <p style="color: {HUSKER_RED}; font-size: 28px; font-weight: 700; margin: 4px 0;">
                            {bb_pct:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Whiff%</p>
                        <p style="color: {HUSKER_RED}; font-size: 28px; font-weight: 700; margin: 4px 0;">
                            {whiff:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Chase%</p>
                        <p style="color: {HUSKER_RED}; font-size: 28px; font-weight: 700; margin: 4px 0;">
                            {chase:.1f}%
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col9:
        st.markdown("#### Batted Ball Type")
        
        bb_profile = create_batted_ball_profile(df_player_fall)
        
        ld_pct = bb_profile['LD%'].iloc[0] if not bb_profile.empty else 0
        gb_pct = bb_profile['GB%'].iloc[0] if not bb_profile.empty else 0
        fb_pct = bb_profile['FB%'].iloc[0] if not bb_profile.empty else 0
        
        st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; 
                        border: 2px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; text-align: center;">
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Line Drive</p>
                        <p style="color: {HUSKER_RED}; font-size: 32px; font-weight: 700; margin: 4px 0;">
                            {ld_pct:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Ground Ball</p>
                        <p style="color: {HUSKER_RED}; font-size: 32px; font-weight: 700; margin: 4px 0;">
                            {gb_pct:.1f}%
                        </p>
                    </div>
                    <div>
                        <p style="color: #6c757d; font-size: 12px; margin: 0; text-transform: uppercase;">Fly Ball</p>
                        <p style="color: {HUSKER_RED}; font-size: 32px; font-weight: 700; margin: 4px 0;">
                            {fb_pct:.1f}%
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Spray chart and heatmaps
    st.markdown("---")
    st.markdown("#### Fall Spray Chart")
    fig_spray = create_profile_spray_chart(df_player_fall, player_display)
    if fig_spray:
        st.pyplot(fig_spray)
    else:
        st.info("No balls in play with valid location data for fall scrimmages.")
    
    st.markdown("---")
    st.markdown("#### Fall Heatmaps")
    fig_hm = hitter_heatmaps(df_player_fall, batter_key)
    if fig_hm:
        st.pyplot(fig_hm)
    else:
        st.info("Not enough data for heatmaps.")
    
    # Only show if we're in the scrimmages period
    if period != "2025/26 Scrimmages":
        st.info("Please select '2025/26 Scrimmages' from the Time Period dropdown to view Fall Summary.")
        st.stop()
    
    # Player selector
    batter_key = st.selectbox(
        "Player",
        options=batters_keys,
        index=0,
        format_func=lambda k: display_name_by_key.get(k, k),
        key="fall_player"
    )
    
    if not batter_key:
        st.info("Select a player to view their fall summary.")
        st.stop()
    
    # Get player's fall data
    df_player_fall = df_neb_bat[df_neb_bat["BatterKey"] == batter_key].copy()
    
    if df_player_fall.empty:
        st.warning(f"No fall scrimmage data found for {display_name_by_key.get(batter_key, batter_key)}.")
        st.stop()
    
    player_display = display_name_by_key.get(batter_key, batter_key)
    
    # Compute player stats
    player_stats = _compute_split_core(df_player_fall)
    
    # Big player name banner with Husker red background
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, {HUSKER_RED} 0%, #c40020 100%); 
                    padding: 30px; border-radius: 15px; margin-bottom: 30px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 48px; 
                       text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                {player_display}
            </h1>
            <p style="color: white; text-align: center; margin-top: 10px; font-size: 24px; 
                      opacity: 0.9;">
                Fall 2025 Performance Report
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Triple slash line in big bold numbers
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 25px; border-radius: 12px; margin-bottom: 25px;
                    border-left: 5px solid {HUSKER_RED};">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <p style="color: #6c757d; font-size: 14px; margin: 0; font-weight: 600;">BATTING AVG</p>
                    <p style="font-size: 42px; font-weight: bold; margin: 5px 0; color: {HUSKER_RED};">
                        {player_stats['AVG']:.3f}
                    </p>
                </div>
                <div>
                    <p style="color: #6c757d; font-size: 14px; margin: 0; font-weight: 600;">ON-BASE %</p>
                    <p style="font-size: 42px; font-weight: bold; margin: 5px 0; color: {HUSKER_RED};">
                        {player_stats['OBP']:.3f}
                    </p>
                </div>
                <div>
                    <p style="color: #6c757d; font-size: 14px; margin: 0; font-weight: 600;">SLUGGING %</p>
                    <p style="font-size: 42px; font-weight: bold; margin: 5px 0; color: {HUSKER_RED};">
                        {player_stats['SLG']:.3f}
                    </p>
                </div>
                <div>
                    <p style="color: #6c757d; font-size: 14px; margin: 0; font-weight: 600;">OPS</p>
                    <p style="font-size: 42px; font-weight: bold; margin: 5px 0; color: #198754;">
                        {player_stats['OPS']:.3f}
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
