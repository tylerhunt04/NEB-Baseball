# pitcher_app.py — PROFESSIONAL UI VERSION WITH ADVANCED ANALYTICS

import os, gc, re, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from matplotlib.patches import Rectangle, Ellipse, Polygon, Circle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2
from numpy.linalg import LinAlgError
from matplotlib import colors
from datetime import date
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nebraska Baseball — Pitcher Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Nebraska Baseball Pitcher Analysis Platform"
    }
)
st.set_option("client.showErrorDetails", True)

DATA_PATH_MAIN  = "pitcher_columns.csv"
DATA_PATH_SCRIM = "Scrimmage(28).csv"
LOGO_PATH   = "Nebraska-Cornhuskers-Logo.png"
BANNER_IMG  = "NebraskaChampions.jpg"
HUSKER_RED  = "#E60026"
HUSKER_CREAM = "#FEFDFA"
DARK_GRAY = "#2B2B2B"
LIGHT_GRAY = "#F5F5F5"
EXT_VIS_WIDTH = 480

# ─── Professional CSS Styling ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #FFFFFF;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #E60026 0%, #B8001F 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        margin: 24px 0 16px 0;
        font-weight: 600;
        font-size: 20px;
        box-shadow: 0 2px 8px rgba(230, 0, 38, 0.15);
    }
    
    /* Subsection headers */
    .subsection-header {
        color: #2B2B2B;
        padding: 12px 0;
        border-bottom: 3px solid #E60026;
        margin: 20px 0 12px 0;
        font-weight: 600;
        font-size: 18px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F8F8 100%);
        border-left: 4px solid #E60026;
        padding: 16px;
        border-radius: 8px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    
    .metric-label {
        color: #666666;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .metric-value {
        color: #2B2B2B;
        font-size: 28px;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .metric-sublabel {
        color: #999999;
        font-size: 11px;
        margin-top: 4px;
    }
    
    /* Filter section */
    .filter-section {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        margin-bottom: 20px;
    }
    
    /* Table styling improvements */
    .dataframe {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Divider */
    .professional-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #E60026, transparent);
        margin: 32px 0;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #F0F7FF;
        border-left: 4px solid #1E88E5;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        color: #1565C0;
    }
    
    .warning-box {
        background-color: #FFF8E1;
        border-left: 4px solid #FFA726;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        color: #EF6C00;
    }
    
    /* Button styling */
    .stDownloadButton button {
        background-color: #E60026;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stDownloadButton button:hover {
        background-color: #B8001F;
        box-shadow: 0 4px 8px rgba(230, 0, 38, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F8F9FA;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        border: 1px solid #E0E0E0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #E60026;
        color: white;
        border-color: #E60026;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Select box and input styling */
    .stSelectbox, .stMultiSelect, .stNumberInput {
        background-color: white;
    }
    
    /* Caption styling */
    .caption-text {
        color: #666666;
        font-size: 13px;
        font-style: italic;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Custom colormap for heatmaps
custom_cmap = colors.LinearSegmentedColormap.from_list(
    'custom_cmap',
    [
        (0.0, 'white'),
        (0.2, 'deepskyblue'),
        (0.3, 'white'),
        (0.7, 'red'),
        (1.0, 'red'),
    ],
    N=256
)

# ─── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def load_banner_b64() -> str | None:
    if not os.path.exists(BANNER_IMG): return None
    with open(BANNER_IMG, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@st.cache_resource
def load_logo_img():
    return mpimg.imread(LOGO_PATH) if os.path.exists(LOGO_PATH) else None

def show_and_close(fig, *, use_container_width=False):
    try:
        st.pyplot(fig=fig, clear_figure=False, use_container_width=use_container_width)
    finally:
        plt.close(fig); gc.collect()

def show_image_scaled(fig, *, width_px=EXT_VIS_WIDTH, dpi=200, pad_inches=0.1):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    buf.seek(0)
    st.image(buf, width=width_px)
    plt.close(fig); gc.collect()

# ─── Professional Hero Banner ─────────────────────────────────────────────────
def hero_banner(title: str, *, subtitle: str | None = None, height_px: int = 280):
    from streamlit.components.v1 import html as _html
    b64 = load_banner_b64() or ""
    sub_html = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    _html(
        f"""
        <style>
        .hero-wrap {{
            position: relative; width: 100%; height: {height_px}px; border-radius: 12px;
            overflow: hidden; margin-bottom: 2rem; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .hero-bg {{
            position: absolute; inset: 0;
            background: linear-gradient(135deg, rgba(0,0,0,0.5) 0%, rgba(230,0,38,0.3) 100%),
                        url('data:image/jpeg;base64,{b64}');
            background-size: cover; background-position: center;
        }}
        .hero-text {{
            position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
            flex-direction: column; color: #fff; text-align: center; z-index: 10;
        }}
        .hero-title {{ 
            font-size: 48px; font-weight: 800; letter-spacing: 1px; 
            text-shadow: 0 4px 12px rgba(0,0,0,0.6); margin: 0;
            text-transform: uppercase;
        }}
        .hero-sub {{ 
            font-size: 20px; font-weight: 500; opacity: .95; margin-top: 8px;
            text-shadow: 0 2px 8px rgba(0,0,0,0.5);
            letter-spacing: 0.5px;
        }}
        </style>
        <div class="hero-wrap">
          <div class="hero-bg"></div>
          <div class="hero-text">
            <h1 class="hero-title">{title}</h1>
            {sub_html}
          </div>
        </div>
        """,
        height=height_px + 40,
    )

hero_banner("Nebraska Baseball", subtitle="Pitcher Analytics Platform", height_px=280)

# ─── Professional Section Header Helper ──────────────────────────────────────
def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def subsection_header(text: str):
    st.markdown(f'<div class="subsection-header">{text}</div>', unsafe_allow_html=True)

def professional_divider():
    st.markdown('<div class="professional-divider"></div>', unsafe_allow_html=True)

def metric_card(label: str, value: str, sublabel: str = ""):
    sub_html = f'<div class="metric-sublabel">{sublabel}</div>' if sublabel else ""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)

def info_message(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def warning_message(text: str):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)

# ─── Outing Summary Helper ────────────────────────────────────────────────────
def make_outing_overall_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Overall outing summary for the currently filtered selection:
    IP, Pitches, Hits, Walks, SO, HBP, Strike%, Whiffs, Zone Whiffs, HardHits
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame([{
            "IP": "0.0", "Pitches": 0, "Hits": 0, "Walks": 0, "SO": 0, "HBP": 0,
            "Strike%": np.nan, "Whiffs": 0, "Whiff%": np.nan,
            "Zone Whiffs": 0, "Zone Whiff%": np.nan,
            "HardHits": 0, "HardHit% (BIP)": np.nan
        }])

    # Pitches thrown
    pitches = int(len(df_in))

    # PA-level terminal rows → counts + IP
    pa_tbl = _terminal_pa_table(df_in)
    box = _box_counts_from_PA(pa_tbl)
    ip_float, ip_disp = _compute_IP_from_outs(box["OUTS"])

    # Strike %
    strike_pct = strike_rate(df_in)  # already returns 0..100

    # Whiffs + Zone Whiffs
    call_col = _first_present(df_in, ["PitchCall","Pitch Call","PitchResult","Call"])
    x_col    = _first_present(df_in, ["PlateLocSide","Plate Loc Side","PlateSide","px","PlateLocX"])
    y_col    = _first_present(df_in, ["PlateLocHeight","Plate Loc Height","PlateHeight","pz","PlateLocZ"])

    whiff_cnt = 0
    whiff_pct = np.nan
    zwhiff_cnt = 0
    zwhiff_pct = np.nan

    if call_col:
        s_call = df_in[call_col].astype(str)
        is_swing = s_call.isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
        is_whiff = s_call.eq('StrikeSwinging')

        swings_total = int(is_swing.sum())
        whiff_cnt = int(is_whiff.sum())
        whiff_pct = (whiff_cnt / swings_total * 100.0) if swings_total > 0 else np.nan

        # Zone whiffs require plate location
        if x_col and y_col and x_col in df_in.columns and y_col in df_in.columns:
            xs = pd.to_numeric(df_in[x_col], errors="coerce")
            ys = pd.to_numeric(df_in[y_col], errors="coerce")
            l, b, w, h = get_zone_bounds()     # same zone as plots
            in_zone = xs.between(l, l+w) & ys.between(b, b+h)

            swings_in_zone = int((is_swing & in_zone).sum())
            zwhiff_cnt = int((is_whiff & in_zone).sum())
            zwhiff_pct = (zwhiff_cnt / swings_in_zone * 100.0) if swings_in_zone > 0 else np.nan

    # HardHits: EV ≥ 95 mph on balls in play
    ev_col = _first_present(df_in, ["ExitSpeed","Exit Velo","ExitVelocity","Exit_Velocity","ExitVel","EV","LaunchSpeed","Launch_Speed"])
    hard_cnt = 0
    hard_pct_bip = np.nan
    if ev_col and call_col:
        ev = pd.to_numeric(df_in[ev_col], errors="coerce")
        inplay = df_in[call_col].astype(str).eq("InPlay")
        bip = int(inplay.sum())
        hard_cnt = int(((ev >= 95.0) & inplay & ev.notna()).sum())
        hard_pct_bip = (hard_cnt / bip * 100.0) if bip > 0 else np.nan

    row = {
        "IP": ip_disp,
        "Pitches": pitches,
        "Hits": int(box["H"]),
        "Walks": int(box["BB"]),
        "SO": int(box["SO"]),
        "HBP": int(box["HBP"]),
        "Strike%": round(float(strike_pct), 1) if pd.notna(strike_pct) else np.nan,
        "Whiffs": whiff_cnt,
        "Whiff%": round(float(whiff_pct), 1) if pd.notna(whiff_pct) else np.nan,
        "Zone Whiffs": zwhiff_cnt,
        "Zone Whiff%": round(float(zwhiff_pct), 1) if pd.notna(zwhiff_pct) else np.nan,
        "HardHits": hard_cnt,
        "HardHit% (BIP)": round(float(hard_pct_bip), 1) if pd.notna(hard_pct_bip) else np.nan,
    }
    return pd.DataFrame([row])

# ─── Date helpers ─────────────────────────────────────────────────────────────
DATE_CANDIDATES = ["Date","date","GameDate","GAME_DATE","Game Date","date_game","Datetime",
                   "DateTime","game_datetime","GameDateTime"]

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower = {c.lower(): c for c in df.columns}
    found = None
    for cand in DATE_CANDIDATES:
        if cand.lower() in lower: found = lower[cand.lower()]; break
    if not found:
        df["Date"] = pd.NaT; return df
    dt = pd.to_datetime(df[found], errors="coerce")
    df["Date"] = pd.to_datetime(dt.dt.date, errors="coerce")
    return df

def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"

def format_date_long(d) -> str:
    if d is None or pd.isna(d): return ""
    d = pd.to_datetime(d).date()
    return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"

from datetime import date as _date
FB_ONLY_DATES = { _date(2025, 9, 3), _date(2025, 9, 4) }
def is_fb_only(d) -> bool:
    try: return pd.to_datetime(d).date() in FB_ONLY_DATES
    except Exception: return False
def label_date_with_fb(d) -> str:
    base = format_date_long(d)
    return f"{base} (FB Only)" if is_fb_only(d) else base

def summarize_dates_range(series_like) -> str:
    if series_like is None: return ""
    ser = pd.to_datetime(pd.Series(series_like), errors="coerce").dropna()
    if ser.empty: return ""
    uniq = ser.dt.date.unique()
    if len(uniq) == 1: return format_date_long(uniq[0])
    dmin, dmax = min(uniq), max(uniq)
    return f"{format_date_long(dmin)} – {format_date_long(dmax)}"

def filter_by_month_day(df, date_col="Date", months=None, days=None):
    if date_col not in df.columns or df.empty: return df
    s = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if months: mask &= s.dt.month.isin(months)
    if days:   mask &= s.dt.day.isin(days)
    return df[mask]

MONTH_CHOICES = [(1,"January"), (2,"February"), (3,"March"), (4,"April"), (5,"May"), (6,"June"),
                 (7,"July"), (8,"August"), (9,"September"), (10,"October"), (11,"November"), (12,"December")]
MONTH_NAME_BY_NUM = {n: name for n, name in MONTH_CHOICES}

def build_pitcher_season_label(months_sel, days_sel, selected_df: pd.DataFrame) -> str:
    if (not months_sel) and (not days_sel): return "Season"
    if months_sel and not days_sel and len(months_sel) == 1:
        return MONTH_NAME_BY_NUM.get(months_sel[0], "Season")
    if selected_df is None or selected_df.empty or "Date" not in selected_df.columns: return "Season"
    rng = summarize_dates_range(selected_df["Date"]); return rng if rng else "Season"

def apply_month_day_lastN(df_in: pd.DataFrame, months: list[int], days: list[int], last_n_games: int):
    df = df_in.copy()
    if df.empty or "Date" not in df.columns: return df, "Season"
    if months or days:
        df = filter_by_month_day(df, months=months or None, days=days or None).copy()
    if last_n_games and last_n_games > 0 and not df.empty:
        ud = pd.to_datetime(df["Date"], errors="coerce").dt.date.dropna().unique()
        ud = sorted(ud)
        keep = set(ud[-last_n_games:])
        df = df[pd.to_datetime(df["Date"], errors="coerce").dt.date.isin(keep)].copy()
    base = build_pitcher_season_label(months, days, df)
    if last_n_games and last_n_games > 0:
        rng = summarize_dates_range(df["Date"]) if not (months or days) else base
        return df, (f"Last {last_n_games} games — {rng}" if rng and rng != "Season" else f"Last {last_n_games} games")
    return df, base or "Season"

# ─── Segments & column pickers ────────────────────────────────────────────────
SESSION_TYPE_CANDIDATES = ["SessionType","Session Type","GameType","Game Type","EventType","Event Type",
                           "Context","context","Type","type","Environment","Env"]

def pick_col(df: pd.DataFrame, *cands) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c and c.lower() in lower_map: return lower_map[c.lower()]
    return None

def find_session_type_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, *SESSION_TYPE_CANDIDATES)

def _norm_session_type(val: str) -> str:
    s = str(val).strip().lower()
    if not s or s == "nan": return ""
    if any(k in s for k in ["scrim", "intra", "fall ball", "exhib"]): return "scrimmage"
    if any(k in s for k in ["bullpen", "pen", "bp"]): return "bullpen"
    if any(k in s for k in ["game", "regular", "season", "conf", "non-conf", "ncaa"]): return "game"
    return ""

SEGMENT_DEFS = {
    "2025 Season":        {"start": "2025-02-01", "end": "2025-08-01", "types": ["game"]},
    "2025/26 Scrimmages": {"start": "2025-08-01", "end": "2026-02-01", "types": ["scrimmage"]},
    "2025/26 Bullpens":   {"start": "2025-08-01", "end": "2026-02-01", "types": ["bullpen"]},
    "2026 Season":        {"start": "2026-02-01", "end": "2026-08-01", "types": ["game"]},
}

def filter_by_segment(df: pd.DataFrame, segment_name: str) -> pd.DataFrame:
    spec = SEGMENT_DEFS.get(segment_name)
    if spec is None or df.empty: return df
    out = df.copy()
    if "Date" in out.columns:
        d = pd.to_datetime(out["Date"], errors="coerce")
        start = pd.to_datetime(spec["start"]); end = pd.to_datetime(spec["end"])
        out = out[(d >= start) & (d < end)]
    st_col = find_session_type_col(out)
    if st_col and len(spec.get("types", [])) > 0:
        st_norm = out[st_col].apply(_norm_session_type)
        out = out[st_norm.isin(spec.get("types", []))]
    return out

def get_type_col_for_segment(df: pd.DataFrame, segment_name: str) -> str:
    if "Scrimmages" in str(segment_name):
        return (pick_col(df, "TaggedPitchType","Tagged Pitch Type","PitchType","AutoPitchType","Auto Pitch Type")
                or "TaggedPitchType")
    return (pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType","Tagged Pitch Type")
            or "AutoPitchType")

def type_col_in_df(df: pd.DataFrame) -> str:
    seg = st.session_state.get("segment_choice", "")
    return get_type_col_for_segment(df, seg)

# ─── Strike zone helpers ──────────────────────────────────────────────────────
def get_zone_bounds():       return -0.83, 1.17, 1.66, 2.75
def get_view_bounds():
    l, b, w, h = get_zone_bounds(); mx, my = w*0.8, h*0.6
    return l-mx, l+w+mx, b-my, b+h+my

def draw_strikezone(ax, sz_left=None, sz_bottom=None, sz_width=None, sz_height=None):
    l, b, w, h = get_zone_bounds()
    sz_left   = l if sz_left   is None else sz_left
    sz_bottom = b if sz_bottom is None else sz_bottom
    sz_width  = w if sz_width  is None else sz_width
    sz_height = h if sz_height is None else sz_height
    ax.add_patch(Rectangle((sz_left, sz_bottom), sz_width, sz_height, fill=False, lw=2, color="black"))
    for f in (1/3, 2/3):
        ax.vlines(sz_left + sz_width*f, sz_bottom, sz_bottom+sz_height, colors="gray", ls="--", lw=1)
        ax.hlines(sz_bottom + sz_height*f, sz_left, sz_left+sz_width, colors="gray", ls="--", lw=1)

def get_pitch_color(ptype):
    if isinstance(ptype, str) and (ptype.lower().startswith("four-seam fastball") or ptype.lower()=="fastball"):
        return "#E60026"
    savant = {"sinker":"#FF9300","cutter":"#800080","changeup":"#008000","curveball":"#0033CC",
              "slider":"#CCCC00","splitter":"#00CCCC","knuckle curve":"#000000","screwball":"#CC0066","eephus":"#666666"}
    return savant.get(str(ptype).lower(), "#E60026")

def format_name(name):
    if isinstance(name, str) and ',' in name:
        last, first = [s.strip() for s in name.split(',', 1)]
        return f"{first} {last}"
    return str(name)

# ─── Name normalization & subset ──────────────────────────────────────────────
def _collapse_ws(s: str) -> str: return re.sub(r"\s+", " ", s).strip()

def canonicalize_person_name(raw) -> str:
    if pd.isna(raw): return ""
    s = str(raw).strip()
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        s = f"{first} {last}"
    return _collapse_ws(s)

def subset_by_pitcher_if_possible(df: pd.DataFrame, pitcher_display: str) -> pd.DataFrame:
    if "PitcherDisplay" in df.columns:
        sub = df[df["PitcherDisplay"] == pitcher_display]
        if not sub.empty: return sub.copy()
    pitch_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name","PitcherLastFirst") or "Pitcher"
    sub2 = df[df.get(pitch_col, "") == pitcher_display]
    return sub2.copy() if not sub2.empty else df.copy()

# ─── PBP helpers (inning/PA/AB tagging) ───────────────────────────────────────
def find_batter_name_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "Batter","BatterName","Batter Name","BatterFullName","Batter Full Name",
                    "Hitter","HitterName","BatterLastFirst","Batter First Last","BatterFirstLast")

def find_pitch_of_pa_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "PitchofPA","PitchOfPA","Pitch_of_PA","Pitch of PA","PitchOfPa","Pitch_of_Pa","Pitch # in AB")

def find_pa_of_inning_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "PAofinning","PAOfInning","PA_of_Inning","PA of Inning","PAofInng","PAOfInn","PA # in Inning")

def find_inning_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "Inning","inning","InningNumber","Inning #","InningNo","InningNum","Inng","Inn")

def find_pitch_of_game_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "PitchofGame","PitchOfGame","Pitch of Game","GamePitchNumber","GamePitchNo","PitchGameNo")

def find_pitch_no_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "PitchNo","Pitch No","Pitch_Number","Pitch Number","PitchIndex","Pitch #")

def find_datetime_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "DateTime","Datetime","Time","Timestamp","PitchTime","Pitch Time")

def _to_num(s): return pd.to_numeric(s, errors="coerce")

def _group_mode(series: pd.Series):
    s = series.dropna()
    if s.empty: return np.nan
    try:
        m = s.mode(); return m.iloc[0] if not m.empty else s.iloc[0]
    except Exception:
        return s.iloc[0]

def _normalize_inning_series(series: pd.Series) -> pd.Series:
    txt = series.astype(str); num = txt.str.extract(r'(\d+)')[0]
    return pd.to_numeric(num, errors="coerce").astype(pd.Int64Dtype())

def sort_for_pbp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); keys = []
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce"); keys.append("Date")
    inn_c = find_inning_col(out)
    if inn_c:
        out["_InningNumTmp"] = _normalize_inning_series(out[inn_c]); keys.append("_InningNumTmp")
    pa_c  = find_pa_of_inning_col(out)
    if pa_c: out[pa_c] = _to_num(out[pa_c]); keys.append(pa_c)
    po_c  = find_pitch_of_pa_col(out); pog = find_pitch_of_game_col(out); pno = find_pitch_no_col(out); dtc = find_datetime_col(out)
    if po_c:   out[po_c] = _to_num(out[po_c]); keys.append(po_c)
    elif pog:  out[pog]  = _to_num(out[pog]);  keys.append(pog)
    elif pno:  out[pno]  = _to_num(out[pno]);  keys.append(pno)
    elif dtc:  out[dtc]  = pd.to_datetime(out[dtc], errors="coerce"); keys.append(dtc)
    return out.sort_values(keys, kind="stable").reset_index(drop=True) if keys else out.reset_index(drop=True)

def add_inning_and_ab(df: pd.DataFrame) -> pd.DataFrame:
    out = sort_for_pbp(df)
    inn_c = find_inning_col(out); pa_c = find_pa_of_inning_col(out); po_c = find_pitch_of_pa_col(out)

    out["Inning #"] = _normalize_inning_series(out[inn_c]) if inn_c else pd.Series([pd.NA]*len(out), dtype="Int64")
    if pa_c:
        out[pa_c] = _to_num(out[pa_c]); out["PA # in Inning"] = out[pa_c].astype(pd.Int64Dtype())
    else:
        out["PA # in Inning"] = pd.Series([pd.NA]*len(out), dtype="Int64")

    if po_c is None:
        out["AB #"] = 1; out["Pitch # in AB"] = np.arange(1, len(out) + 1)
    else:
        is_start = (_to_num(out[po_c]) == 1); ab_id = is_start.cumsum()
        if (ab_id == 0).any(): ab_id = ab_id.replace(0, np.nan).ffill().fillna(1)
        out["AB #"] = ab_id.astype(int)
        out["Pitch # in AB"] = _to_num(out[po_c]).astype(pd.Int64Dtype())
        miss = out["Pitch # in AB"].isna()
        if miss.any():
            out.loc[miss, "Pitch # in AB"] = (out.loc[miss].groupby("AB #").cumcount() + 1).astype(pd.Int64Dtype())

    batter_c = find_batter_name_col(out); side_c = pick_col(out, "BatterSide","Batter Side","Bats","Stand","BatSide")
    if batter_c:
        names_by_ab = out.groupby("AB #")[batter_c].agg(_group_mode)
        out["Batter_AB"] = out["AB #"].map(names_by_ab).apply(format_name)
    if side_c:
        s_norm = out[side_c].astype(str).str.strip().str[0].str.upper().replace({"B":"S"})
        side_by_ab = s_norm.groupby(out["AB #"]).agg(_group_mode)
        out["BatterSide_AB"] = out["AB #"].map(side_by_ab)

    inn_by_ab = out.groupby("AB #")["Inning #"].agg(_group_mode)
    out["Inning #"] = out["AB #"].map(inn_by_ab).astype(pd.Int64Dtype())
    if pa_c:
        pa_by_ab = out.groupby("AB #")[pa_c].agg(_group_mode)
        out["PA # in Inning"] = out["AB #"].map(pa_by_ab).astype(pd.Int64Dtype())
    return out

# ─── PBP Table (KEEPS plate loc columns for plotting) ─────────────────────────
def build_pitch_by_inning_pa_table(df: pd.DataFrame) -> pd.DataFrame:
    work = add_inning_and_ab(df)

    type_col   = type_col_in_df(work)
    result_col = pick_col(work, "PitchCall","Pitch Call","Call") or "PitchCall"
    velo_col   = pick_col(work, "RelSpeed","Relspeed","ReleaseSpeed","RelSpeedMPH","release_speed")
    spin_col   = pick_col(work, "SpinRate","Spinrate","ReleaseSpinRate","Spin")
    ivb_col    = pick_col(work, "InducedVertBreak","IVB","Induced Vert Break","IndVertBreak")
    hb_col     = pick_col(work, "HorzBreak","HorizontalBreak","HB","HorizBreak")
    relh_col   = pick_col(work, "RelHeight","Relheight","ReleaseHeight","Release_Height","release_pos_z")
    ext_col    = pick_col(work, "Extension","Ext","ReleaseExtension","ExtensionInFt","Extension(ft)")

    # Plate location columns (so PA plot can read them)
    x_col      = pick_col(work, "PlateLocSide","Plate Loc Side","PlateSide","px","PlateLocX")
    y_col      = pick_col(work, "PlateLocHeight","Plate Loc Height","PlateHeight","pz","PlateLocZ")

    batter_col = "Batter_AB" if "Batter_AB" in work.columns else find_batter_name_col(work)
    side_col   = "BatterSide_AB" if "BatterSide_AB" in work.columns else pick_col(work, "BatterSide","Batter Side","Bats","Stand","BatSide")

    col_play_result = pick_col(work, "PlayResult","Result","Event","PAResult","Outcome")
    col_korbb       = pick_col(work, "KorBB","K_BB","KBB","K_or_BB","PA_KBB")
    col_pitch_call  = result_col

    for c in [col_play_result, col_korbb, col_pitch_call]:
        if c and c in work.columns:
            work[c] = work[c].fillna("").astype(str)

    def _terminal_row_idx(g: pd.DataFrame) -> int:
        if col_play_result and g[col_play_result].str.strip().ne("").any():
            return g[g[col_play_result].str.strip().ne("")].index[-1]
        if col_korbb and g[col_korbb].str.strip().ne("").any():
            return g[g[col_korbb].str.strip().ne("")].index[-1]
        if col_pitch_call and g[col_pitch_call].str.lower().isin({"hitbypitch","hit by pitch","hbp"}).any():
            return g[g[col_pitch_call].str.lower().isin({"hitbypitch","hit by pitch","hbp"})].index[-1]
        if "Pitch # in AB" in g.columns and g["Pitch # in AB"].notna().any():
            return g["Pitch # in AB"].astype("Int64").idxmax()
        return g.index[-1]

    def _pa_label(row) -> str:
        pr = row.get(col_play_result, "")
        if isinstance(pr, str) and pr.strip(): return pr.strip()
        kb = row.get(col_korbb, "")
        if isinstance(kb, str):
            low = kb.strip().lower()
            if low in {"k","so","strikeout","strikeout swinging","strikeout looking"}: return "Strikeout"
            if "walk" in low or low in {"bb","ibb"}: return "Walk"
        pc = row.get(col_pitch_call, "")
        if isinstance(pc, str) and pc.strip().lower() in {"hitbypitch","hit by pitch","hbp"}: return "Hit By Pitch"
        return "—"

    idx_by_ab = work.groupby("AB #", sort=True, dropna=False).apply(_terminal_row_idx)
    pa_row = work.loc[idx_by_ab.values].copy()
    pa_row["PA Result"] = pa_row.apply(_pa_label, axis=1)
    work = work.merge(pa_row[["AB #","PA Result"]], on="AB #", how="left")

    ordered = [
        "Inning #","PA # in Inning","AB #","Pitch # in AB",
        batter_col, "PA Result", type_col, result_col, velo_col, spin_col, ivb_col, hb_col, relh_col, ext_col,
        # KEEP these for plotting (they won't show in table if we slice later)
        x_col, y_col
    ]
    present = [c for c in ordered if c and c in work.columns]
    tbl = work[present].copy()

    rename_map = {
        batter_col: "Batter", side_col: "Batter Side",
        type_col: "Pitch Type", result_col: "Result",
        velo_col: "Velo", spin_col: "Spin Rate", ivb_col: "IVB", hb_col: "HB",
        relh_col: "Rel Height", ext_col: "Extension"
    }
    for k, v in list(rename_map.items()):
        if k and k in tbl.columns: tbl = tbl.rename(columns={k: v})

    # numeric rounding
    for c in ["Velo","Spin Rate","IVB","HB","Rel Height","Extension"]:
        if c in tbl.columns: tbl[c] = pd.to_numeric(tbl[c], errors="coerce")
    if "Velo" in tbl:       tbl["Velo"] = tbl["Velo"].round(1)
    if "Spin Rate" in tbl:  tbl["Spin Rate"] = tbl["Spin Rate"].round(0)
    if "IVB" in tbl:        tbl["IVB"] = tbl["IVB"].round(1)
    if "HB" in tbl:         tbl["HB"] = tbl["HB"].round(1)
    if "Rel Height" in tbl: tbl["Rel Height"] = tbl["Rel Height"].round(2)
    if "Extension" in tbl:  tbl["Extension"] = tbl["Extension"].round(2)

    sort_cols = [c for c in ["Inning #","PA # in Inning","AB #","Pitch # in AB"] if c in tbl.columns]
    if sort_cols: tbl = tbl.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return tbl

# ─── Styling helpers ──────────────────────────────────────────────────────────
def themed_table(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    integer_like_names = {"Pitches","Zone Pitches","Hits","Strikeouts","Walks","AB","PA","Plate Appearances","Zone Swings","Zone Contacts"}
    integer_like = set(c for c in numeric_cols if (c in integer_like_names) or c.lower().endswith(" pitches") or c.lower().endswith(" counts") or c.lower().endswith(" count"))
    for c in numeric_cols:
        if pd.api.types.is_integer_dtype(df[c]): integer_like.add(c)
    percent_cols_numeric = [c for c in numeric_cols if c.strip().endswith('%')]
    fmt_map = {}
    for c in numeric_cols:
        if c in integer_like: fmt_map[c] = "{:.0f}"
        elif c in percent_cols_numeric: fmt_map[c] = "{:.1f}"
        else: fmt_map[c] = "{:.1f}"
    styles = [
        {'selector': 'thead th', 'props': f'background: linear-gradient(135deg, {HUSKER_RED} 0%, #B8001F 100%); color: white; font-weight: 600; text-align: center; padding: 12px 8px; border: none;'},
        {'selector': 'th',        'props': f'background: linear-gradient(135deg, {HUSKER_RED} 0%, #B8001F 100%); color: white; font-weight: 600; text-align: center; padding: 12px 8px; border: none;'},
        {'selector': 'td',        'props': 'white-space: nowrap; color: #2B2B2B; padding: 10px 8px; border-bottom: 1px solid #E0E0E0;'},
        {'selector': 'tr:hover',  'props': 'background-color: #F8F9FA;'},
    ]
    return (df.style.hide(axis="index").format(fmt_map, na_rep="—").set_table_styles(styles))

def style_pbp_expanders():
    st.markdown(
        f"""
        <style>
        .pbp-scope div[data-testid="stExpander"] > details > summary {{
            background-color: #FFFFFF !important; color: #2B2B2B !important; border-radius: 6px !important;
            padding: 10px 14px !important; font-weight: 600 !important; border: 1px solid #E0E0E0 !important;
            transition: all 0.3s ease;
        }}
        .pbp-scope div[data-testid="stExpander"] > details > summary:hover {{
            background-color: #F8F9FA !important; border-color: {HUSKER_RED} !important;
        }}
        .pbp-scope .inning-block > div[data-testid="stExpander"] > details > summary {{
            background: linear-gradient(135deg, {HUSKER_RED} 0%, #B8001F 100%) !important; 
            color: #FFFFFF !important; border: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ─── Movement summary (kept) ──────────────────────────────────────────────────
def combined_pitcher_report(df, pitcher_name, logo_img, coverage=0.8, season_label="Season"):
    type_col = type_col_in_df(df)
    pitch_col = pick_col(df, "PitchCall","Pitch Call","Call") or "PitchCall"
    speed_col = pick_col(df, "RelSpeed","Relspeed","ReleaseSpeed","RelSpeedMPH","release_speed")
    spin_col  = pick_col(df, "SpinRate","Spinrate","ReleaseSpinRate","Spin")
    ivb_col   = pick_col(df, "InducedVertBreak","IVB","Induced Vert Break","IndVertBreak")
    hb_col    = pick_col(df, "HorzBreak","HorizontalBreak","HB","HorizBreak")
    rh_col    = pick_col(df, "RelHeight","Relheight","ReleaseHeight","Release_Height","release_pos_z")
    vaa_col   = pick_col(df, "VertApprAngle","VAA","VerticalApproachAngle")
    ext_col   = pick_col(df, "Extension","Ext","ReleaseExtension","ExtensionInFt","Extension(ft)")

    df_p = subset_by_pitcher_if_possible(df, pitcher_name)
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' with the current filters.")
        return None

    try:
        grp = df_p.groupby(type_col, dropna=False)
    except KeyError:
        st.error(f"Pitch type column not found (tried '{type_col}').")
        return None

    counts = grp.size(); total = int(len(df_p))
    summary = pd.DataFrame({
        'Pitch Type': counts.index.astype(str),
        'Pitches': counts.values,
        'Usage %': np.round((counts.values / max(total, 1)) * 100, 1),
    })

    if pitch_col in df_p.columns:
        is_strike = df_p[pitch_col].isin(['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
        strike_pct = grp.apply(lambda g: is_strike.loc[g.index].mean() * 100 if len(g) else np.nan).values
        summary['Strike %'] = np.round(strike_pct, 1)

    def add_mean(col_name, label, r=1):
        if col_name and col_name in df_p.columns:
            vals = grp[col_name].mean().values; summary[label] = np.round(vals, r)
    add_mean(speed_col, 'Rel Speed', 1); add_mean(spin_col, 'Spin Rate', 1)
    add_mean(ivb_col, 'IVB', 1); add_mean(hb_col, 'HB', 1)
    add_mean(rh_col, 'Rel Height', 2); add_mean(vaa_col, 'VAA', 1)
    add_mean(ext_col, 'Extension', 2)
    summary = summary.sort_values('Pitches', ascending=False)

    fig = plt.figure(figsize=(8, 12)); gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.7], hspace=0.3)
    axm = fig.add_subplot(gs[0, 0]); axm.set_title('Movement Profile', fontweight='bold', fontsize=14)
    axm.axhline(0, ls='--', color='grey', alpha=0.5); axm.axvline(0, ls='--', color='grey', alpha=0.5)
    chi2v = chi2.ppf(coverage, df=2)

    for ptype, g in df_p.groupby(type_col, dropna=False):
        clr = get_pitch_color(ptype)
        x = pd.to_numeric(g.get('HorzBreak', g.get('HB')), errors='coerce')
        y = pd.to_numeric(g.get('InducedVertBreak', g.get('IVB')), errors='coerce')
        mask = x.notna() & y.notna()
        if mask.any():
            axm.scatter(x[mask], y[mask], label=str(ptype), color=clr, alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            if mask.sum() > 1:
                X = np.vstack((x[mask], y[mask])); cov = np.cov(X)
                try:
                    vals, vecs = np.linalg.eigh(cov)
                    ord_ = vals.argsort()[::-1]; vals, vecs = vals[ord_], vecs[:, ord_]
                    ang = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                    w, h = 2*np.sqrt(vals*chi2v)
                    axm.add_patch(Ellipse((x[mask].mean(), y[mask].mean()), w, h, angle=ang,
                                          edgecolor=clr, facecolor=clr, alpha=0.15, ls='--', lw=1.5))
                except Exception:
                    pass
        else:
            axm.scatter([], [], label=str(ptype), color=clr, alpha=0.7, s=60)

    axm.set_xlim(-30,30); axm.set_ylim(-30,30); axm.set_aspect('equal','box')
    axm.set_xlabel('Horizontal Break (inches)', fontsize=11, fontweight='500')
    axm.set_ylabel('Induced Vertical Break (inches)', fontsize=11, fontweight='500')
    axm.legend(title='Pitch Type', fontsize=9, title_fontsize=10, loc='upper right', framealpha=0.95)
    axm.grid(True, alpha=0.2, linestyle=':')

    axt = fig.add_subplot(gs[1, 0]); axt.axis('off')
    tbl = axt.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.5, 1.5)
    
    # Professional table styling
    for i in range(len(summary.columns)):
        tbl[(0, i)].set_facecolor(HUSKER_RED)
        tbl[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary) + 1):
        for j in range(len(summary.columns)):
            tbl[(i, j)].set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')
            tbl[(i, j)].set_edgecolor('#E0E0E0')
    
    axt.set_title('Performance Metrics by Pitch Type', fontweight='bold', y=0.88, fontsize=12)

    logo_img = load_logo_img()
    if logo_img is not None:
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10); axl.imshow(logo_img); axl.axis('off')

    fig.suptitle(f"{canonicalize_person_name(pitcher_name)}\n{season_label}", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, summary

# [CONTINUED IN NEXT MESSAGE DUE TO LENGTH - This contains the helpers and new features]# ─── Per-PA interactive strike zone (Plotly) ──────────────────────────────────
def _zone_shapes_for_subplot():
    l, b, w, h = get_zone_bounds()
    x0, x1, y0, y1 = l, l+w, b, b+h
    thirds_x = [x0 + w/3, x0 + 2*w/3]
    thirds_y = [y0 + h/3, y0 + 2*h/3]
    return [
        dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color="black", width=2)),
        dict(type="line", x0=thirds_x[0], x1=thirds_x[0], y0=y0, y1=y1, line=dict(color="gray", dash="dash")),
        dict(type="line", x0=thirds_x[1], x1=thirds_x[1], y0=y0, y1=y1, line=dict(color="gray", dash="dash")),
        dict(type="line", x0=x0, x1=x1, y0=thirds_y[0], y1=thirds_y[0], line=dict(color="gray", dash="dash")),
        dict(type="line", x0=x0, x1=x1, y0=thirds_y[1], y1=thirds_y[1], line=dict(color="gray", dash="dash")),
    ]

def pa_interactive_strikezone(pa_df: pd.DataFrame, title: str | None = None):
    """
    Plot a single-PA interactive strike zone using PlateLocSide/PlateLocHeight from the PA's pitches.
    """
    if pa_df is None or pa_df.empty: return None

    # Resolve columns (robust to variants)
    type_col = pick_col(pa_df, type_col_in_df(pa_df), "Pitch Type","TaggedPitchType","PitchType","AutoPitchType","Auto Pitch Type")
    speed_col = pick_col(pa_df, "RelSpeed","Relspeed","ReleaseSpeed","RelSpeedMPH","release_speed")
    ivb_col   = pick_col(pa_df, "InducedVertBreak","IVB","Induced Vert Break","IndVertBreak")
    hb_col    = pick_col(pa_df, "HorzBreak","HorizontalBreak","HB","HorizBreak")
    exit_col  = pick_col(pa_df, "ExitSpeed","Exit Velo","ExitVelocity","Exit_Velocity","ExitVel","EV","LaunchSpeed","Launch_Speed")
    call_col  = pick_col(pa_df, "PitchCall","Pitch Call","PitchResult","Call")
    pno_col   = pick_col(pa_df, "Pitch # in AB","PitchofPA","PitchOfPA","Pitch_of_PA","Pitch #")
    x_col     = pick_col(pa_df, "PlateLocSide","Plate Loc Side","PlateSide","px","PlateLocX")
    y_col     = pick_col(pa_df, "PlateLocHeight","Plate Loc Height","PlateHeight","pz","PlateLocZ")

    xs = pd.to_numeric(pa_df.get(x_col, pd.Series(dtype=float)), errors="coerce")
    ys = pd.to_numeric(pa_df.get(y_col, pd.Series(dtype=float)), errors="coerce")
    if xs.isna().all() or ys.isna().all(): return None  # no plate-loc data in this PA

    x_min, x_max, y_min, y_max = get_view_bounds()
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)
    for shp in _zone_shapes_for_subplot():
        fig.add_shape(shp, row=1, col=1)

    cd = np.column_stack([
        pa_df.get(type_col, pd.Series(dtype=object)).astype(str).values if type_col else np.array([""]*len(pa_df)),
        pd.to_numeric(pa_df.get(speed_col, pd.Series(dtype=float)), errors="coerce").values if speed_col else np.full(len(pa_df), np.nan),
        pd.to_numeric(pa_df.get(ivb_col,   pd.Series(dtype=float)), errors="coerce").values if ivb_col   else np.full(len(pa_df), np.nan),
        pd.to_numeric(pa_df.get(hb_col,    pd.Series(dtype=float)), errors="coerce").values if hb_col    else np.full(len(pa_df), np.nan),
        pa_df.get(call_col, pd.Series(dtype=object)).astype(str).values if call_col else np.array([""]*len(pa_df)),
        pd.to_numeric(pa_df.get(exit_col,  pd.Series(dtype=float)), errors="coerce").values if exit_col  else np.full(len(pa_df), np.nan),
        pd.to_numeric(pa_df.get(pno_col,   pd.Series(dtype=float)), errors="coerce").values if pno_col   else np.full(len(pa_df), np.nan),
    ])

    if type_col and type_col in pa_df.columns:
        colors_pts = [get_pitch_color(t) for t in pa_df[type_col].astype(str).tolist()]
    else:
        colors_pts = [HUSKER_RED] * len(pa_df)

    fig.add_trace(
        go.Scattergl(
            x=xs, y=ys,
            mode="markers+text",
            text=[str(int(n)) if pd.notna(n) else "" for n in cd[:,6]],
            textposition="top center",
            marker=dict(size=12, line=dict(width=1, color="white"), color=colors_pts),
            customdata=cd,
            hovertemplate=(
                "<b>Pitch %{customdata[6]:.0f}</b><br>"
                "Type: %{customdata[0]}<br>"
                "Velocity: %{customdata[1]:.1f} mph<br>"
                "IVB: %{customdata[2]:.1f}\"<br>"
                "HB: %{customdata[3]:.1f}\"<br>"
                "Result: %{customdata[4]}<br>"
                "Exit Velo: %{customdata[5]:.1f} mph<br>"
                "<extra></extra>"
            ),
            showlegend=False,
            name=""
        ),
        row=1, col=1
    )

    fig.update_xaxes(range=[x_min, x_max], showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[y_min, y_max], showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_layout(
        height=400, 
        title_text=(title or "Plate Appearance Strike Zone"), 
        title_x=0.5,
        title_font=dict(size=16, color=DARK_GRAY, family="Arial Black"),
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

# ─── Outcome summary helpers (needed in Profiles) ─────────────────────────────
def _first_present(df: pd.DataFrame, cands: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    return None

def _is_terminal_row(row, col_result, col_korbb, col_call) -> bool:
    pr = str(row.get(col_result, "")) if col_result else ""
    kc = str(row.get(col_korbb, "")) if col_korbb else ""
    pc = str(row.get(col_call, ""))  if col_call  else ""
    return (
        (pr.strip() != "") or
        (kc.lower() in {"k","so","strikeout","strikeout swinging","strikeout looking","bb","walk"}) or
        (pc.lower() in {"hitbypitch","hit by pitch","hbp"})
    )

def _pct(x):   return f"{x*100:.1f}%" if pd.notna(x) else ""
def _rate3(x): return f"{x:.3f}" if pd.notna(x) else ""

def strike_rate(df):
    if len(df) == 0 or "PitchCall" not in df.columns: return np.nan
    strike_calls = ['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']
    return df['PitchCall'].isin(strike_calls).mean() * 100

def make_pitcher_outcome_summary_table(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame([{
            "Average exit velo": np.nan, "Max exit velo": np.nan, "Hits": 0, "Strikeouts": 0,
            "AVG":"", "OBP":"", "SLG":"", "OPS":"", "HardHit%":"", "K%":"", "Walk%":""
        }])

    col_exitv  = _first_present(df_in, ["ExitSpeed","Exit Velo","ExitVelocity","Exit_Velocity","ExitVel","EV","LaunchSpeed","Launch_Speed"])
    col_result = _first_present(df_in, ["PlayResult","Result","Event","PAResult","Outcome"])
    col_call   = _first_present(df_in, ["PitchCall","Pitch Call","PitchResult","Call"])
    col_korbb  = _first_present(df_in, ["KorBB","K_BB","KBB","K_or_BB","PA_KBB"])

    work = add_inning_and_ab(df_in.copy())
    po_c = find_pitch_of_pa_col(work)

    for c in [col_result, col_call, col_korbb]:
        if c and c in work.columns:
            if work[c].dtype != "O": work[c] = work[c].astype("string")
            work[c] = work[c].fillna("").astype(str)

    is_term = work.apply(lambda r: _is_terminal_row(r, col_result, col_korbb, col_call), axis=1) \
             if any(c for c in [col_result, col_korbb, col_call]) else pd.Series(False, index=work.index)

    def _pick_row_idx(g: pd.DataFrame) -> int:
        gm = is_term.loc[g.index]
        if gm.any(): return gm[gm].index[-1]
        if po_c and po_c in g.columns:
            if "Pitch # in AB" in g.columns and g["Pitch # in AB"].notna().any():
                return g["Pitch # in AB"].astype("Int64").idxmax()
        return g.index[-1]

    ab_rows_idx = work.groupby("AB #", sort=True, dropna=False).apply(_pick_row_idx).values
    df_pa = work.loc[ab_rows_idx].copy()

    PR = df_pa[col_result].astype(str) if col_result else pd.Series([""]*len(df_pa), index=df_pa.index)
    KC = df_pa[col_korbb].astype(str)  if col_korbb else pd.Series([""]*len(df_pa), index=df_pa.index)
    PC = df_pa[col_call].astype(str)   if col_call  else pd.Series([""]*len(df_pa), index=df_pa.index)

    pr_low = PR.str.lower()
    is_single = pr_low.str.contains(r"\bsingle\b", regex=True)
    is_double = pr_low.str.contains(r"\bdouble\b", regex=True)
    is_triple = pr_low.str.contains(r"\btriple\b", regex=True)
    is_hr     = pr_low.str.contains(r"\bhome\s*run\b", regex=True) | pr_low.eq("hr")
    hits_mask = is_single | is_double | is_triple | is_hr
    TB = (is_single.astype(int)*1 + is_double.astype(int)*2 + is_triple.astype(int)*3 + is_hr.astype(int)*4).sum()

    is_bb  = (pr_low.str.contains(r"\bwalk\b|intentional\s*walk|int\.?\s*bb|ib[bB]\b", regex=True)
              | KC.str.lower().isin({"bb","walk","ibb","intentional walk"})
              | KC.str.contains(r"\bwalk\b", case=False, regex=True))
    is_so  = (pr_low.str.contains(r"strikeout", case=False, regex=True)
              | KC.str.lower().isin({"k","so","strikeout","strikeout swinging","strikeout looking"}))
    is_hbp = (pr_low.str.contains(r"hit\s*by\s*pitch", case=False, regex=True)
              | PC.str.lower().isin({"hitbypitch","hit by pitch","hbp"}))
    is_sf  = pr_low.str.contains(r"sac(rifice)?\s*fly|\bsf\b", regex=True)
    is_sh  = pr_low.str.contains(r"sac(rifice)?\s*(bunt|hit)|\bsh\b", regex=True)
    is_ci  = pr_low.str.contains(r"interference", regex=True)

    PA  = int(len(df_pa)); H = int(hits_mask.sum()); BB = int(is_bb.sum()); SO = int(is_so.sum())
    HBP = int(is_hbp.sum()); SF = int(is_sf.sum()); SH = int(is_sh.sum()); CI = int(is_ci.sum())
    AB  = max(PA - (BB + HBP + SF + SH + CI), 0)

    AVG = (H / AB) if AB > 0 else np.nan
    OBP = ((H + BB + HBP) / (AB + BB + HBP + SF)) if (AB + BB + HBP + SF) > 0 else np.nan
    SLG = (TB / AB) if AB > 0 else np.nan
    OPS = (OBP + SLG) if (pd.notna(OBP) and pd.notna(SLG)) else np.nan

    K_rate  = (SO / PA) if PA > 0 else np.nan
    BB_rate = (BB / PA) if PA > 0 else np.nan

    if col_exitv:
        ev_all = pd.to_numeric(df_in[col_exitv], errors="coerce").dropna()
        avg_ev = float(ev_all.mean()) if len(ev_all) else np.nan
        max_ev = float(ev_all.max())  if len(ev_all) else np.nan
        hard_hit_pct = float((ev_all >= 95.0).mean()) if len(ev_all) else np.nan
    else:
        avg_ev = max_ev = hard_hit_pct = np.nan

    row = {
        "Average exit velo": round(avg_ev, 1) if pd.notna(avg_ev) else np.nan,
        "Max exit velo":     round(max_ev, 1) if pd.notna(max_ev) else np.nan,
        "Hits":              H, "Strikeouts": SO,
        "AVG": _rate3(AVG), "OBP": _rate3(OBP), "SLG": _rate3(SLG), "OPS": _rate3(OPS),
        "HardHit%": _pct(hard_hit_pct), "K%": _pct(K_rate), "Walk%": _pct(BB_rate),
    }
    return pd.DataFrame([row])

# ─── Rankings helpers ─────────────────────────────────────────────────────────
def _first_present_strict(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns: return n
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower: return lower[n.lower()]
    return None

def _to_float(s): return pd.to_numeric(s, errors="coerce")

def _terminal_pa_table(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return one row per AB (terminal row), robust to your various column names."""
    if df_in is None or df_in.empty: return df_in.iloc[0:0].copy()
    work = add_inning_and_ab(df_in.copy())
    col_result = _first_present_strict(work, ["PlayResult","Result","Event","PAResult","Outcome"])
    col_call   = _first_present_strict(work, ["PitchCall","Pitch Call","PitchResult","Call"])
    col_korbb  = _first_present_strict(work, ["KorBB","K_BB","KBB","K_or_BB","PA_KBB"])
    for c in [col_result, col_call, col_korbb]:
        if c and c in work.columns:
            work[c] = work[c].fillna("").astype(str)

    po_c = find_pitch_of_pa_col(work)
    term_mask = work.apply(lambda r: _is_terminal_row(r, col_result, col_korbb, col_call), axis=1) \
                if any([col_result, col_korbb, col_call]) else pd.Series(False, index=work.index)

    def _pick_row_idx(g: pd.DataFrame) -> int:
        gm = term_mask.loc[g.index]
        if gm.any(): return gm[gm].index[-1]
        if po_c and po_c in g.columns and g["Pitch # in AB"].notna().any():
            return g["Pitch # in AB"].astype("Int64").idxmax()
        return g.index[-1]

    ab_rows_idx = work.groupby("AB #", sort=True, dropna=False).apply(_pick_row_idx).values
    out = work.loc[ab_rows_idx].copy()
    out["_PlayResult"] = out[col_result] if col_result else ""
    out["_PitchCall"]  = out[col_call]   if col_call   else ""
    out["_KorBB"]      = out[col_korbb]  if col_korbb  else ""
    return out

def _compute_IP_from_outs(total_outs: int) -> tuple[float, str]:
    """Returns (IP_float_for_rates, IP_display_baseball) where display shows .1/.2 for 1/2 outs."""
    ip_float = total_outs / 3.0
    whole = total_outs // 3
    rem   = total_outs % 3
    ip_disp = f"{whole}.{rem}"  # baseball notation
    return ip_float, ip_disp

def _box_counts_from_PA(pa_df: pd.DataFrame) -> dict:
    """Count H, HR, BB, SO, HBP, OUTS from PA-level terminal rows."""
    if pa_df is None or pa_df.empty:
        return dict(H=0, HR=0, BB=0, SO=0, HBP=0, OUTS=0)

    PR = pa_df["_PlayResult"].astype(str).str.lower()
    KC = pa_df["_KorBB"].astype(str).str.lower() if "_KorBB" in pa_df.columns else pd.Series([""]*len(pa_df), index=pa_df.index)
    PC = pa_df["_PitchCall"].astype(str).str.lower() if "_PitchCall" in pa_df.columns else pd.Series([""]*len(pa_df), index=pa_df.index)

    # Hits
    is_single = PR.str.contains(r"\bsingle\b")
    is_double = PR.str.contains(r"\bdouble\b")
    is_triple = PR.str.contains(r"\btriple\b")
    is_hr     = PR.str.contains(r"\bhome\s*run\b") | PR.eq("hr")

    # BB / SO / HBP (with KorBB and PitchCall backups)
    is_bb  = (PR.str.contains(r"\bwalk\b|intentional\s*walk|ib[bB]\b")
              | KC.isin({"bb","walk","ibb","intentional walk"})
              | KC.str.contains(r"\bwalk\b"))
    is_so  = (PR.str.contains(r"strikeout")
              | KC.isin({"k","so","strikeout","strikeout swinging","strikeout looking"}))
    is_hbp = (PR.str.contains(r"hit\s*by\s*pitch")
              | PC.isin({"hitbypitch","hit by pitch","hbp"}))

    # Outs (heuristic from PlayResult text)
    is_dp  = PR.str.contains("double play")
    is_tp  = PR.str.contains("triple play")
    is_outword = PR.str.contains("out") | PR.str.contains("groundout") | PR.str.contains("flyout") \
                 | PR.str.contains("lineout") | PR.str.contains("popout") | PR.str.contains("forceout") \
                 | PR.str.contains("fielder'?s choice") | PR.str.contains("reached on error and out")

    outs = (is_so.astype(int)*1 + is_dp.astype(int)*2 + is_tp.astype(int)*3)
    # Count 1 out for generic "out" rows that aren't K/DP/TP and not hits/BB/HBP
    base_hit_like = is_single | is_double | is_triple | is_hr | is_bb | is_hbp
    outs += ((~base_hit_like) & is_outword & (~is_so) & (~is_dp) & (~is_tp)).astype(int)*1

    return dict(
        H = int((is_single | is_double | is_triple | is_hr).sum()),
        HR= int(is_hr.sum()),
        BB= int(is_bb.sum()),
        SO= int(is_so.sum()),
        HBP= int(is_hbp.sum()),
        OUTS=int(outs.sum()),
    )

def _plate_metrics_detailed(sub: pd.DataFrame) -> dict:
    """Return Zone%, Zwhiff%, Chase%, Whiff%, Strike% (all as 0..100 floats)."""
    s_call = sub.get('PitchCall', pd.Series(dtype=object))
    xs = _to_float(sub.get('PlateLocSide',   pd.Series(dtype=float)))
    ys = _to_float(sub.get('PlateLocHeight', pd.Series(dtype=float)))

    isswing   = s_call.isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
    iswhiff   = s_call.eq('StrikeSwinging')

    # Zone bounds consistent with your draw
    z_left, z_bot, z_w, z_h = get_zone_bounds()
    isinzone = xs.between(z_left, z_left+z_w) & ys.between(z_bot, z_bot+z_h)

    swingsZ = int(isswing[isinzone].sum())

    zone_pct   = 100.0 * float(isinzone.mean()) if len(sub) else np.nan
    zwhiff_pct = 100.0 * (iswhiff[isinzone].sum() / swingsZ) if swingsZ > 0 else np.nan
    chase_pct  = 100.0 * float(isswing[~isinzone].mean()) if (~isinzone).any() else np.nan
    whiff_pct  = 100.0 * float(iswhiff.sum() / max(int(isswing.sum()),1)) if int(isswing.sum())>0 else np.nan
    strike_pct = strike_rate(sub)  # uses your helper

    return dict(ZonePct=zone_pct, ZwhiffPct=zwhiff_pct, ChasePct=chase_pct, WhiffPct=whiff_pct, StrikePct=strike_pct)

def _hardhit_barrel_metrics(sub: pd.DataFrame) -> dict:
    """
    Returns HardHitPct and BarrelPct for a subset of pitches/batted balls.

    Definitions:
      • HardHitPct: share of balls in play with EV ≥ 95 mph
      • BarrelPct (college): share of balls in play with EV ≥ 95 mph AND 10° ≤ LA ≤ 35°
    Denominator for both = balls in play only.
    """
    ev_col = _first_present_strict(sub, ["ExitSpeed","Exit Velo","ExitVelocity","Exit_Velocity","ExitVel","EV","LaunchSpeed","Launch_Speed"])
    la_col = _first_present_strict(sub, ["LaunchAngle","Launch_Angle","LA","Angle","LaunchAngleDeg"])
    call   = _first_present_strict(sub, ["PitchCall","Pitch Call","PitchResult","Call"])
    if call is None:
        return dict(HardHitPct=np.nan, BarrelPct=np.nan)

    # Balls in play only
    inplay = sub[sub[call].astype(str).eq("InPlay")]

    # Hard-hit %
    if ev_col:
        ev_bip = _to_float(inplay[ev_col]).dropna()
        hh = float((ev_bip >= 95).mean()) * 100 if len(ev_bip) else np.nan
    else:
        hh = np.nan

    # College barrel %: EV ≥ 95 and 10°–35° (inclusive), BIP denominator
    if ev_col and la_col:
        ev_all = _to_float(inplay[ev_col])
        la_all = _to_float(inplay[la_col])
        mask = ev_all.notna() & la_all.notna()
        if mask.any():
            barrel_mask = (ev_all[mask] >= 95.0) & la_all[mask].between(10.0, 35.0, inclusive="both")
            br = float(barrel_mask.mean()) * 100 if barrel_mask.size else np.nan
        else:
            br = np.nan
    else:
        br = np.nan

    return dict(HardHitPct=hh, BarrelPct=br)

def make_pitcher_rankings(df_segment: pd.DataFrame, pitch_types_filter: list[str] | None = None) -> pd.DataFrame:
    """
    Build rankings across NEB pitchers with requested columns:
    App, IP, H, HR, BB, HBP, SO, WHIP, H9, BB%, SO%, Strike%, HH%, Barrel%, Zone%, Zwhiff%, Chase%, Whiff%
    """
    if df_segment is None or df_segment.empty:
        return pd.DataFrame()

    # Restrict to NEB pitchers
    base = df_segment[df_segment.get('PitcherTeam','') == 'NEB'].copy()
    if base.empty: return pd.DataFrame()

    type_col = type_col_in_df(base)
    if pitch_types_filter is not None and len(pitch_types_filter) > 0 and type_col in base.columns:
        base = base[base[type_col].astype(str).isin(pitch_types_filter)].copy()
        if base.empty: return pd.DataFrame()

    # Build Date column (already normalized earlier), and per-pitcher key/display
    base['Date'] = pd.to_datetime(base['Date'], errors='coerce')
    base["PitcherDisplay"] = base.get("Pitcher", pd.Series(dtype=object)).map(canonicalize_person_name)
    base["PitcherKey"]     = base["PitcherDisplay"].str.lower()

    rows = []
    for pkey, sub in base.groupby("PitcherKey", dropna=False):
        name = sub["PitcherDisplay"].iloc[0] if "PitcherDisplay" in sub.columns else str(pkey)

        # Appearances
        app = int(sub['Date'].dropna().dt.date.nunique())

        # PA-level table for counting H/BB/SO/etc + outs → IP
        pa = _terminal_pa_table(sub)
        box = _box_counts_from_PA(pa)
        outs = box["OUTS"]
        ip_float, ip_disp = _compute_IP_from_outs(outs)

        # Plate discipline + Strike%
        pdm = _plate_metrics_detailed(sub)

        # HardHit / Barrel
        hhbm = _hardhit_barrel_metrics(sub)

        # Derived
        H, HR, BB, HBP, SO = box["H"], box["HR"], box["BB"], box["HBP"], box["SO"]
        WHIP = (BB + H) / ip_float if ip_float > 0 else np.nan
        H9   = (H * 9.0) / ip_float if ip_float > 0 else np.nan

        rows.append({
            "Pitcher": name,
            "App": app,
            "IP": ip_disp,                    # baseball display (e.g., 5.2)
            "_IP_num": ip_float,              # keep numeric for sorting/ratios
            "H": H, "HR": HR, "BB": BB, "HBP": HBP, "SO": SO,
            "WHIP": WHIP, "H9": H9,
            "BB%": (BB / max(len(pa),1))*100 if len(pa) else np.nan,
            "SO%": (SO / max(len(pa),1))*100 if len(pa) else np.nan,
            "Strike%": pdm["StrikePct"],
            "HH%": hhbm["HardHitPct"],
            "Barrel%": hhbm["BarrelPct"],
            "Zone%": pdm["ZonePct"],
            "Zwhiff%": pdm["ZwhiffPct"],
            "Chase%": pdm["ChasePct"],
            "Whiff%": pdm["WhiffPct"],
        })

    out = pd.DataFrame(rows)
    if out.empty: return out

    # Order & formatting
    desired_order = ["Pitcher","App","IP","H","HR","BB","HBP","SO","WHIP","H9",
                     "BB%","SO%","Strike%","HH%","Barrel%","Zone%","Zwhiff%","Chase%","Whiff%"]
    for c in desired_order:
        if c not in out.columns:
            out[c] = np.nan
    out = out[desired_order + ["_IP_num"]]

    # Round numeric columns
    for c in ["WHIP","H9"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    for c in ["BB%","SO%","Strike%","HH%","Barrel%","Zone%","Zwhiff%","Chase%","Whiff%"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(1)

    # Integers
    for c in ["App","H","HR","BB","HBP","SO"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # Default sort by WHIP asc, then SO desc
    out = out.sort_values(["WHIP","SO"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return out

def make_team_averages(df_segment: pd.DataFrame, pitch_types_filter: list[str] | None = None) -> pd.DataFrame:
    """
    Team-level numbers for NEB using the same logic as make_pitcher_rankings:
    WHIP, H9, BB%, SO%, Strike%, Zone%, Zwhiff%, HH%, Barrel%, Whiff%, and Chase%.
    Optional pitch_types_filter applies before aggregation.
    """
    if df_segment is None or df_segment.empty:
        return pd.DataFrame()

    # Restrict to NEB
    base = df_segment[df_segment.get('PitcherTeam','') == 'NEB'].copy()
    if base.empty:
        return pd.DataFrame()

    # Optional pitch-type filter (consistent with pitcher rankings)
    type_col = type_col_in_df(base)
    if pitch_types_filter:
        if type_col in base.columns:
            base = base[base[type_col].astype(str).isin(pitch_types_filter)].copy()
        if base.empty:
            return pd.DataFrame()

    # PA-level counts → H, BB, SO, outs → IP
    pa = _terminal_pa_table(base)
    box = _box_counts_from_PA(pa)
    outs = box["OUTS"]
    ip_float, _ = _compute_IP_from_outs(outs)

    # Plate discipline & strike%
    pdm = _plate_metrics_detailed(base)         # % values 0..100
    # HardHit / Barrel %
    hhbm = _hardhit_barrel_metrics(base)        # % values 0..100

    H, BB, SO = box["H"], box["BB"], box["SO"]

    # Rates
    WHIP = (BB + H) / ip_float if ip_float > 0 else np.nan
    H9   = (H * 9.0) / ip_float if ip_float > 0 else np.nan

    PA_n = len(pa)
    BBpct = (BB / max(PA_n, 1)) * 100.0
    SOpct = (SO / max(PA_n, 1)) * 100.0

    row = {
        "Team":     "NEB",
        "WHIP":     round(WHIP, 3) if pd.notna(WHIP) else np.nan,
        "H9":       round(H9, 2)   if pd.notna(H9)   else np.nan,
        "BB%":      round(BBpct, 1),
        "SO%":      round(SOpct, 1),
        "Strike%":  round(pdm.get("StrikePct", np.nan), 1) if pdm else np.nan,
        "Zone%":    round(pdm.get("ZonePct",   np.nan), 1) if pdm else np.nan,
        "Zwhiff%":  round(pdm.get("ZwhiffPct", np.nan), 1) if pdm else np.nan,
        "HH%":      round(hhbm.get("HardHitPct", np.nan), 1) if hhbm else np.nan,
        "Barrel%":  round(hhbm.get("BarrelPct",  np.nan), 1) if hhbm else np.nan,
        "Whiff%":   round(pdm.get("WhiffPct",  np.nan), 1) if pdm else np.nan,
        "Chase%":   round(pdm.get("ChasePct",  np.nan), 1) if pdm else np.nan,
    }

    # Column order to match the UI needs (so formatting/coloring aligns with rankings)
    cols = ["Team","WHIP","H9","BB%","SO%","Strike%","Zone%","Zwhiff%","HH%","Barrel%","Whiff%","Chase%"]
    return pd.DataFrame([row])[cols]

# [Continuing in next message with NEW FEATURES...]# ═══════════════════════════════════════════════════════════════════════════════
# NEW ADVANCED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 1. COUNT LEVERAGING ANALYSIS ─────────────────────────────────────────────
def create_count_leverage_heatmap(df: pd.DataFrame, metric: str = "Strike%"):
    """
    Creates a heatmap showing pitcher effectiveness by count (balls-strikes).
    
    Args:
        df: DataFrame with pitch-level data
        metric: One of "Strike%", "Whiff%", "Chase%", "InPlay%", "HardHit%"
    
    Returns:
        (fig, summary_df): matplotlib figure and DataFrame with top/bottom counts
    """
    if df is None or df.empty:
        return None, pd.DataFrame()
    
    # Find required columns
    balls_col = pick_col(df, "Balls", "Ball Count", "BallCount", "balls")
    strikes_col = pick_col(df, "Strikes", "Strike Count", "StrikeCount", "strikes")
    call_col = pick_col(df, "PitchCall", "Pitch Call", "Call", "PitchResult")
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")
    ev_col = pick_col(df, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV", "LaunchSpeed")
    
    if not balls_col or not strikes_col:
        return None, pd.DataFrame()
    
    # Convert to numeric
    balls = pd.to_numeric(df[balls_col], errors="coerce")
    strikes = pd.to_numeric(df[strikes_col], errors="coerce")
    
    # Valid counts only
    valid = balls.between(0, 3) & strikes.between(0, 2)
    work = df[valid].copy()
    work["Balls"] = balls[valid].astype(int)
    work["Strikes"] = strikes[valid].astype(int)
    
    if work.empty:
        return None, pd.DataFrame()
    
    # Calculate metric for each count
    matrix = np.full((4, 3), np.nan)  # 4 balls (0-3) x 3 strikes (0-2)
    count_labels = np.empty((4, 3), dtype=object)
    
    for b in range(4):
        for s in range(3):
            subset = work[(work["Balls"] == b) & (work["Strikes"] == s)]
            n = len(subset)
            count_labels[b, s] = f"n={n}"
            
            if n == 0:
                continue
            
            if metric == "Strike%":
                if call_col:
                    is_strike = subset[call_col].isin(['StrikeCalled','StrikeSwinging',
                                                       'FoulBallNotFieldable','FoulBallFieldable','InPlay'])
                    matrix[b, s] = is_strike.mean() * 100
            
            elif metric == "Whiff%":
                if call_col:
                    is_swing = subset[call_col].isin(['StrikeSwinging','FoulBallNotFieldable',
                                                      'FoulBallFieldable','InPlay'])
                    is_whiff = subset[call_col].eq('StrikeSwinging')
                    swings = is_swing.sum()
                    matrix[b, s] = (is_whiff.sum() / swings * 100) if swings > 0 else np.nan
            
            elif metric == "Chase%":
                if call_col and x_col and y_col:
                    is_swing = subset[call_col].isin(['StrikeSwinging','FoulBallNotFieldable',
                                                      'FoulBallFieldable','InPlay'])
                    xs = pd.to_numeric(subset[x_col], errors="coerce")
                    ys = pd.to_numeric(subset[y_col], errors="coerce")
                    l, b_zone, w, h = get_zone_bounds()
                    in_zone = xs.between(l, l+w) & ys.between(b_zone, b_zone+h)
                    out_zone = ~in_zone & xs.notna() & ys.notna()
                    chases = (is_swing & out_zone).sum()
                    out_zone_pitches = out_zone.sum()
                    matrix[b, s] = (chases / out_zone_pitches * 100) if out_zone_pitches > 0 else np.nan
            
            elif metric == "InPlay%":
                if call_col:
                    is_inplay = subset[call_col].eq('InPlay')
                    matrix[b, s] = is_inplay.mean() * 100
            
            elif metric == "HardHit%":
                if call_col and ev_col:
                    is_inplay = subset[call_col].eq('InPlay')
                    ev = pd.to_numeric(subset[ev_col], errors="coerce")
                    bip = is_inplay & ev.notna()
                    hard = (ev >= 95.0) & bip
                    bip_count = bip.sum()
                    matrix[b, s] = (hard.sum() / bip_count * 100) if bip_count > 0 else np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use RdYlGn colormap (green=good, red=poor)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(['0', '1', '2'])
    ax.set_yticklabels(['0', '1', '2', '3'])
    
    ax.set_xlabel('Strikes', fontsize=12, fontweight='600')
    ax.set_ylabel('Balls', fontsize=12, fontweight='600')
    ax.set_title(f'{metric} by Count', fontsize=14, fontweight='bold', pad=15)
    
    # Add text annotations
    for b in range(4):
        for s in range(3):
            val = matrix[b, s]
            label = count_labels[b, s]
            
            if pd.notna(val):
                text = f"{val:.1f}%\n{label}"
                color = 'white' if val < 50 else 'black'
            else:
                text = label
                color = 'gray'
            
            ax.text(s, b, text, ha="center", va="center", 
                   color=color, fontsize=10, fontweight='500')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, rotation=270, labelpad=20, fontsize=11, fontweight='600')
    
    plt.tight_layout()
    
    # Create summary DataFrame with best/worst counts
    summary_rows = []
    for b in range(4):
        for s in range(3):
            if pd.notna(matrix[b, s]):
                summary_rows.append({
                    'Count': f"{b}-{s}",
                    metric: round(matrix[b, s], 1),
                    'Pitches': int(count_labels[b, s].replace('n=', ''))
                })
    
    summary_df = pd.DataFrame(summary_rows).sort_values(metric, ascending=False)
    
    return fig, summary_df


def create_count_situation_comparison(df: pd.DataFrame):
    """
    Compare pitcher effectiveness across count situations:
    - First Pitch (0-0)
    - Ahead in Count (more strikes than balls)
    - Even Count (equal balls and strikes)
    - Behind in Count (more balls than strikes)
    - Two Strikes (0-2, 1-2, 2-2, 3-2)
    - Three Balls (3-0, 3-1, 3-2)
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    balls_col = pick_col(df, "Balls", "Ball Count", "BallCount", "balls")
    strikes_col = pick_col(df, "Strikes", "Strike Count", "StrikeCount", "strikes")
    call_col = pick_col(df, "PitchCall", "Pitch Call", "Call", "PitchResult")
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")
    ev_col = pick_col(df, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV", "LaunchSpeed")
    
    if not balls_col or not strikes_col:
        return pd.DataFrame()
    
    balls = pd.to_numeric(df[balls_col], errors="coerce")
    strikes = pd.to_numeric(df[strikes_col], errors="coerce")
    
    work = df.copy()
    work["Balls"] = balls
    work["Strikes"] = strikes
    
    # Define situations
    situations = {
        'First Pitch (0-0)': (balls == 0) & (strikes == 0),
        'Ahead in Count': strikes > balls,
        'Even Count': (strikes == balls) & ((balls + strikes) > 0),
        'Behind in Count': balls > strikes,
        'Two Strikes': strikes == 2,
        'Three Balls': balls == 3,
    }
    
    rows = []
    for sit_name, mask in situations.items():
        subset = work[mask & balls.notna() & strikes.notna()]
        n = len(subset)
        
        if n == 0:
            continue
        
        row = {'Situation': sit_name, 'Pitches': n}
        
        # Strike%
        if call_col:
            is_strike = subset[call_col].isin(['StrikeCalled','StrikeSwinging',
                                               'FoulBallNotFieldable','FoulBallFieldable','InPlay'])
            row['Strike%'] = round(is_strike.mean() * 100, 1)
        
        # Whiff%
        if call_col:
            is_swing = subset[call_col].isin(['StrikeSwinging','FoulBallNotFieldable',
                                              'FoulBallFieldable','InPlay'])
            is_whiff = subset[call_col].eq('StrikeSwinging')
            swings = is_swing.sum()
            row['Whiff%'] = round((is_whiff.sum() / swings * 100), 1) if swings > 0 else np.nan
        
        # Zone%
        if x_col and y_col:
            xs = pd.to_numeric(subset[x_col], errors="coerce")
            ys = pd.to_numeric(subset[y_col], errors="coerce")
            l, b_zone, w, h = get_zone_bounds()
            in_zone = xs.between(l, l+w) & ys.between(b_zone, b_zone+h)
            row['Zone%'] = round(in_zone.mean() * 100, 1) if in_zone.notna().any() else np.nan
        
        # Chase%
        if call_col and x_col and y_col:
            is_swing = subset[call_col].isin(['StrikeSwinging','FoulBallNotFieldable',
                                              'FoulBallFieldable','InPlay'])
            xs = pd.to_numeric(subset[x_col], errors="coerce")
            ys = pd.to_numeric(subset[y_col], errors="coerce")
            l, b_zone, w, h = get_zone_bounds()
            in_zone = xs.between(l, l+w) & ys.between(b_zone, b_zone+h)
            out_zone = ~in_zone & xs.notna() & ys.notna()
            chases = (is_swing & out_zone).sum()
            out_zone_pitches = out_zone.sum()
            row['Chase%'] = round((chases / out_zone_pitches * 100), 1) if out_zone_pitches > 0 else np.nan
        
        # HardHit%
        if call_col and ev_col:
            is_inplay = subset[call_col].eq('InPlay')
            ev = pd.to_numeric(subset[ev_col], errors="coerce")
            bip = is_inplay & ev.notna()
            hard = (ev >= 95.0) & bip
            bip_count = bip.sum()
            row['HardHit%'] = round((hard.sum() / bip_count * 100), 1) if bip_count > 0 else np.nan
        
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values('Pitches', ascending=False)


# ─── 2. SPRAY CHARTS FOR HITS ALLOWED ────────────────────────────────────────
def create_spray_chart(df: pd.DataFrame, pitcher_name: str, season_label: str = "Season"):
    """
    Creates a spray chart showing batted ball locations using polar coordinates.
    
    Args:
        df: DataFrame with pitch-level data
        pitcher_name: Name of pitcher to filter
        season_label: Label for the chart title
    
    Returns:
        (fig, summary_df): matplotlib figure and summary DataFrame
    """
    df_p = subset_by_pitcher_if_possible(df, pitcher_name)
    
    # Find required columns
    bearing_col = pick_col(df_p, "Bearing", "HitBearing", "Hit Bearing", "Direction", "Angle")
    distance_col = pick_col(df_p, "Distance", "HitDistance", "Hit Distance", "Dist")
    result_col = pick_col(df_p, "PlayResult", "Result", "Event", "PAResult")
    type_col = pick_col(df_p, "HitType", "Hit Type", "BattedBallType", "BBType")
    ev_col = pick_col(df_p, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV")
    call_col = pick_col(df_p, "PitchCall", "Pitch Call", "Call")
    
    if not bearing_col or not distance_col:
        # Fallback to simplified spray chart
        return create_simplified_spray_chart(df_p, pitcher_name, season_label)
    
    # Filter to balls in play only
    if call_col:
        bip = df_p[df_p[call_col].eq('InPlay')].copy()
    else:
        bip = df_p.copy()
    
    if bip.empty:
        return None, pd.DataFrame()
    
    # Convert bearing and distance to numeric
    bearing = pd.to_numeric(bip[bearing_col], errors="coerce")
    distance = pd.to_numeric(bip[distance_col], errors="coerce")
    
    valid = bearing.notna() & distance.notna()
    bip = bip[valid].copy()
    bearing = bearing[valid]
    distance = distance[valid]
    
    if bip.empty:
        return None, pd.DataFrame()
    
    # Convert polar to cartesian (bearing is angle from center field)
    # Bearing: 0° = center field, positive = pull side
    theta_rad = np.radians(bearing)
    x = distance * np.sin(theta_rad)
    y = distance * np.cos(theta_rad)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determine colors and sizes by outcome
    colors_list = []
    sizes_list = []
    labels_list = []
    
    for idx in bip.index:
        # Determine outcome
        result = str(bip.loc[idx, result_col]) if result_col else ""
        hit_type = str(bip.loc[idx, type_col]) if type_col else ""
        
        result_lower = result.lower()
        hit_type_lower = hit_type.lower()
        
        # Color and size by outcome
        if "home run" in result_lower or result_lower == "hr":
            color = 'red'
            size = 200
            label = "Home Run"
        elif any(x in result_lower for x in ["single", "double", "triple"]) or "hit" in result_lower:
            color = 'gold'
            size = 120
            label = "Hit"
        elif "line" in hit_type_lower:
            color = 'orange'
            size = 100
            label = "Line Drive"
        elif "fly" in hit_type_lower:
            color = 'skyblue'
            size = 80
            label = "Fly Ball"
        elif "ground" in hit_type_lower:
            color = 'green'
            size = 80
            label = "Ground Ball"
        elif "popup" in hit_type_lower or "pop" in hit_type_lower:
            color = 'purple'
            size = 60
            label = "Pop Up"
        elif "error" in result_lower:
            color = 'orange'
            size = 80
            label = "Error"
        else:
            color = 'gray'
            size = 60
            label = "Out"
        
        colors_list.append(color)
        sizes_list.append(size)
        labels_list.append(label)
    
    # Get exit velocity for opacity
    if ev_col:
        ev = pd.to_numeric(bip[ev_col], errors="coerce")
        # Normalize EV to opacity (70-120 mph range)
        alpha_vals = ((ev - 70) / 50).clip(0.3, 1.0).fillna(0.5).values
    else:
        alpha_vals = [0.7] * len(bip)
    
    # Plot each point
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.scatter(xi, yi, c=colors_list[i], s=sizes_list[i], 
                  alpha=alpha_vals[i], edgecolors='white', linewidths=1)
    
    # Draw field outline
    # Home plate at origin, outfield fence at ~400ft
    fence_radius = 400
    theta_fence = np.linspace(-45, 45, 100)  # -45° to +45° (foul lines)
    theta_rad_fence = np.radians(theta_fence)
    fence_x = fence_radius * np.sin(theta_rad_fence)
    fence_y = fence_radius * np.cos(theta_rad_fence)
    ax.plot(fence_x, fence_y, 'k-', linewidth=2, label='Outfield Fence')
    
    # Draw foul lines
    ax.plot([0, -fence_radius*np.sin(np.radians(45))], 
           [0, fence_radius*np.cos(np.radians(45))], 
           'k--', linewidth=1.5, alpha=0.6)
    ax.plot([0, fence_radius*np.sin(np.radians(45))], 
           [0, fence_radius*np.cos(np.radians(45))], 
           'k--', linewidth=1.5, alpha=0.6)
    
    # Draw infield dirt (~130ft radius)
    infield = Circle((0, 0), 130, fill=False, edgecolor='brown', 
                     linewidth=2, linestyle=':', alpha=0.5)
    ax.add_patch(infield)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Home Run'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
               markersize=10, label='Hit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markersize=8, label='Line Drive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', 
               markersize=8, label='Fly Ball'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=8, label='Ground Ball'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, label='Out'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    ax.set_xlim(-450, 450)
    ax.set_ylim(-50, 450)
    ax.set_aspect('equal')
    ax.set_xlabel('Horizontal Distance (ft)', fontsize=11, fontweight='500')
    ax.set_ylabel('Distance from Home (ft)', fontsize=11, fontweight='500')
    ax.set_title(f'Spray Chart: {canonicalize_person_name(pitcher_name)}\n{season_label}', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2, linestyle=':')
    
    plt.tight_layout()
    
    # Create summary DataFrame
    summary_data = []
    for label in set(labels_list):
        mask = np.array(labels_list) == label
        count = mask.sum()
        avg_dist = distance[bip.index[mask]].mean() if count > 0 else np.nan
        
        if ev_col:
            ev_subset = pd.to_numeric(bip.loc[bip.index[mask], ev_col], errors="coerce")
            avg_ev = ev_subset.mean() if len(ev_subset.dropna()) > 0 else np.nan
            max_ev = ev_subset.max() if len(ev_subset.dropna()) > 0 else np.nan
        else:
            avg_ev = max_ev = np.nan
        
        summary_data.append({
            'Type': label,
            'Count': int(count),
            'Avg Distance': round(avg_dist, 1) if pd.notna(avg_dist) else np.nan,
            'Avg Exit Velo': round(avg_ev, 1) if pd.notna(avg_ev) else np.nan,
            'Max Exit Velo': round(max_ev, 1) if pd.notna(max_ev) else np.nan,
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Count', ascending=False)
    
    return fig, summary_df


def create_simplified_spray_chart(df: pd.DataFrame, pitcher_name: str, season_label: str = "Season"):
    """
    Fallback spray chart when Bearing/Distance not available.
    Shows grouped bar chart by spray direction.
    """
    df_p = subset_by_pitcher_if_possible(df, pitcher_name)
    
    result_col = pick_col(df_p, "PlayResult", "Result", "Event", "PAResult")
    call_col = pick_col(df_p, "PitchCall", "Pitch Call", "Call")
    
    if call_col:
        bip = df_p[df_p[call_col].eq('InPlay')].copy()
    else:
        bip = df_p.copy()
    
    if bip.empty or not result_col:
        return None, pd.DataFrame()
    
    # Categorize by result
    result = bip[result_col].astype(str).str.lower()
    
    categories = {
        'Single': result.str.contains('single'),
        'Double': result.str.contains('double'),
        'Triple': result.str.contains('triple'),
        'Home Run': result.str.contains('home run') | result.eq('hr'),
        'Out': ~(result.str.contains('single|double|triple|home run|hr', regex=True))
    }
    
    # Mock spray direction (without actual coordinates)
    # Just show outcome distribution
    data = []
    for cat, mask in categories.items():
        count = mask.sum()
        if count > 0:
            data.append({'Outcome': cat, 'Count': int(count)})
    
    if not data:
        return None, pd.DataFrame()
    
    chart_df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_map = {
        'Home Run': 'red', 'Triple': 'purple', 'Double': 'orange',
        'Single': 'gold', 'Out': 'gray'
    }
    colors = [colors_map.get(x, 'blue') for x in chart_df['Outcome']]
    
    ax.bar(chart_df['Outcome'], chart_df['Count'], color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Count', fontsize=11, fontweight='500')
    ax.set_title(f'Batted Ball Outcomes: {canonicalize_person_name(pitcher_name)}\n{season_label}',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    return fig


# ─── 3. PITCH SEQUENCING ANALYSIS ────────────────────────────────────────────
def analyze_pitch_sequences(df: pd.DataFrame, pitcher_name: str):
    """
    Analyzes pitch-to-pitch sequences and their effectiveness.
    
    Returns:
        (transition_matrix, effectiveness_df, sankey_fig)
    """
    df_p = subset_by_pitcher_if_possible(df, pitcher_name)
    df_p = add_inning_and_ab(df_p)
    
    type_col = type_col_in_df(df_p)
    call_col = pick_col(df_p, "PitchCall", "Pitch Call", "Call", "PitchResult")
    
    if not type_col or type_col not in df_p.columns:
        return None, None, None
    
    # Create sequences (current pitch → next pitch within same AB)
    df_p = df_p.sort_values(['AB #', 'Pitch # in AB']).reset_index(drop=True)
    df_p['_current_pitch'] = df_p[type_col].astype(str)
    df_p['_next_pitch'] = df_p.groupby('AB #')[type_col].shift(-1)
    
    # Remove sequences that cross AB boundaries
    sequences = df_p[df_p['_next_pitch'].notna()].copy()
    
    if sequences.empty:
        return None, None, None
    
    # Transition matrix: % of time pitch B follows pitch A
    transition_counts = sequences.groupby(['_current_pitch', '_next_pitch']).size()
    transition_totals = sequences.groupby('_current_pitch').size()
    transition_pct = (transition_counts / transition_totals * 100).reset_index(name='Percentage')
    
    transition_matrix = transition_pct.pivot(index='_current_pitch', 
                                            columns='_next_pitch', 
                                            values='Percentage').fillna(0)
    
    # Effectiveness analysis
    effectiveness_data = []
    
    for (curr, nxt), grp in sequences.groupby(['_current_pitch', '_next_pitch']):
        n = len(grp)
        if n < 3:  # Minimum sample size
            continue
        
        # Calculate metrics for the NEXT pitch
        if call_col:
            next_pitches = df_p[df_p.index.isin(grp.index + 1)]
            
            # Strike%
            is_strike = next_pitches[call_col].isin(['StrikeCalled','StrikeSwinging',
                                                     'FoulBallNotFieldable','FoulBallFieldable','InPlay'])
            strike_pct = is_strike.mean() * 100
            
            # Whiff%
            is_swing = next_pitches[call_col].isin(['StrikeSwinging','FoulBallNotFieldable',
                                                    'FoulBallFieldable','InPlay'])
            is_whiff = next_pitches[call_col].eq('StrikeSwinging')
            swings = is_swing.sum()
            whiff_pct = (is_whiff.sum() / swings * 100) if swings > 0 else 0
            
            # InPlay%
            is_inplay = next_pitches[call_col].eq('InPlay')
            inplay_pct = is_inplay.mean() * 100
            
            # Effectiveness score (higher strike% and whiff%, lower inplay%)
            effectiveness = (strike_pct * 0.4) + (whiff_pct * 0.4) - (inplay_pct * 0.2)
            
            effectiveness_data.append({
                'Sequence': f"{curr} → {nxt}",
                'Count': n,
                'Strike%': round(strike_pct, 1),
                'Whiff%': round(whiff_pct, 1),
                'InPlay%': round(inplay_pct, 1),
                'Effectiveness Score': round(effectiveness, 1),
                '_current': curr,
                '_next': nxt
            })
    
    effectiveness_df = pd.DataFrame(effectiveness_data).sort_values('Count', ascending=False)
    
    # Create Sankey diagram for top sequences
    top_sequences = effectiveness_df.head(15)
    
    if len(top_sequences) == 0:
        return transition_matrix, effectiveness_df, None
    
    # Build Sankey
    pitch_types = list(set(top_sequences['_current'].tolist() + top_sequences['_next'].tolist()))
    
    # Create node labels (need to double them for source and target)
    source_labels = [f"{p} (from)" for p in pitch_types]
    target_labels = [f"{p} (to)" for p in pitch_types]
    all_labels = source_labels + target_labels
    
    # Map pitch types to indices
    source_map = {p: i for i, p in enumerate(pitch_types)}
    target_map = {p: i + len(pitch_types) for i, p in enumerate(pitch_types)}
    
    # Build links
    sources = []
    targets = []
    values = []
    colors = []
    
    for _, row in top_sequences.iterrows():
        sources.append(source_map[row['_current']])
        targets.append(target_map[row['_next']])
        values.append(row['Count'])
        
        # Color by effectiveness
        eff = row['Effectiveness Score']
        if eff > 60:
            color = 'rgba(0, 200, 0, 0.4)'  # Green
        elif eff > 40:
            color = 'rgba(255, 255, 0, 0.4)'  # Yellow
        else:
            color = 'rgba(255, 0, 0, 0.4)'  # Red
        colors.append(color)
    
    # Node colors (use pitch type colors)
    node_colors = []
    for p in pitch_types:
        color = get_pitch_color(p)
        node_colors.append(color)
        node_colors.append(color)  # Duplicate for target nodes
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=2),
            label=[label.replace(" (from)", "").replace(" (to)", "") for label in all_labels],
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])
    
    fig.update_layout(
        title_text=f"Pitch Sequencing Flow: {canonicalize_person_name(pitcher_name)}",
        title_font=dict(size=16, color=DARK_GRAY, family="Arial Black"),
        font_size=11,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return transition_matrix, effectiveness_df, fig


def find_best_sequences(df: pd.DataFrame, pitcher_name: str, min_count: int = 5):
    """
    Returns the most effective pitch sequences.
    """
    _, effectiveness_df, _ = analyze_pitch_sequences(df, pitcher_name)
    
    if effectiveness_df is None or effectiveness_df.empty:
        return pd.DataFrame()
    
    best = effectiveness_df[effectiveness_df['Count'] >= min_count].copy()
    best = best.sort_values('Effectiveness Score', ascending=False)
    
    return best[['Sequence', 'Count', 'Strike%', 'Whiff%', 'InPlay%', 'Effectiveness Score']]


def analyze_sequence_by_count(df: pd.DataFrame, pitcher_name: str):
    """
    Show pitch usage % by count situation (First Pitch, Ahead, Behind, 2-Strike, Other).
    """
    df_p = subset_by_pitcher_if_possible(df, pitcher_name)
    df_p = add_inning_and_ab(df_p)
    
    type_col = type_col_in_df(df_p)
    balls_col = pick_col(df_p, "Balls", "Ball Count", "BallCount", "balls")
    strikes_col = pick_col(df_p, "Strikes", "Strike Count", "StrikeCount", "strikes")
    
    if not type_col or not balls_col or not strikes_col:
        return pd.DataFrame()
    
    balls = pd.to_numeric(df_p[balls_col], errors="coerce")
    strikes = pd.to_numeric(df_p[strikes_col], errors="coerce")
    
    df_p['Count_Situation'] = 'Other'
    df_p.loc[(balls == 0) & (strikes == 0), 'Count_Situation'] = 'First Pitch'
    df_p.loc[strikes > balls, 'Count_Situation'] = 'Ahead'
    df_p.loc[balls > strikes, 'Count_Situation'] = 'Behind'
    df_p.loc[strikes == 2, 'Count_Situation'] = '2-Strike'
    
    # Pitch usage by situation
    usage = df_p.groupby([type_col, 'Count_Situation']).size().unstack(fill_value=0)
    usage_pct = usage.div(usage.sum(axis=0), axis=1) * 100
    
    return usage_pct.round(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONTINUE WITH DATA LOADING AND MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

# [REST OF THE CODE CONTINUES IN PART 4...]# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_main_csv():
    if not os.path.exists(DATA_PATH_MAIN):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH_MAIN, low_memory=False)
    df = ensure_date_column(df)
    
    # Ensure PitcherDisplay column
    pitcher_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name","PitcherLastFirst")
    if pitcher_col:
        df["PitcherDisplay"] = df[pitcher_col].map(canonicalize_person_name)
    else:
        df["PitcherDisplay"] = "Unknown"
    
    return df

@st.cache_data
def load_scrimmage_csv():
    if not os.path.exists(DATA_PATH_SCRIM):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH_SCRIM, low_memory=False)
    df = ensure_date_column(df)
    
    # Ensure PitcherDisplay column
    pitcher_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name","PitcherLastFirst")
    if pitcher_col:
        df["PitcherDisplay"] = df[pitcher_col].map(canonicalize_person_name)
    else:
        df["PitcherDisplay"] = "Unknown"
    
    return df

df_main = load_main_csv()
df_scrim = load_scrimmage_csv()

if df_main.empty and df_scrim.empty:
    st.error("No data files found. Please ensure pitcher_columns.csv or Scrimmage(28).csv are present.")
    st.stop()

# Combine datasets
df_all = pd.concat([df_main, df_scrim], ignore_index=True) if not df_scrim.empty else df_main.copy()
df_all = ensure_date_column(df_all)

# Get NEB pitchers
neb_pitchers = sorted(df_all[df_all.get('PitcherTeam','') == 'NEB']['PitcherDisplay'].dropna().unique())

if len(neb_pitchers) == 0:
    st.error("No Nebraska pitchers found in data.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image(LOGO_PATH, width=150) if os.path.exists(LOGO_PATH) else None
    
    st.markdown("### 📊 Data Filters")
    
    # Segment selection
    segment_choice = st.selectbox(
        "Season/Segment",
        options=["All Data"] + list(SEGMENT_DEFS.keys()),
        index=0,
        key="segment_choice"
    )
    
    # Pitcher selection
    pitcher_choice = st.selectbox(
        "Select Pitcher",
        options=neb_pitchers,
        index=0,
        key="pitcher_choice"
    )
    
    # Batter handedness filter
    batter_side_options = ["All", "RHH", "LHH"]
    batter_side_choice = st.selectbox(
        "Batter Handedness",
        options=batter_side_options,
        index=0
    )
    
    st.markdown("---")
    
    # Date filters
    st.markdown("### 📅 Date Filters")
    
    month_choices = st.multiselect(
        "Month(s)",
        options=[name for _, name in MONTH_CHOICES],
        default=[]
    )
    months_sel = [num for num, name in MONTH_CHOICES if name in month_choices]
    
    day_choices = st.multiselect(
        "Day(s) of Month",
        options=list(range(1, 32)),
        default=[]
    )
    
    last_n_games = st.number_input(
        "Last N Games (0 = all)",
        min_value=0,
        max_value=50,
        value=0,
        step=1
    )
    
    st.markdown("---")
    st.caption("Nebraska Baseball Analytics Platform v2.0")

# Apply segment filter
if segment_choice == "All Data":
    df_pitcher_all = df_all.copy()
else:
    df_pitcher_all = filter_by_segment(df_all, segment_choice)

# Apply pitcher filter
df_pitcher_all = subset_by_pitcher_if_possible(df_pitcher_all, pitcher_choice)

# Apply batter handedness filter
if batter_side_choice != "All":
    side_col = pick_col(df_pitcher_all, "BatterSide","Batter Side","Bats","Stand","BatSide")
    if side_col:
        side_norm = df_pitcher_all[side_col].astype(str).str.strip().str[0].str.upper()
        target = "R" if batter_side_choice == "RHH" else "L"
        df_pitcher_all = df_pitcher_all[side_norm == target].copy()

# Apply date filters
df_pitcher_all, season_label_display = apply_month_day_lastN(
    df_pitcher_all, months_sel, day_choices, last_n_games
)

if df_pitcher_all.empty:
    st.warning("No data matches the current filter selection.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs(["📈 Standard", "👤 Profiles", "🏆 Rankings", "🍂 Fall Summary"])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1: STANDARD
# ───────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    section_header("Season Overview")
    
    # Summary metrics
    summary_table = make_outing_overall_summary(df_pitcher_all)
    st.dataframe(themed_table(summary_table), use_container_width=True)
    
    professional_divider()
    
    # Movement profile
    section_header("Movement Profile & Metrics")
    
    logo_img = load_logo_img()
    fig_movement, summary_movement = combined_pitcher_report(
        df_pitcher_all, pitcher_choice, logo_img, season_label=season_label_display
    )
    
    if fig_movement:
        show_and_close(fig_movement, use_container_width=True)
    else:
        st.info("Movement profile not available for current selection.")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2: PROFILES (WITH NEW FEATURES)
# ───────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    section_header(f"Pitcher Profile: {canonicalize_person_name(pitcher_choice)}")
    st.caption(f"**{season_label_display}** • Batter: {batter_side_choice}")
    
    # Performance summary
    outcome_table = make_pitcher_outcome_summary_table(df_pitcher_all)
    st.dataframe(themed_table(outcome_table), use_container_width=True)
    
    professional_divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW FEATURE 1: COUNT LEVERAGING ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("Count Leveraging Analysis")
    
    metric_choice = st.selectbox(
        "Select Metric",
        options=["Strike%", "Whiff%", "Chase%", "InPlay%", "HardHit%"],
        index=0,
        key="count_metric"
    )
    
    fig_count, summary_count = create_count_leverage_heatmap(df_pitcher_all, metric_choice)
    
    if fig_count:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            show_and_close(fig_count, use_container_width=True)
        
        with col2:
            if not summary_count.empty:
                st.markdown("#### Best Counts")
                top_counts = summary_count.head(5)
                st.dataframe(themed_table(top_counts), use_container_width=True, hide_index=True)
                
                st.markdown("#### Worst Counts")
                bottom_counts = summary_count.tail(5)
                st.dataframe(themed_table(bottom_counts), use_container_width=True, hide_index=True)
        
        # Count situation comparison
        st.markdown("#### Performance by Count Situation")
        situation_df = create_count_situation_comparison(df_pitcher_all)
        if not situation_df.empty:
            st.dataframe(themed_table(situation_df), use_container_width=True, hide_index=True)
    else:
        info_message("Count leveraging data not available. Requires ball/strike count information.")
    
    professional_divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW FEATURE 2: SPRAY CHART
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("Spray Chart: Hits Allowed")
    
    fig_spray, summary_spray = create_spray_chart(df_pitcher_all, pitcher_choice, season_label_display)
    
    if fig_spray:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            show_and_close(fig_spray, use_container_width=True)
        
        with col2:
            if summary_spray is not None and not summary_spray.empty:
                st.markdown("#### Batted Ball Summary")
                st.dataframe(themed_table(summary_spray), use_container_width=True, hide_index=True)
    else:
        info_message("Spray chart not available. Requires batted ball location data (Bearing/Distance).")
    
    professional_divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW FEATURE 3: PITCH SEQUENCING
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("Pitch Sequencing Analysis")
    
    trans_matrix, effectiveness, sankey_fig = analyze_pitch_sequences(df_pitcher_all, pitcher_choice)
    
    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)
        
        st.markdown("#### Most Effective Sequences")
        best_sequences = find_best_sequences(df_pitcher_all, pitcher_choice, min_count=5)
        
        if not best_sequences.empty:
            # Style the effectiveness scores
            def style_effectiveness(val):
                try:
                    v = float(val)
                    if v > 60:
                        return 'background-color: rgba(0, 200, 0, 0.3)'
                    elif v < 40:
                        return 'background-color: rgba(255, 0, 0, 0.3)'
                except:
                    pass
                return ''
            
            styled = best_sequences.style.applymap(
                style_effectiveness, 
                subset=['Effectiveness Score']
            ).hide(axis="index").format({
                'Strike%': '{:.1f}',
                'Whiff%': '{:.1f}',
                'InPlay%': '{:.1f}',
                'Effectiveness Score': '{:.1f}'
            }, na_rep="—")
            
            st.dataframe(styled, use_container_width=True)
        else:
            info_message("Insufficient data for sequence effectiveness analysis (minimum 5 occurrences required).")
        
        # Sequencing by count situation
        with st.expander("📊 Sequencing Strategy by Count"):
            seq_by_count = analyze_sequence_by_count(df_pitcher_all, pitcher_choice)
            if not seq_by_count.empty:
                st.dataframe(themed_table(seq_by_count), use_container_width=True)
            else:
                st.info("Count situation data not available.")
        
        # Full transition matrix
        with st.expander("🔢 Full Pitch Transition Matrix"):
            if trans_matrix is not None and not trans_matrix.empty:
                styled_matrix = trans_matrix.style.background_gradient(
                    cmap='YlGn', axis=None
                ).format("{:.1f}%")
                st.dataframe(styled_matrix, use_container_width=True)
            else:
                st.info("Transition matrix not available.")
    else:
        info_message("Pitch sequencing requires multiple pitches per at-bat. Not enough sequence data available.")
    
    professional_divider()
    
    # Pitch-by-pitch table (existing feature)
    section_header("Pitch-by-Pitch Breakdown")
    
    pbp_df = build_pitch_by_inning_pa_table(df_pitcher_all)
    
    if not pbp_df.empty and "Inning #" in pbp_df.columns:
        style_pbp_expanders()
        st.markdown('<div class="pbp-scope">', unsafe_allow_html=True)
        
        innings = sorted(pbp_df["Inning #"].dropna().unique())
        
        for inn in innings:
            inn_data = pbp_df[pbp_df["Inning #"] == inn]
            
            with st.expander(f"⚾ Inning {inn}", expanded=False):
                st.markdown('<div class="inning-block">', unsafe_allow_html=True)
                
                if "PA # in Inning" in inn_data.columns:
                    pas = sorted(inn_data["PA # in Inning"].dropna().unique())
                    
                    for pa_num in pas:
                        pa_data = inn_data[inn_data["PA # in Inning"] == pa_num]
                        
                        batter_name = pa_data["Batter"].iloc[0] if "Batter" in pa_data.columns else "Unknown"
                        pa_result = pa_data["PA Result"].iloc[0] if "PA Result" in pa_data.columns else "—"
                        
                        with st.expander(f"PA #{int(pa_num)}: {batter_name} ({pa_result})", expanded=False):
                            # Interactive strike zone
                            fig_pa = pa_interactive_strikezone(pa_data, title=f"PA #{int(pa_num)}: {batter_name}")
                            if fig_pa:
                                st.plotly_chart(fig_pa, use_container_width=True)
                            
                            # Pitch table (drop plate loc columns for display)
                            display_cols = [c for c in pa_data.columns 
                                          if c not in ["PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX",
                                                      "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ"]]
                            pa_display = pa_data[display_cols]
                            st.dataframe(themed_table(pa_display), use_container_width=True)
                else:
                    st.dataframe(themed_table(inn_data), use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No pitch-by-pitch data available for current selection.")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3: RANKINGS
# ───────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    section_header("Team Rankings")
    
    st.markdown("### Filter Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rank_segment = st.selectbox(
            "Segment",
            options=["All Data"] + list(SEGMENT_DEFS.keys()),
            index=0,
            key="rank_segment"
        )
    
    with col2:
        # Get available pitch types for this segment
        if rank_segment == "All Data":
            df_rank_base = df_all.copy()
        else:
            df_rank_base = filter_by_segment(df_all, rank_segment)
        
        type_col_rank = type_col_in_df(df_rank_base)
        available_types = sorted(df_rank_base[type_col_rank].dropna().unique()) if type_col_rank in df_rank_base.columns else []
        
        pitch_types_filter = st.multiselect(
            "Pitch Type(s)",
            options=available_types,
            default=[]
        )
    
    professional_divider()
    
    # Generate rankings
    rankings_df = make_pitcher_rankings(
        df_rank_base,
        pitch_types_filter if pitch_types_filter else None
    )
    
    if not rankings_df.empty:
        # Team averages
        st.markdown("### Team Averages")
        team_avg = make_team_averages(
            df_rank_base,
            pitch_types_filter if pitch_types_filter else None
        )
        st.dataframe(themed_table(team_avg), use_container_width=True, hide_index=True)
        
        professional_divider()
        
        # Individual rankings
        st.markdown("### Individual Pitcher Rankings")
        
        # Sort options
        sort_col = st.selectbox(
            "Sort by",
            options=["WHIP", "SO", "Strike%", "Whiff%", "HH%", "Barrel%", "IP"],
            index=0
        )
        
        # Sort
        if sort_col == "IP":
            rankings_display = rankings_df.sort_values("_IP_num", ascending=False)
        elif sort_col in ["WHIP", "HH%", "Barrel%"]:
            rankings_display = rankings_df.sort_values(sort_col, ascending=True, na_position="last")
        else:
            rankings_display = rankings_df.sort_values(sort_col, ascending=False, na_position="last")
        
        # Drop internal column
        rankings_display = rankings_display.drop(columns=["_IP_num"])
        
        st.dataframe(themed_table(rankings_display), use_container_width=True, hide_index=True)
        
        # Export option
        csv = rankings_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Rankings CSV",
            data=csv,
            file_name=f"nebraska_pitcher_rankings_{rank_segment.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No ranking data available for the selected filters.")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 4: FALL SUMMARY
# ───────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    section_header("Fall 2025/26 Summary")
    
    # Filter to fall scrimmages
    df_fall = filter_by_segment(df_all, "2025/26 Scrimmages")
    
    if df_fall.empty:
        st.info("No fall scrimmage data available.")
    else:
        st.markdown(f"### Data Summary: {len(df_fall)} pitches from fall scrimmages")
        
        # Per-pitcher summaries
        fall_pitchers = sorted(df_fall[df_fall.get('PitcherTeam','') == 'NEB']['PitcherDisplay'].dropna().unique())
        
        if fall_pitchers:
            selected_fall = st.selectbox(
                "Select Pitcher",
                options=fall_pitchers,
                key="fall_pitcher"
            )
            
            df_fall_p = subset_by_pitcher_if_possible(df_fall, selected_fall)
            
            professional_divider()
            
            # Movement profile
            logo_img = load_logo_img()
            fig_fall, summary_fall = combined_pitcher_report(
                df_fall_p, selected_fall, logo_img, season_label="Fall 2025/26"
            )
            
            if fig_fall:
                show_and_close(fig_fall, use_container_width=True)
            
            professional_divider()
            
            # Outing summary
            st.markdown("### Outing Summary")
            fall_summary = make_outing_overall_summary(df_fall_p)
            st.dataframe(themed_table(fall_summary), use_container_width=True)
        else:
            st.info("No Nebraska pitchers found in fall data.")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; padding: 20px; font-size: 14px;'>
        <strong>Nebraska Baseball Pitcher Analytics Platform</strong><br>
        Built with Streamlit • Data updated through {date.today().strftime('%B %d, %Y')}<br>
        <span style='color: {HUSKER_RED}; font-weight: 600;'>Go Big Red! 🌽</span>
    </div>
    """,
    unsafe_allow_html=True
)
