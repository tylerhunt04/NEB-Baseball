# pitcher_app.py — UPDATED VERSION WITH 2026 SEASON DATA SUPPORT
# Part 1 of 6: Imports, Configuration, CSS, and Basic Helpers

import math
from matplotlib.patches import Wedge, Polygon
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

DATA_PATH_SCRIM  = "Scrimmage(28).csv"
DATA_PATH_SEASON = "Season2026.csv"          # ← 2026 season master CSV
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
    .main { background-color: #FFFFFF; }
    .section-header {
        background: linear-gradient(135deg, #E60026 0%, #B8001F 100%);
        color: white; padding: 16px 24px; border-radius: 8px;
        margin: 24px 0 16px 0; font-weight: 600; font-size: 20px;
        box-shadow: 0 2px 8px rgba(230, 0, 38, 0.15);
    }
    .subsection-header {
        color: #2B2B2B; padding: 12px 0; border-bottom: 3px solid #E60026;
        margin: 20px 0 12px 0; font-weight: 600; font-size: 18px;
    }
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F8F8 100%);
        border-left: 4px solid #E60026; padding: 16px; border-radius: 8px;
        margin: 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .professional-divider {
        height: 2px; background: linear-gradient(to right, transparent, #E60026, transparent);
        margin: 32px 0;
    }
    .info-box {
        background-color: #F0F7FF; border-left: 4px solid #1E88E5;
        padding: 12px 16px; border-radius: 4px; margin: 12px 0; color: #1565C0;
    }
    .warning-box {
        background-color: #FFF8E1; border-left: 4px solid #FFA726;
        padding: 12px 16px; border-radius: 4px; margin: 12px 0; color: #EF6C00;
    }
    .stDownloadButton button {
        background-color: #E60026; color: white; border: none;
        border-radius: 6px; padding: 8px 20px; font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background-color: #F8F9FA; padding: 8px; border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white; border-radius: 6px; padding: 8px 16px;
        font-weight: 500; border: 1px solid #E0E0E0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E60026; color: white; border-color: #E60026;
    }
</style>
""", unsafe_allow_html=True)

custom_cmap = colors.LinearSegmentedColormap.from_list(
    'custom_cmap', [(0.0, 'white'), (0.2, 'deepskyblue'), (0.3, 'white'),
                    (0.7, 'red'), (1.0, 'red')], N=256
)

# ─── Cached loaders & display helpers ─────────────────────────────────────────
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

def hero_banner(title: str, *, subtitle: str | None = None, height_px: int = 280):
    from streamlit.components.v1 import html as _html
    b64 = load_banner_b64() or ""
    sub_html = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    _html(f"""
        <style>
        .hero-wrap {{
            position: relative; width: 100%; height: {height_px}px; border-radius: 12px;
            overflow: hidden; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .hero-bg {{
            position: absolute; inset: 0;
            background: linear-gradient(135deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.4) 100%),
                        url('data:image/jpeg;base64,{b64}');
            background-size: cover; background-position: center;
        }}
        .hero-text {{
            position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
            flex-direction: column; color: #fff; text-align: center; z-index: 10;
        }}
        .hero-title {{
            font-size: 42px; font-weight: 700; letter-spacing: 0.5px;
            text-shadow: 0 4px 12px rgba(0,0,0,0.6); margin: 0; text-transform: uppercase;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        }}
        .hero-sub {{
            font-size: 18px; font-weight: 400; opacity: .95; margin-top: 8px;
            text-shadow: 0 2px 8px rgba(0,0,0,0.5); letter-spacing: 0.3px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        }}
        </style>
        <div class="hero-wrap"><div class="hero-bg"></div>
          <div class="hero-text"><h1 class="hero-title">{title}</h1>{sub_html}</div>
        </div>
        """, height=height_px + 40)

hero_banner("Nebraska Baseball", subtitle="Pitcher Analytics Platform", height_px=280)

def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def professional_divider():
    st.markdown('<div class="professional-divider"></div>', unsafe_allow_html=True)

def info_message(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

# ─── Date helpers ─────────────────────────────────────────────────────────────
DATE_CANDIDATES = ["Date","date","GameDate","GAME_DATE","Game Date","Datetime","DateTime"]

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

MONTH_CHOICES = [(1,"January"), (2,"February"), (3,"March"), (4,"April"), (5,"May"), (6,"June"),
                 (7,"July"), (8,"August"), (9,"September"), (10,"October"), (11,"November"), (12,"December")]
MONTH_NAME_BY_NUM = {n: name for n, name in MONTH_CHOICES}

def filter_by_month_day(df, date_col="Date", months=None, days=None):
    if date_col not in df.columns or df.empty: return df
    s = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if months: mask &= s.dt.month.isin(months)
    if days:   mask &= s.dt.day.isin(days)
    return df[mask]

def build_pitcher_season_label(months_sel, days_sel, selected_df: pd.DataFrame) -> str:
    if (not months_sel) and (not days_sel): return "Season"
    if months_sel and not days_sel and len(months_sel) == 1:
        return MONTH_NAME_BY_NUM.get(months_sel[0], "Season")
    if selected_df is None or selected_df.empty or "Date" not in selected_df.columns: return "Season"
    return "Season"

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
        return df, f"Last {last_n_games} games"
    return df, base or "Season"

# ─── Column picker & segment helpers ──────────────────────────────────────────
def pick_col(df: pd.DataFrame, *cands) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c and c.lower() in lower_map: return lower_map[c.lower()]
    return None

SEGMENT_DEFS = {
    "2025/26 Scrimmages": {"start": "2025-08-01", "end": "2026-02-01", "types": ["scrimmage"]},
    "2026 Season":        {"start": "2026-02-01", "end": "2099-01-01", "types": ["game"]},
}

def filter_by_segment(df: pd.DataFrame, segment_name: str, scrimmage_pitchers: set = None) -> pd.DataFrame:
    """Filter by segment and optionally restrict to scrimmage pitchers only"""
    spec = SEGMENT_DEFS.get(segment_name)
    if spec is None or df.empty: return df
    out = df.copy()

    if "Date" in out.columns:
        d = pd.to_datetime(out["Date"], errors="coerce")
        start = pd.to_datetime(spec["start"]); end = pd.to_datetime(spec["end"])
        out = out[(d >= start) & (d < end)]

    if segment_name == "2025/26 Scrimmages" and scrimmage_pitchers is not None:
        if "PitcherDisplay" in out.columns:
            out = out[out["PitcherDisplay"].isin(scrimmage_pitchers)]

    return out

def type_col_in_df(df: pd.DataFrame) -> str:
    return (pick_col(df, "TaggedPitchType","Tagged Pitch Type","PitchType") or "TaggedPitchType")

# ─── Strike zone & pitch color helpers ────────────────────────────────────────
def get_zone_bounds(): return -0.83, 1.17, 1.66, 2.75
def get_view_bounds():
    l, b, w, h = get_zone_bounds(); mx, my = w*0.8, h*0.6
    return l-mx, l+w+mx, b-my, b+h+my

def get_pitch_color(ptype):
    if isinstance(ptype, str) and (ptype.lower().startswith("four-seam fastball") or ptype.lower()=="fastball"):
        return "#E60026"
    savant = {"sinker":"#FF9300","cutter":"#800080","changeup":"#008000","curveball":"#0033CC",
              "slider":"#CCCC00","splitter":"#00CCCC","knuckle curve":"#000000"}
    return savant.get(str(ptype).lower(), "#E60026")

def format_name(name):
    if isinstance(name, str) and ',' in name:
        last, first = [s.strip() for s in name.split(',', 1)]
        return f"{first} {last}"
    return str(name)

def canonicalize_person_name(raw) -> str:
    if pd.isna(raw): return ""
    s = str(raw).strip()
    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        s = f"{first} {last}"
    return re.sub(r"\s+", " ", s).strip()

def subset_by_pitcher_if_possible(df: pd.DataFrame, pitcher_display: str) -> pd.DataFrame:
    if "PitcherDisplay" in df.columns:
        sub = df[df["PitcherDisplay"] == pitcher_display]
        if not sub.empty: return sub.copy()
    pitch_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name") or "Pitcher"
    sub2 = df[df.get(pitch_col, "") == pitcher_display]
    return sub2.copy() if not sub2.empty else df.copy()

# ─── PBP helpers ──────────────────────────────────────────────────────────────
def find_batter_name_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "Batter","BatterName","Batter Name","BatterFullName")

def find_pitch_of_pa_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "PitchofPA","PitchOfPA","Pitch_of_PA","Pitch of PA","Pitch # in AB","PitchofPA","PitchNo")

def find_inning_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "Inning","inning","InningNumber","Inning #")

def _to_num(s): return pd.to_numeric(s, errors="coerce")

def _normalize_inning_series(series: pd.Series) -> pd.Series:
    txt = series.astype(str); num = txt.str.extract(r'(\d+)')[0]
    return pd.to_numeric(num, errors="coerce").astype(pd.Int64Dtype())

def add_inning_and_ab(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    inn_c = find_inning_col(out)
    po_c  = find_pitch_of_pa_col(out)

    out["Inning #"] = _normalize_inning_series(out[inn_c]) if inn_c else pd.Series([pd.NA]*len(out), dtype="Int64")

    # Season CSV uses PitchofPA column name
    pitch_of_pa_col = pick_col(out, "PitchofPA", "PitchOfPA", "Pitch_of_PA", "Pitch of PA", "Pitch # in AB")

    if pitch_of_pa_col is None:
        out["AB #"] = 1
        out["Pitch # in AB"] = np.arange(1, len(out) + 1)
    else:
        is_start = (_to_num(out[pitch_of_pa_col]) == 1)
        ab_id = is_start.cumsum()
        if (ab_id == 0).any(): ab_id = ab_id.replace(0, np.nan).ffill().fillna(1)
        out["AB #"] = ab_id.astype(int)
        out["Pitch # in AB"] = _to_num(out[pitch_of_pa_col]).astype(pd.Int64Dtype())

    batter_c = find_batter_name_col(out)
    if batter_c:
        names_by_ab = out.groupby("AB #")[batter_c].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        out["Batter_AB"] = out["AB #"].map(names_by_ab).apply(format_name)

    return out

def themed_table(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    integer_like = set(c for c in numeric_cols if pd.api.types.is_integer_dtype(df[c]))
    percent_cols = [c for c in numeric_cols if c.strip().endswith('%')]
    fmt_map = {}
    for c in numeric_cols:
        if c in integer_like: fmt_map[c] = "{:.0f}"
        elif c in percent_cols: fmt_map[c] = "{:.1f}"
        else: fmt_map[c] = "{:.1f}"
    styles = [
        {'selector': 'thead th', 'props': f'background: linear-gradient(135deg, {HUSKER_RED} 0%, #B8001F 100%); color: white; font-weight: 600; text-align: center; padding: 12px 8px;'},
        {'selector': 'td', 'props': 'padding: 10px 8px; border-bottom: 1px solid #E0E0E0;'},
        {'selector': 'tr:hover', 'props': 'background-color: #F8F9FA;'},
    ]
    return df.style.hide(axis="index").format(fmt_map, na_rep="—").set_table_styles(styles)

# Part 2 of 6: Helper Functions, Outcome Summaries, and Rankings

# ─── First present helper ─────────────────────────────────────────────────────
def _first_present(df: pd.DataFrame, cands: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    return None

def _first_present_strict(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns: return n
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower: return lower[n.lower()]
    return None

def _to_float(s): return pd.to_numeric(s, errors="coerce")

# ─── Terminal PA & box score helpers ──────────────────────────────────────────
def _is_terminal_row(row, col_result, col_korbb, col_call) -> bool:
    pr = str(row.get(col_result, "")) if col_result else ""
    kc = str(row.get(col_korbb, "")) if col_korbb else ""
    pc = str(row.get(col_call, ""))  if col_call  else ""
    return (
        (pr.strip() != "") or
        (kc.lower() in {"k","so","strikeout","strikeout swinging","strikeout looking","bb","walk"}) or
        (pc.lower() in {"hitbypitch","hit by pitch","hbp"})
    )

def _terminal_pa_table(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty: return df_in.iloc[0:0].copy()
    work = add_inning_and_ab(df_in.copy())
    col_result = _first_present_strict(work, ["PlayResult","Result","Event","PAResult","Outcome"])
    col_call   = _first_present_strict(work, ["PitchCall","Pitch Call","PitchResult","Call"])
    col_korbb  = _first_present_strict(work, ["KorBB","K_BB","KBB","K_or_BB","PA_KBB"])
    for c in [col_result, col_call, col_korbb]:
        if c and c in work.columns:
            work[c] = work[c].fillna("").astype(str)

    term_mask = work.apply(lambda r: _is_terminal_row(r, col_result, col_korbb, col_call), axis=1) \
                if any([col_result, col_korbb, col_call]) else pd.Series(False, index=work.index)

    def _pick_row_idx(g: pd.DataFrame) -> int:
        gm = term_mask.loc[g.index]
        if gm.any(): return gm[gm].index[-1]
        if "Pitch # in AB" in g.columns and g["Pitch # in AB"].notna().any():
            return g["Pitch # in AB"].astype("Int64").idxmax()
        return g.index[-1]

    ab_rows_idx = work.groupby("AB #", sort=True, dropna=False).apply(_pick_row_idx).values
    out = work.loc[ab_rows_idx].copy()
    out["_PlayResult"] = out[col_result] if col_result else ""
    out["_PitchCall"]  = out[col_call]   if col_call   else ""
    out["_KorBB"]      = out[col_korbb]  if col_korbb  else ""
    return out

def _compute_IP_from_outs(total_outs: int) -> tuple[float, str]:
    ip_float = total_outs / 3.0
    whole = total_outs // 3
    rem   = total_outs % 3
    ip_disp = f"{whole}.{rem}"
    return ip_float, ip_disp

def _box_counts_from_PA(pa_df: pd.DataFrame) -> dict:
    if pa_df is None or pa_df.empty:
        return dict(H=0, HR=0, BB=0, SO=0, HBP=0, OUTS=0)

    PR = pa_df["_PlayResult"].astype(str).str.lower()
    KC = pa_df["_KorBB"].astype(str).str.lower() if "_KorBB" in pa_df.columns else pd.Series([""]*len(pa_df), index=pa_df.index)
    PC = pa_df["_PitchCall"].astype(str).str.lower() if "_PitchCall" in pa_df.columns else pd.Series([""]*len(pa_df), index=pa_df.index)

    is_single = PR.str.contains(r"\bsingle\b")
    is_double = PR.str.contains(r"\bdouble\b")
    is_triple = PR.str.contains(r"\btriple\b")
    is_hr     = PR.str.contains(r"\bhome\s*run\b") | PR.eq("hr")

    is_bb  = (PR.str.contains(r"\bwalk\b|intentional\s*walk|ib[bB]\b")
              | KC.isin({"bb","walk","ibb","intentional walk"})
              | KC.str.contains(r"\bwalk\b"))
    is_so  = (PR.str.contains(r"strikeout")
              | KC.isin({"k","so","strikeout","strikeout swinging","strikeout looking"}))
    is_hbp = (PR.str.contains(r"hit\s*by\s*pitch")
              | PC.isin({"hitbypitch","hit by pitch","hbp"}))

    is_dp  = PR.str.contains("double play")
    is_tp  = PR.str.contains("triple play")
    is_outword = PR.str.contains("out") | PR.str.contains("groundout") | PR.str.contains("flyout") \
                 | PR.str.contains("lineout") | PR.str.contains("popout") | PR.str.contains("forceout")

    outs = (is_so.astype(int)*1 + is_dp.astype(int)*2 + is_tp.astype(int)*3)
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

# ─── Strike rate and outcome helpers ──────────────────────────────────────────
def strike_rate(df):
    call_col = _first_present(df, ["PitchCall","Pitch Call","Call"])
    if len(df) == 0 or not call_col: return np.nan
    strike_calls = ['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']
    return df[call_col].isin(strike_calls).mean() * 100

def _pct(x):   return f"{x*100:.1f}%" if pd.notna(x) else ""
def _rate3(x): return f"{x:.3f}" if pd.notna(x) else ""

def make_outing_overall_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame([{
            "IP": "0.0", "Pitches": 0, "Hits": 0, "Walks": 0, "SO": 0, "HBP": 0,
            "Strike%": np.nan, "Whiffs": 0, "Whiff%": np.nan,
            "Zone Whiffs": 0, "Zone Whiff%": np.nan,
            "HardHits": 0, "HardHit%": np.nan
        }])

    pitches = int(len(df_in))
    pa_tbl = _terminal_pa_table(df_in)
    box = _box_counts_from_PA(pa_tbl)
    ip_float, ip_disp = _compute_IP_from_outs(box["OUTS"])

    strike_pct = strike_rate(df_in)
    call_col = _first_present(df_in, ["PitchCall","Pitch Call","PitchResult","Call"])
    x_col    = _first_present(df_in, ["PlateLocSide","Plate Loc Side","PlateSide","px","PlateLocX"])
    y_col    = _first_present(df_in, ["PlateLocHeight","Plate Loc Height","PlateHeight","pz","PlateLocZ"])

    whiff_cnt = 0; whiff_pct = np.nan; zwhiff_cnt = 0; zwhiff_pct = np.nan

    if call_col:
        s_call = df_in[call_col].astype(str)
        is_swing = s_call.isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
        is_whiff = s_call.eq('StrikeSwinging')
        swings_total = int(is_swing.sum())
        whiff_cnt = int(is_whiff.sum())
        whiff_pct = (whiff_cnt / swings_total * 100.0) if swings_total > 0 else np.nan

        if x_col and y_col:
            xs = pd.to_numeric(df_in[x_col], errors="coerce")
            ys = pd.to_numeric(df_in[y_col], errors="coerce")
            l, b, w, h = get_zone_bounds()
            in_zone = xs.between(l, l+w) & ys.between(b, b+h)
            swings_in_zone = int((is_swing & in_zone).sum())
            zwhiff_cnt = int((is_whiff & in_zone).sum())
            zwhiff_pct = (zwhiff_cnt / swings_in_zone * 100.0) if swings_in_zone > 0 else np.nan

    ev_col = _first_present(df_in, ["ExitSpeed","Exit Velo","ExitVelocity","EV","LaunchSpeed"])
    hard_cnt = 0; hard_pct_bip = np.nan
    if ev_col and call_col:
        ev = pd.to_numeric(df_in[ev_col], errors="coerce")
        inplay = df_in[call_col].astype(str).eq("InPlay")
        bip = int(inplay.sum())
        hard_cnt = int(((ev >= 95.0) & inplay & ev.notna()).sum())
        hard_pct_bip = (hard_cnt / bip * 100.0) if bip > 0 else np.nan

    return pd.DataFrame([{
        "IP": ip_disp, "Pitches": pitches, "Hits": int(box["H"]), "Walks": int(box["BB"]),
        "SO": int(box["SO"]), "HBP": int(box["HBP"]),
        "Strike%": round(float(strike_pct), 1) if pd.notna(strike_pct) else np.nan,
        "Whiffs": whiff_cnt, "Whiff%": round(float(whiff_pct), 1) if pd.notna(whiff_pct) else np.nan,
        "Zone Whiffs": zwhiff_cnt, "Zone Whiff%": round(float(zwhiff_pct), 1) if pd.notna(zwhiff_pct) else np.nan,
        "HardHits": hard_cnt, "HardHit%": round(float(hard_pct_bip), 1) if pd.notna(hard_pct_bip) else np.nan,
    }])

def make_pitcher_outcome_summary_table(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame([{
            "Average exit velo": np.nan, "Max exit velo": np.nan, "Hits": 0, "Strikeouts": 0,
            "AVG":"", "OBP":"", "SLG":"", "OPS":"", "HardHit%":"", "K%":"", "Walk%":""
        }])

    col_exitv  = _first_present(df_in, ["ExitSpeed","Exit Velo","ExitVelocity","EV","LaunchSpeed"])
    col_result = _first_present(df_in, ["PlayResult","Result","Event","PAResult","Outcome"])
    col_call   = _first_present(df_in, ["PitchCall","Pitch Call","PitchResult","Call"])
    col_korbb  = _first_present(df_in, ["KorBB","K_BB","KBB","K_or_BB","PA_KBB"])

    work = add_inning_and_ab(df_in.copy())
    for c in [col_result, col_call, col_korbb]:
        if c and c in work.columns:
            work[c] = work[c].fillna("").astype(str)

    is_term = work.apply(lambda r: _is_terminal_row(r, col_result, col_korbb, col_call), axis=1)

    def _pick_row_idx(g: pd.DataFrame) -> int:
        gm = is_term.loc[g.index]
        if gm.any(): return gm[gm].index[-1]
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

    is_bb  = (pr_low.str.contains(r"\bwalk\b|intentional\s*walk", regex=True)
              | KC.str.lower().isin({"bb","walk","ibb"}))
    is_so  = (pr_low.str.contains(r"strikeout", case=False, regex=True)
              | KC.str.lower().isin({"k","so","strikeout"}))
    is_hbp = (pr_low.str.contains(r"hit\s*by\s*pitch", case=False, regex=True)
              | PC.str.lower().isin({"hitbypitch","hit by pitch","hbp"}))
    is_sf  = pr_low.str.contains(r"sac(rifice)?\s*fly|\bsf\b", regex=True)
    is_sh  = pr_low.str.contains(r"sac(rifice)?\s*(bunt|hit)|\bsh\b", regex=True)

    PA  = int(len(df_pa)); H = int(hits_mask.sum()); BB = int(is_bb.sum()); SO = int(is_so.sum())
    HBP = int(is_hbp.sum()); SF = int(is_sf.sum()); SH = int(is_sh.sum())
    AB  = max(PA - (BB + HBP + SF + SH), 0)

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

    return pd.DataFrame([{
        "Average exit velo": round(avg_ev, 1) if pd.notna(avg_ev) else np.nan,
        "Max exit velo":     round(max_ev, 1) if pd.notna(max_ev) else np.nan,
        "Hits": H, "Strikeouts": SO,
        "AVG": _rate3(AVG), "OBP": _rate3(OBP), "SLG": _rate3(SLG), "OPS": _rate3(OPS),
        "HardHit%": _pct(hard_hit_pct), "K%": _pct(K_rate), "Walk%": _pct(BB_rate),
    }])

# ─── Plate metrics for rankings ───────────────────────────────────────────────
def _plate_metrics_detailed(sub: pd.DataFrame) -> dict:
    call_col = _first_present(sub, ["PitchCall","Pitch Call","Call"])
    if not call_col: return {}

    s_call = sub[call_col]
    xs = _to_float(sub.get('PlateLocSide', pd.Series(dtype=float)))
    ys = _to_float(sub.get('PlateLocHeight', pd.Series(dtype=float)))

    isswing = s_call.isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
    iswhiff = s_call.eq('StrikeSwinging')

    z_left, z_bot, z_w, z_h = get_zone_bounds()
    isinzone = xs.between(z_left, z_left+z_w) & ys.between(z_bot, z_bot+z_h)

    swingsZ = int(isswing[isinzone].sum())

    zone_pct   = 100.0 * float(isinzone.mean()) if len(sub) else np.nan
    zwhiff_pct = 100.0 * (iswhiff[isinzone].sum() / swingsZ) if swingsZ > 0 else np.nan
    chase_pct  = 100.0 * float(isswing[~isinzone].mean()) if (~isinzone).any() else np.nan
    whiff_pct  = 100.0 * float(iswhiff.sum() / max(int(isswing.sum()),1)) if int(isswing.sum())>0 else np.nan
    strike_pct = strike_rate(sub)

    return dict(ZonePct=zone_pct, ZwhiffPct=zwhiff_pct, ChasePct=chase_pct, WhiffPct=whiff_pct, StrikePct=strike_pct)

def _hardhit_barrel_metrics(sub: pd.DataFrame) -> dict:
    ev_col = _first_present_strict(sub, ["ExitSpeed","Exit Velo","ExitVelocity","EV","LaunchSpeed"])
    la_col = _first_present_strict(sub, ["LaunchAngle","Launch_Angle","LA","Angle"])
    call   = _first_present_strict(sub, ["PitchCall","Pitch Call","Call"])
    if call is None:
        return dict(HardHitPct=np.nan, BarrelPct=np.nan)

    inplay = sub[sub[call].astype(str).eq("InPlay")]

    if ev_col:
        ev_bip = _to_float(inplay[ev_col]).dropna()
        hh = float((ev_bip >= 95).mean()) * 100 if len(ev_bip) else np.nan
    else:
        hh = np.nan

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

# Part 3 of 6: Rankings and Advanced Features

# ─── Rankings functions ───────────────────────────────────────────────────────
def make_pitcher_rankings(df_segment: pd.DataFrame, pitch_types_filter: list[str] | None = None) -> pd.DataFrame:
    if df_segment is None or df_segment.empty:
        return pd.DataFrame()

    base = df_segment[df_segment.get('PitcherTeam','') == 'NEB'].copy()
    if base.empty: return pd.DataFrame()

    type_col = type_col_in_df(base)
    if pitch_types_filter is not None and len(pitch_types_filter) > 0 and type_col in base.columns:
        base = base[base[type_col].astype(str).isin(pitch_types_filter)].copy()
        if base.empty: return pd.DataFrame()

    base['Date'] = pd.to_datetime(base['Date'], errors='coerce')
    base["PitcherDisplay"] = base.get("Pitcher", pd.Series(dtype=object)).map(canonicalize_person_name)
    base["PitcherKey"]     = base["PitcherDisplay"].str.lower()

    rows = []
    for pkey, sub in base.groupby("PitcherKey", dropna=False):
        name = sub["PitcherDisplay"].iloc[0] if "PitcherDisplay" in sub.columns else str(pkey)
        app = int(sub['Date'].dropna().dt.date.nunique())

        pa = _terminal_pa_table(sub)
        box = _box_counts_from_PA(pa)
        outs = box["OUTS"]
        ip_float, ip_disp = _compute_IP_from_outs(outs)

        pdm = _plate_metrics_detailed(sub)
        hhbm = _hardhit_barrel_metrics(sub)

        H, HR, BB, HBP, SO = box["H"], box["HR"], box["BB"], box["HBP"], box["SO"]
        WHIP = (BB + H) / ip_float if ip_float > 0 else np.nan
        H9   = (H * 9.0) / ip_float if ip_float > 0 else np.nan

        rows.append({
            "Pitcher": name, "App": app, "IP": ip_disp, "_IP_num": ip_float,
            "H": H, "HR": HR, "BB": BB, "HBP": HBP, "SO": SO,
            "WHIP": WHIP, "H9": H9,
            "BB%": (BB / max(len(pa),1))*100 if len(pa) else np.nan,
            "SO%": (SO / max(len(pa),1))*100 if len(pa) else np.nan,
            "Strike%": pdm.get("StrikePct", np.nan),
            "HH%": hhbm.get("HardHitPct", np.nan),
            "Barrel%": hhbm.get("BarrelPct", np.nan),
            "Zone%": pdm.get("ZonePct", np.nan),
            "Zwhiff%": pdm.get("ZwhiffPct", np.nan),
            "Chase%": pdm.get("ChasePct", np.nan),
            "Whiff%": pdm.get("WhiffPct", np.nan),
        })

    out = pd.DataFrame(rows)
    if out.empty: return out

    desired_order = ["Pitcher","App","IP","H","HR","BB","HBP","SO","WHIP","H9",
                     "BB%","SO%","Strike%","HH%","Barrel%","Zone%","Zwhiff%","Chase%","Whiff%"]
    for c in desired_order:
        if c not in out.columns: out[c] = np.nan
    out = out[desired_order + ["_IP_num"]]

    for c in ["WHIP","H9"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    for c in ["BB%","SO%","Strike%","HH%","Barrel%","Zone%","Zwhiff%","Chase%","Whiff%"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(1)
    for c in ["App","H","HR","BB","HBP","SO"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    out = out.sort_values(["WHIP","SO"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return out

def make_team_averages(df_segment: pd.DataFrame, pitch_types_filter: list[str] | None = None) -> pd.DataFrame:
    if df_segment is None or df_segment.empty:
        return pd.DataFrame()

    base = df_segment[df_segment.get('PitcherTeam','') == 'NEB'].copy()
    if base.empty: return pd.DataFrame()

    type_col = type_col_in_df(base)
    if pitch_types_filter:
        if type_col in base.columns:
            base = base[base[type_col].astype(str).isin(pitch_types_filter)].copy()
        if base.empty: return pd.DataFrame()

    pa = _terminal_pa_table(base)
    box = _box_counts_from_PA(pa)
    outs = box["OUTS"]
    ip_float, _ = _compute_IP_from_outs(outs)

    pdm = _plate_metrics_detailed(base)
    hhbm = _hardhit_barrel_metrics(base)

    H, BB, SO = box["H"], box["BB"], box["SO"]
    WHIP = (BB + H) / ip_float if ip_float > 0 else np.nan
    H9   = (H * 9.0) / ip_float if ip_float > 0 else np.nan

    PA_n = len(pa)
    BBpct = (BB / max(PA_n, 1)) * 100.0
    SOpct = (SO / max(PA_n, 1)) * 100.0

    row = {
        "Team": "NEB",
        "WHIP": round(WHIP, 3) if pd.notna(WHIP) else np.nan,
        "H9": round(H9, 2) if pd.notna(H9) else np.nan,
        "BB%": round(BBpct, 1),
        "SO%": round(SOpct, 1),
        "Strike%": round(pdm.get("StrikePct", np.nan), 1),
        "Zone%": round(pdm.get("ZonePct", np.nan), 1),
        "Zwhiff%": round(pdm.get("ZwhiffPct", np.nan), 1),
        "HH%": round(hhbm.get("HardHitPct", np.nan), 1),
        "Barrel%": round(hhbm.get("BarrelPct", np.nan), 1),
        "Whiff%": round(pdm.get("WhiffPct", np.nan), 1),
        "Chase%": round(pdm.get("ChasePct", np.nan), 1),
    }

    cols = ["Team","WHIP","H9","BB%","SO%","Strike%","Zone%","Zwhiff%","HH%","Barrel%","Whiff%","Chase%"]
    return pd.DataFrame([row])[cols]

# ─── Movement profile report ──────────────────────────────────────────────────
def combined_pitcher_report(df, pitcher_name, logo_img, coverage=0.8, season_label="Season"):
    type_col = type_col_in_df(df)
    pitch_col = pick_col(df, "PitchCall","Pitch Call","Call") or "PitchCall"
    speed_col = pick_col(df, "RelSpeed","Relspeed","ReleaseSpeed","RelSpeedMPH")
    spin_col  = pick_col(df, "SpinRate","Spinrate","ReleaseSpinRate","Spin")
    ivb_col   = pick_col(df, "InducedVertBreak","IVB","Induced Vert Break")
    hb_col    = pick_col(df, "HorzBreak","HorizontalBreak","HB")
    rh_col    = pick_col(df, "RelHeight","Relheight","ReleaseHeight")
    ext_col   = pick_col(df, "Extension","Ext","ReleaseExtension")

    df_p = subset_by_pitcher_if_possible(df, pitcher_name)
    if df_p.empty:
        return None, None

    try:
        grp = df_p.groupby(type_col, dropna=False)
    except KeyError:
        return None, None

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
    add_mean(rh_col, 'Rel Height', 2); add_mean(ext_col, 'Extension', 2)
    summary = summary.sort_values('Pitches', ascending=False)

    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.7], hspace=0.3)
    axm = fig.add_subplot(gs[0, 0])
    axm.set_title('Movement Profile', fontweight='bold', fontsize=14)
    axm.axhline(0, ls='--', color='grey', alpha=0.5)
    axm.axvline(0, ls='--', color='grey', alpha=0.5)
    chi2v = chi2.ppf(coverage, df=2)

    for ptype, g in df_p.groupby(type_col, dropna=False):
        clr = get_pitch_color(ptype)
        x = pd.to_numeric(g.get('HorzBreak', g.get('HB')), errors='coerce')
        y = pd.to_numeric(g.get('InducedVertBreak', g.get('IVB')), errors='coerce')
        mask = x.notna() & y.notna()
        if mask.any():
            axm.scatter(x[mask], y[mask], label=str(ptype), color=clr, alpha=0.7, s=60, edgecolors='white', linewidths=0.5)
            if mask.sum() > 1:
                X = np.vstack((x[mask], y[mask]))
                cov = np.cov(X)
                try:
                    vals, vecs = np.linalg.eigh(cov)
                    ord_ = vals.argsort()[::-1]
                    vals, vecs = vals[ord_], vecs[:, ord_]
                    ang = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                    w, h = 2*np.sqrt(vals*chi2v)
                    axm.add_patch(Ellipse((x[mask].mean(), y[mask].mean()), w, h, angle=ang,
                                          edgecolor=clr, facecolor=clr, alpha=0.15, ls='--', lw=1.5))
                except Exception:
                    pass

    axm.set_xlim(-30,30); axm.set_ylim(-30,30); axm.set_aspect('equal','box')
    axm.set_xlabel('Horizontal Break (inches)', fontsize=11, fontweight='500')
    axm.set_ylabel('Induced Vertical Break (inches)', fontsize=11, fontweight='500')
    axm.legend(title='Pitch Type', fontsize=9, title_fontsize=10, loc='upper right', framealpha=0.95)
    axm.grid(True, alpha=0.2, linestyle=':')

    axt = fig.add_subplot(gs[1, 0]); axt.axis('off')
    tbl = axt.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.5, 1.5)

    for i in range(len(summary.columns)):
        tbl[(0, i)].set_facecolor(HUSKER_RED)
        tbl[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(summary) + 1):
        for j in range(len(summary.columns)):
            tbl[(i, j)].set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')
            tbl[(i, j)].set_edgecolor('#E0E0E0')

    axt.set_title('Performance Metrics by Pitch Type', fontweight='bold', y=0.88, fontsize=12)

    if logo_img is not None:
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10)
        axl.imshow(logo_img); axl.axis('off')

    fig.suptitle(f"{canonicalize_person_name(pitcher_name)}\n{season_label}", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, summary

# ─── Overall Performance Metrics ──────────────────────────────────────────────
def create_overall_performance_table(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()

    type_col = type_col_in_df(df)
    call_col = pick_col(df, "PitchCall", "Pitch Call", "Call", "PitchResult")
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")
    ev_col = pick_col(df, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV", "LaunchSpeed")

    if not call_col:
        return pd.DataFrame()

    total_pitches = len(df)

    def calc_metrics(subset):
        n = len(subset)
        if n == 0:
            return {}

        row = {'Pitches': n}
        usage_pct = (n / total_pitches) * 100
        row['Usage%'] = round(usage_pct, 1)

        is_strike = subset[call_col].isin(['StrikeCalled','StrikeSwinging',
                                           'FoulBallNotFieldable','FoulBallFieldable','InPlay'])
        row['Strike%'] = round(is_strike.mean() * 100, 1)

        is_swing = subset[call_col].isin(['StrikeSwinging','FoulBallNotFieldable',
                                          'FoulBallFieldable','InPlay'])
        is_whiff = subset[call_col].eq('StrikeSwinging')
        swings = is_swing.sum()
        row['Whiff%'] = round((is_whiff.sum() / swings * 100), 1) if swings > 0 else np.nan

        if x_col and y_col:
            xs = pd.to_numeric(subset[x_col], errors="coerce")
            ys = pd.to_numeric(subset[y_col], errors="coerce")
            l, b, w, h = get_zone_bounds()
            in_zone = xs.between(l, l+w) & ys.between(b, b+h)

            row['Zone%'] = round(in_zone.mean() * 100, 1) if in_zone.notna().any() else np.nan

            out_zone = ~in_zone & xs.notna() & ys.notna()
            chases = (is_swing & out_zone).sum()
            out_zone_pitches = out_zone.sum()
            row['Chase%'] = round((chases / out_zone_pitches * 100), 1) if out_zone_pitches > 0 else np.nan

            zone_swings = (is_swing & in_zone).sum()
            zone_contacts = ((is_swing & ~is_whiff) & in_zone).sum()
            row['Zone Contact%'] = round((zone_contacts / zone_swings * 100), 1) if zone_swings > 0 else np.nan

        is_inplay = subset[call_col].eq('InPlay')
        row['InPlay%'] = round(is_inplay.mean() * 100, 1)

        if ev_col:
            ev = pd.to_numeric(subset[ev_col], errors="coerce")
            bip = is_inplay & ev.notna()
            hard = (ev >= 95.0) & bip
            bip_count = bip.sum()
            row['HardHit%'] = round((hard.sum() / bip_count * 100), 1) if bip_count > 0 else np.nan
            ev_bip = ev[bip]
            row['Avg EV'] = round(ev_bip.mean(), 1) if len(ev_bip) > 0 else np.nan

        return row

    rows = []
    if type_col and type_col in df.columns:
        for ptype, grp in df.groupby(type_col, dropna=False):
            metrics = calc_metrics(grp)
            if metrics:
                metrics['Pitch Type'] = str(ptype)
                rows.append(metrics)

    overall_metrics = calc_metrics(df)
    if overall_metrics:
        overall_metrics['Pitch Type'] = 'OVERALL'
        rows.append(overall_metrics)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    cols = ['Pitch Type', 'Pitches', 'Usage%', 'Strike%', 'Whiff%', 'Zone%', 'Chase%',
            'Zone Contact%', 'InPlay%', 'HardHit%', 'Avg EV']
    cols = [c for c in cols if c in result.columns]
    result = result[cols]
    result = result.sort_values('Pitches', ascending=False)
    overall_row = result[result['Pitch Type'] == 'OVERALL']
    other_rows = result[result['Pitch Type'] != 'OVERALL']
    result = pd.concat([other_rows, overall_row], ignore_index=True)
    return result

# ─── Count Situation Comparison ───────────────────────────────────────────────
def create_count_situation_comparison(df: pd.DataFrame, pitch_type_filter: str = None):
    if df is None or df.empty:
        return pd.DataFrame()

    type_col = type_col_in_df(df)
    balls_col = pick_col(df, "Balls", "Ball Count", "BallCount", "balls")
    strikes_col = pick_col(df, "Strikes", "Strike Count", "StrikeCount", "strikes")
    call_col = pick_col(df, "PitchCall", "Pitch Call", "Call", "PitchResult")
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")
    ev_col = pick_col(df, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV", "LaunchSpeed")

    if not balls_col or not strikes_col:
        return pd.DataFrame()

    original_df = df.copy()
    orig_balls = pd.to_numeric(original_df[balls_col], errors="coerce")
    orig_strikes = pd.to_numeric(original_df[strikes_col], errors="coerce")
    original_df["Balls"] = orig_balls
    original_df["Strikes"] = orig_strikes

    work = df.copy()
    is_filtered = False
    if pitch_type_filter and pitch_type_filter != "Overall" and type_col and type_col in work.columns:
        work = work[work[type_col].astype(str) == pitch_type_filter].copy()
        is_filtered = True

    balls = pd.to_numeric(work[balls_col], errors="coerce")
    strikes = pd.to_numeric(work[strikes_col], errors="coerce")
    work["Balls"] = balls
    work["Strikes"] = strikes

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

        if is_filtered:
            orig_mask_balls = orig_balls.notna() & orig_strikes.notna()
            if sit_name == 'First Pitch (0-0)':
                orig_mask = (orig_balls == 0) & (orig_strikes == 0) & orig_mask_balls
            elif sit_name == 'Ahead in Count':
                orig_mask = (orig_strikes > orig_balls) & orig_mask_balls
            elif sit_name == 'Even Count':
                orig_mask = (orig_strikes == orig_balls) & ((orig_balls + orig_strikes) > 0) & orig_mask_balls
            elif sit_name == 'Behind in Count':
                orig_mask = (orig_balls > orig_strikes) & orig_mask_balls
            elif sit_name == 'Two Strikes':
                orig_mask = (orig_strikes == 2) & orig_mask_balls
            elif sit_name == 'Three Balls':
                orig_mask = (orig_balls == 3) & orig_mask_balls
            else:
                orig_mask = pd.Series(False, index=original_df.index)

            total_in_situation = int(orig_mask.sum())
            row['Usage%'] = round((n / total_in_situation * 100), 1) if total_in_situation > 0 else 0.0

        if call_col:
            is_strike = subset[call_col].isin(['StrikeCalled','StrikeSwinging',
                                               'FoulBallNotFieldable','FoulBallFieldable','InPlay'])
            row['Strike%'] = round(is_strike.mean() * 100, 1)

            is_swing = subset[call_col].isin(['StrikeSwinging','FoulBallNotFieldable',
                                              'FoulBallFieldable','InPlay'])
            is_whiff = subset[call_col].eq('StrikeSwinging')
            swings = is_swing.sum()
            row['Whiff%'] = round((is_whiff.sum() / swings * 100), 1) if swings > 0 else np.nan

        if x_col and y_col:
            xs = pd.to_numeric(subset[x_col], errors="coerce")
            ys = pd.to_numeric(subset[y_col], errors="coerce")
            l, b_zone, w, h = get_zone_bounds()
            in_zone = xs.between(l, l+w) & ys.between(b_zone, b_zone+h)
            row['Zone%'] = round(in_zone.mean() * 100, 1) if in_zone.notna().any() else np.nan

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

        if call_col and ev_col:
            is_inplay = subset[call_col].eq('InPlay')
            ev = pd.to_numeric(subset[ev_col], errors="coerce")
            bip = is_inplay & ev.notna()
            hard = (ev >= 95.0) & bip
            bip_count = bip.sum()
            row['HardHit%'] = round((hard.sum() / bip_count * 100), 1) if bip_count > 0 else np.nan

        rows.append(row)

    result = pd.DataFrame(rows).sort_values('Pitches', ascending=False)

    if 'Usage%' in result.columns:
        cols = result.columns.tolist()
        cols.remove('Usage%')
        pitches_idx = cols.index('Pitches')
        cols.insert(pitches_idx + 1, 'Usage%')
        result = result[cols]

    return result

# ─── Density computation ───────────────────────────────────────────────────────
def compute_density_pitcher(x, y, xi_m, yi_m):
    coords = np.vstack([x, y])
    mask = np.isfinite(coords).all(axis=0)
    if mask.sum() <= 1:
        return np.zeros(xi_m.shape)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(coords[:, mask])
        return kde(np.vstack([xi_m.ravel(), yi_m.ravel()])).reshape(xi_m.shape)
    except Exception:
        return np.zeros(xi_m.shape)

# ─── Three Outcome Heatmaps ────────────────────────────────────────────────────
def create_outcome_heatmaps(df: pd.DataFrame, pitcher_name: str, pitch_type_filter: str = None):
    if df is None or df.empty:
        return None

    type_col = type_col_in_df(df)
    call_col = pick_col(df, "PitchCall", "Pitch Call", "Call", "PitchResult")
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")
    ev_col = pick_col(df, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV", "LaunchSpeed")

    if not call_col or not x_col or not y_col:
        return None

    work = df.copy()
    if pitch_type_filter and pitch_type_filter != "Overall" and type_col and type_col in work.columns:
        work = work[work[type_col].astype(str) == pitch_type_filter].copy()

    xs = pd.to_numeric(work[x_col], errors="coerce")
    ys = pd.to_numeric(work[y_col], errors="coerce")

    is_whiff = work[call_col].eq('StrikeSwinging')
    is_contact = work[call_col].isin(['FoulBallNotFieldable', 'FoulBallFieldable', 'InPlay'])

    if ev_col:
        ev = pd.to_numeric(work[ev_col], errors="coerce")
        is_hardhit = (ev >= 95.0) & work[call_col].eq('InPlay')
    else:
        is_hardhit = pd.Series(False, index=work.index)

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.25)

    def _panel(ax, title, mask):
        l, b, w, h = get_zone_bounds()
        ax.add_patch(Rectangle((l, b), w, h, fill=False, linewidth=2, color='black'))
        dx, dy = w/3, h/3
        for i in (1, 2):
            ax.add_line(Line2D([l+i*dx]*2, [b, b+h], linestyle='--', color='gray', linewidth=1))
            ax.add_line(Line2D([l, l+w], [b+i*dy]*2, linestyle='--', color='gray', linewidth=1))

        subset_x = xs[mask].to_numpy()
        subset_y = ys[mask].to_numpy()
        valid_mask = np.isfinite(subset_x) & np.isfinite(subset_y)
        subset_x = subset_x[valid_mask]
        subset_y = subset_y[valid_mask]

        if len(subset_x) < 5:
            ax.plot(subset_x, subset_y, 'o', color='red', alpha=0.8, markersize=8)
        else:
            xi = np.linspace(-3, 3, 200)
            yi = np.linspace(0, 5, 200)
            xi_m, yi_m = np.meshgrid(xi, yi)
            zi = compute_density_pitcher(subset_x, subset_y, xi_m, yi_m)
            ax.imshow(zi, origin='lower', extent=[-3, 3, 0, 5],
                     aspect='equal', cmap=custom_cmap, alpha=0.8)
            ax.add_patch(Rectangle((l, b), w, h, fill=False, linewidth=2, color='black'))
            for i in (1, 2):
                ax.add_line(Line2D([l+i*dx]*2, [b, b+h], linestyle='--', color='gray', linewidth=1))
                ax.add_line(Line2D([l, l+w], [b+i*dy]*2, linestyle='--', color='gray', linewidth=1))

        ax.set_xlim(-3, 3); ax.set_ylim(0, 5)
        ax.set_aspect('equal', 'box')
        ax.set_title(title, fontsize=14, pad=10, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        n_pitches = mask.sum()
        ax.text(0.5, 0.02, f"n = {n_pitches}", transform=ax.transAxes,
               ha='center', va='bottom', fontsize=11, style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax1 = fig.add_subplot(gs[0, 0])
    _panel(ax1, "Whiff Locations", is_whiff & xs.notna() & ys.notna())
    ax2 = fig.add_subplot(gs[0, 1])
    _panel(ax2, "Contact Locations", is_contact & xs.notna() & ys.notna())
    ax3 = fig.add_subplot(gs[0, 2])
    _panel(ax3, "Hard Hit Locations (95+ mph)", is_hardhit & xs.notna() & ys.notna())

    title_text = f"{canonicalize_person_name(pitcher_name)} - Outcome Heatmaps"
    if pitch_type_filter and pitch_type_filter != "Overall":
        title_text += f" ({pitch_type_filter})"

    plt.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ─── Count Leverage Heatmaps ──────────────────────────────────────────────────
def create_count_leverage_heatmaps(df: pd.DataFrame, pitcher_name: str, pitch_type_filter: str = None):
    if df is None or df.empty:
        return None

    type_col = type_col_in_df(df)
    balls_col = pick_col(df, "Balls", "Ball Count", "BallCount", "balls")
    strikes_col = pick_col(df, "Strikes", "Strike Count", "StrikeCount", "strikes")
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")

    if not balls_col or not strikes_col or not x_col or not y_col:
        return None

    work = df.copy()
    if pitch_type_filter and pitch_type_filter != "Overall" and type_col and type_col in work.columns:
        work = work[work[type_col].astype(str) == pitch_type_filter].copy()

    balls = pd.to_numeric(work[balls_col], errors="coerce")
    strikes = pd.to_numeric(work[strikes_col], errors="coerce")
    work["Balls"] = balls
    work["Strikes"] = strikes

    ahead_mask = strikes > balls
    behind_mask = balls > strikes
    two_strike_mask = strikes == 2

    xs = pd.to_numeric(work[x_col], errors="coerce")
    ys = pd.to_numeric(work[y_col], errors="coerce")

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.25)

    def _panel(ax, title, mask):
        l, b, w, h = get_zone_bounds()
        ax.add_patch(Rectangle((l, b), w, h, fill=False, linewidth=2, color='black'))
        dx, dy = w/3, h/3
        for i in (1, 2):
            ax.add_line(Line2D([l+i*dx]*2, [b, b+h], linestyle='--', color='gray', linewidth=1))
            ax.add_line(Line2D([l, l+w], [b+i*dy]*2, linestyle='--', color='gray', linewidth=1))

        subset_x = xs[mask].to_numpy()
        subset_y = ys[mask].to_numpy()
        valid_mask = np.isfinite(subset_x) & np.isfinite(subset_y)
        subset_x = subset_x[valid_mask]
        subset_y = subset_y[valid_mask]

        if len(subset_x) < 10:
            ax.plot(subset_x, subset_y, 'o', color='deepskyblue', alpha=0.8, markersize=6)
        else:
            xi = np.linspace(-3, 3, 200)
            yi = np.linspace(0, 5, 200)
            xi_m, yi_m = np.meshgrid(xi, yi)
            zi = compute_density_pitcher(subset_x, subset_y, xi_m, yi_m)
            ax.imshow(zi, origin='lower', extent=[-3, 3, 0, 5],
                     aspect='equal', cmap=custom_cmap, alpha=0.8)
            ax.add_patch(Rectangle((l, b), w, h, fill=False, linewidth=2, color='black'))
            for i in (1, 2):
                ax.add_line(Line2D([l+i*dx]*2, [b, b+h], linestyle='--', color='gray', linewidth=1))
                ax.add_line(Line2D([l, l+w], [b+i*dy]*2, linestyle='--', color='gray', linewidth=1))

        ax.set_xlim(-3, 3); ax.set_ylim(0, 5)
        ax.set_aspect('equal', 'box')
        ax.set_title(title, fontsize=12, pad=8, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        n_pitches = mask.sum()
        ax.text(0.5, 0.02, f"n = {n_pitches}", transform=ax.transAxes,
               ha='center', va='bottom', fontsize=10, style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax1 = fig.add_subplot(gs[0, 0])
    _panel(ax1, "Pitcher Ahead in Count", ahead_mask & balls.notna() & strikes.notna())
    ax2 = fig.add_subplot(gs[0, 1])
    _panel(ax2, "Hitter Ahead in Count", behind_mask & balls.notna() & strikes.notna())
    ax3 = fig.add_subplot(gs[0, 2])
    _panel(ax3, "Two Strike Counts", two_strike_mask & balls.notna() & strikes.notna())

    title_text = f"{canonicalize_person_name(pitcher_name)} - Count Leveraging"
    if pitch_type_filter and pitch_type_filter != "Overall":
        title_text += f" ({pitch_type_filter})"

    plt.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# ─── Pitch Type Location Heatmaps ─────────────────────────────────────────────
def create_pitch_type_location_heatmaps(df: pd.DataFrame, pitcher_name: str, pitch_types_to_show: list = None, show_top_n: int = 3):
    if df is None or df.empty:
        return None

    type_col = type_col_in_df(df)
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")

    if not type_col or not x_col or not y_col:
        return None

    work = subset_by_pitcher_if_possible(df, pitcher_name)

    if pitch_types_to_show is None:
        pitch_counts = work[type_col].value_counts()
        pitch_types = pitch_counts.head(show_top_n).index.tolist()
    else:
        pitch_types = pitch_types_to_show

    if len(pitch_types) == 0:
        return None

    xs = pd.to_numeric(work[x_col], errors="coerce")
    ys = pd.to_numeric(work[y_col], errors="coerce")

    n_pitches = len(pitch_types)
    n_cols = min(3, n_pitches)
    n_rows = (n_pitches + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(7 * n_cols, 8 * n_rows), facecolor='white')
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.3, hspace=0.4)

    def _panel(ax, pitch_type):
        ax.set_facecolor('#f8f9fa')
        l, b, w, h = get_zone_bounds()
        zone_rect = Rectangle((l, b), w, h, fill=False, linewidth=3, edgecolor='#2c3e50', zorder=10)
        ax.add_patch(zone_rect)
        dx, dy = w/3, h/3
        for i in (1, 2):
            ax.add_line(Line2D([l+i*dx]*2, [b, b+h], linestyle='--', color='#34495e', linewidth=1.5, alpha=0.5, zorder=10))
            ax.add_line(Line2D([l, l+w], [b+i*dy]*2, linestyle='--', color='#34495e', linewidth=1.5, alpha=0.5, zorder=10))

        mask = work[type_col].astype(str) == pitch_type
        subset_x = xs[mask].to_numpy()
        subset_y = ys[mask].to_numpy()
        valid_mask = np.isfinite(subset_x) & np.isfinite(subset_y)
        subset_x = subset_x[valid_mask]
        subset_y = subset_y[valid_mask]
        n_pitches_type = len(subset_x)

        in_zone = ((subset_x >= l) & (subset_x <= l+w) & (subset_y >= b) & (subset_y <= b+h))
        zone_pct = (in_zone.sum() / n_pitches_type * 100) if n_pitches_type > 0 else 0
        avg_x = np.mean(subset_x) if n_pitches_type > 0 else 0
        avg_y = np.mean(subset_y) if n_pitches_type > 0 else 0

        if n_pitches_type < 10:
            pitch_color = get_pitch_color(pitch_type)
            ax.plot(subset_x, subset_y, 'o', color=pitch_color, alpha=0.8,
                   markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=5)
        else:
            xi = np.linspace(-3, 3, 200)
            yi = np.linspace(0, 5, 200)
            xi_m, yi_m = np.meshgrid(xi, yi)
            zi = compute_density_pitcher(subset_x, subset_y, xi_m, yi_m)
            ax.imshow(zi, origin='lower', extent=[-3, 3, 0, 5],
                     aspect='equal', cmap=custom_cmap, alpha=0.9, zorder=1)
            zone_rect = Rectangle((l, b), w, h, fill=False, linewidth=3, edgecolor='#2c3e50', zorder=10)
            ax.add_patch(zone_rect)
            for i in (1, 2):
                ax.add_line(Line2D([l+i*dx]*2, [b, b+h], linestyle='--', color='#34495e', linewidth=1.5, alpha=0.5, zorder=10))
                ax.add_line(Line2D([l, l+w], [b+i*dy]*2, linestyle='--', color='#34495e', linewidth=1.5, alpha=0.5, zorder=10))
            ax.plot(avg_x, avg_y, 'X', color='#2c3e50', markersize=15,
                   markeredgecolor='white', markeredgewidth=2, zorder=15)

        stats_text = f"{pitch_type}\n{n_pitches_type} pitches  |  Zone: {zone_pct:.1f}%"
        ax.text(0.5, -0.12, stats_text, transform=ax.transAxes,
               fontsize=14, color='#2c3e50', ha='center', va='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#bbb', linewidth=2, alpha=0.95))

        xmin, xmax, ymin, ymax = get_view_bounds()
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

    for idx, pitch_type in enumerate(pitch_types):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        _panel(ax, pitch_type)

    title_suffix = " (Top 3 Pitches)" if pitch_types_to_show is None else ""
    fig.text(0.5, 0.985, f"{canonicalize_person_name(pitcher_name)} - Pitch Location Analysis{title_suffix}",
            fontsize=20, fontweight='bold', color='#2c3e50', ha='center', va='top')
    fig.text(0.5, 0.965, "(Pitcher's Perspective)",
            fontsize=12, color='#7f8c8d', ha='center', va='top', style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

# ─── Miss Location Heatmaps ───────────────────────────────────────────────────
def create_miss_location_heatmaps(df: pd.DataFrame, pitcher_name: str, pitch_types_to_show: list = None, show_top_n: int = 3):
    if df is None or df.empty:
        return None

    type_col = type_col_in_df(df)
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")
    call_col = pick_col(df, "PitchCall", "Pitch Call", "Call", "PitchResult")

    if not type_col or not x_col or not y_col or not call_col:
        return None

    work = subset_by_pitcher_if_possible(df, pitcher_name)
    work = work[work[call_col].astype(str).str.lower().isin(['ball', 'ballcalled'])].copy()

    if work.empty:
        return None

    if pitch_types_to_show is None:
        pitch_counts = work[type_col].value_counts()
        pitch_types = pitch_counts.head(show_top_n).index.tolist()
    else:
        pitch_types = pitch_types_to_show

    if len(pitch_types) == 0:
        return None

    xs = pd.to_numeric(work[x_col], errors="coerce")
    ys = pd.to_numeric(work[y_col], errors="coerce")

    n_pitches = len(pitch_types)
    n_cols = min(3, n_pitches)
    n_rows = (n_pitches + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(7 * n_cols, 8 * n_rows), facecolor='white')
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.3, hspace=0.4)

    def _panel(ax, pitch_type):
        ax.set_facecolor('#f8f9fa')
        l, b, w, h = get_zone_bounds()
        zone_rect = Rectangle((l, b), w, h, fill=False, linewidth=3, edgecolor='#2c3e50', zorder=10)
        ax.add_patch(zone_rect)
        dx, dy = w/3, h/3
        for i in (1, 2):
            ax.add_line(Line2D([l+i*dx]*2, [b, b+h], linestyle='--', color='#34495e', linewidth=1.5, alpha=0.5, zorder=10))
            ax.add_line(Line2D([l, l+w], [b+i*dy]*2, linestyle='--', color='#34495e', linewidth=1.5, alpha=0.5, zorder=10))

        mask = work[type_col].astype(str) == pitch_type
        subset_x = xs[mask].to_numpy()
        subset_y = ys[mask].to_numpy()
        valid_mask = np.isfinite(subset_x) & np.isfinite(subset_y)
        subset_x = subset_x[valid_mask]
        subset_y = subset_y[valid_mask]
        n_misses = len(subset_x)

        avg_x = np.mean(subset_x) if n_misses > 0 else 0
        avg_y = np.mean(subset_y) if n_misses > 0 else 0

        if n_misses < 5:
            ax.plot(subset_x, subset_y, 'o', color='#e74c3c', alpha=0.8,
                   markersize=12, markeredgecolor='white', markeredgewidth=2, zorder=5)
        else:
            xi = np.linspace(-3, 3, 200)
            yi = np.linspace(0, 5, 200)
            xi_m, yi_m = np.meshgrid(xi, yi)
            zi = compute_density_pitcher(subset_x, subset_y, xi_m, yi_m)
            ax.imshow(zi, origin='lower', extent=[-3, 3, 0, 5],
                     aspect='equal', cmap=custom_cmap, alpha=0.9, zorder=1)
            zone_rect = Rectangle((l, b), w, h, fill=False, linewidth=3, edgecolor='#2c3e50', zorder=10)
            ax.add_patch(zone_rect)
            for i in (1, 2):
                ax.add_line(Line2D([l+i*dx]*2, [b, b+h], linestyle='--', color='#34495e', linewidth=1.5, alpha=0.5, zorder=10))
                ax.add_line(Line2D([l, l+w], [b+i*dy]*2, linestyle='--', color='#34495e', linewidth=1.5, alpha=0.5, zorder=10))
            ax.plot(avg_x, avg_y, 'X', color='#c0392b', markersize=15,
                   markeredgecolor='white', markeredgewidth=2, zorder=15)

        stats_text = f"{pitch_type}\n{n_misses} misses (called balls)"
        ax.text(0.5, -0.12, stats_text, transform=ax.transAxes,
               fontsize=14, color='#c0392b', ha='center', va='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#e74c3c', linewidth=2, alpha=0.95))

        xmin, xmax, ymin, ymax = get_view_bounds()
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

    for idx, pitch_type in enumerate(pitch_types):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        _panel(ax, pitch_type)

    title_suffix = " (Top 3 Pitches)" if pitch_types_to_show is None else ""
    fig.text(0.5, 0.985, f"{canonicalize_person_name(pitcher_name)} - Miss Locations{title_suffix}",
            fontsize=20, fontweight='bold', color='#c0392b', ha='center', va='top')
    fig.text(0.5, 0.965, "(Where Called Balls Land - Pitcher's Perspective)",
            fontsize=12, color='#7f8c8d', ha='center', va='top', style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

# Part 4: Spray Charts and Pitch Sequencing

# ─── Spray Chart Helpers ───────────────────────────────────────────────────────
def draw_dirt_diamond(ax, origin=(0.0, 0.0), size: float = 80, base_size: float = 8,
                      grass_scale: float = 0.4, custom_wall_distances: list = None):
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
    else:
        outfield_radius = size * 1.7
        ax.add_patch(Wedge(home, outfield_radius, 45, 135, facecolor='#228B22', edgecolor='black', linewidth=2))

    ax.add_patch(Wedge(home, size, 45, 135, facecolor='#ED8B00', edgecolor='black', linewidth=2))

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
        ax.add_patch(Rectangle((pos[0] - base_size/2, pos[1] - base_size/2),
                              base_size, base_size, facecolor='white', edgecolor='black', linewidth=1))

    half = base_size / 2
    plate = Polygon([
        (home[0] - half, home[1]), (home[0] + half, home[1]),
        (home[0] + half * 0.6, home[1] - half * 0.8),
        (home[0], home[1] - base_size),
        (home[0] - half * 0.6, home[1] - half * 0.8)
    ], closed=True, facecolor='white', edgecolor='black', linewidth=1)
    ax.add_patch(plate)

    for angle in [45, 135]:
        rad = math.radians(angle)
        end = home + np.array([outfield_radius * 1.1 * math.cos(rad), outfield_radius * 1.1 * math.sin(rad)])
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

def create_spray_chart(df: pd.DataFrame, pitcher_name: str, season_label: str = "Season"):
    df_p = subset_by_pitcher_if_possible(df, pitcher_name)

    bearing_col = pick_col(df_p, "Bearing", "HitBearing", "Hit Bearing", "Direction", "Angle")
    distance_col = pick_col(df_p, "Distance", "HitDistance", "Hit Distance", "Dist")
    result_col = pick_col(df_p, "PlayResult", "Result", "Event", "PAResult")
    type_col = pick_col(df_p, "HitType", "Hit Type", "BattedBallType", "BBType", "TaggedHitType")
    ev_col = pick_col(df_p, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV")
    call_col = pick_col(df_p, "PitchCall", "Pitch Call", "Call")

    if not bearing_col or not distance_col:
        return None, pd.DataFrame()

    if call_col:
        bip = df_p[df_p[call_col].eq('InPlay')].copy()
    else:
        bip = df_p.copy()

    if bip.empty:
        return None, pd.DataFrame()

    bearing = pd.to_numeric(bip[bearing_col], errors="coerce")
    distance = pd.to_numeric(bip[distance_col], errors="coerce")
    valid = bearing.notna() & distance.notna()
    bip = bip[valid].copy()
    bearing = bearing[valid]
    distance = distance[valid]

    if bip.empty:
        return None, pd.DataFrame()

    coords = [bearing_distance_to_xy(b, d) for b, d in zip(bearing, distance)]
    bip['x'] = [c[0] for c in coords]
    bip['y'] = [c[1] for c in coords]

    fig, ax = plt.subplots(figsize=(8, 8))

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

    draw_dirt_diamond(ax, origin=(0.0, 0.0), size=70, custom_wall_distances=wall_data)

    wall_x = [dist * np.cos(np.radians(ang)) for ang, dist in wall_data]
    wall_y = [dist * np.sin(np.radians(ang)) for ang, dist in wall_data]
    ax.plot(wall_x, wall_y, 'k-', linewidth=2, zorder=10)

    for angle, dist, label in [(45, 335, '335'), (90, 395, '395'), (135, 325, '325')]:
        rad = np.radians(angle)
        x = dist * np.cos(rad)
        y = dist * np.sin(rad)
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', linewidth=1.5, alpha=0.9), zorder=11)

    def categorize_hit_type(hit_type):
        if pd.isna(hit_type): return 'Other'
        ht = str(hit_type).lower()
        if 'ground' in ht: return 'GroundBall'
        elif 'line' in ht: return 'LineDrive'
        elif 'fly' in ht: return 'FlyBall'
        elif 'popup' in ht or 'pop' in ht: return 'Popup'
        else: return 'Other'

    bip['HitCategory'] = bip.get(type_col, pd.Series(dtype=object)).apply(categorize_hit_type)

    hit_type_colors = {
        'GroundBall': '#DC143C', 'LineDrive': '#FFD700',
        'FlyBall': '#1E90FF', 'Popup': '#FF69B4', 'Other': '#A9A9A9'
    }

    for idx, row in bip.iterrows():
        hit_cat = row['HitCategory']
        play_result = str(row.get(result_col, ''))
        edgecolor = 'black'
        linewidth = 1.5 if play_result in ['Single', 'Double', 'Triple', 'HomeRun'] else 0.8
        ax.scatter(row['x'], row['y'], c=hit_type_colors.get(hit_cat, '#A9A9A9'),
                  s=80, marker='o', edgecolors=edgecolor, linewidths=linewidth, alpha=0.85, zorder=20)

    legend_elements = []
    for hit_type in ['GroundBall', 'LineDrive', 'FlyBall', 'Popup']:
        count = (bip['HitCategory'] == hit_type).sum()
        if count > 0:
            label = hit_type.replace('GroundBall', 'Ground Ball').replace('LineDrive', 'Line Drive').replace('FlyBall', 'Fly Ball')
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=hit_type_colors[hit_type], markersize=7,
                                         markeredgecolor='black', markeredgewidth=1, label=f'{label} ({count})'))

    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=7,
               markeredgecolor='black', markeredgewidth=1.5, label='Hit (thick edge)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=7,
               markeredgecolor='black', markeredgewidth=0.8, label='Out (thin edge)')
    ])

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
             frameon=True, fancybox=True, shadow=True, fontsize=8)

    max_dist = max(bip['Distance'].max(), 400)
    ax.set_xlim(-max_dist * 0.85, max_dist * 0.85)
    ax.set_ylim(-30, max_dist * 1.1)
    ax.set_aspect('equal')
    ax.set_title(f"{canonicalize_person_name(pitcher_name)} — Hits Allowed (Batter's View)\n{season_label}",
                fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    summary_data = []
    for label in set(bip['HitCategory']):
        mask = bip['HitCategory'] == label
        count = mask.sum()
        avg_dist = bip.loc[mask, 'Distance'].mean() if count > 0 else np.nan
        if ev_col:
            ev_subset = pd.to_numeric(bip.loc[mask, ev_col], errors="coerce")
            avg_ev = ev_subset.mean() if len(ev_subset.dropna()) > 0 else np.nan
            max_ev = ev_subset.max() if len(ev_subset.dropna()) > 0 else np.nan
        else:
            avg_ev = max_ev = np.nan
        summary_data.append({
            'Type': label.replace('GroundBall', 'Ground Ball').replace('LineDrive', 'Line Drive').replace('FlyBall', 'Fly Ball'),
            'Count': int(count),
            'Avg Distance': round(avg_dist, 1) if pd.notna(avg_dist) else np.nan,
            'Avg Exit Velo': round(avg_ev, 1) if pd.notna(avg_ev) else np.nan,
            'Max Exit Velo': round(max_ev, 1) if pd.notna(max_ev) else np.nan,
        })

    summary_df = pd.DataFrame(summary_data).sort_values('Count', ascending=False)
    return fig, summary_df

# ─── Pitch Sequencing ─────────────────────────────────────────────────────────
def analyze_pitch_sequences(df: pd.DataFrame, pitcher_name: str):
    df_p = subset_by_pitcher_if_possible(df, pitcher_name)
    df_p = add_inning_and_ab(df_p)

    type_col = type_col_in_df(df_p)
    call_col = pick_col(df_p, "PitchCall", "Pitch Call", "Call", "PitchResult")

    if not type_col or type_col not in df_p.columns:
        return None, None, None

    df_p = df_p.sort_values(['AB #', 'Pitch # in AB']).reset_index(drop=True)
    df_p['_current_pitch'] = df_p[type_col].astype(str)
    df_p['_next_pitch'] = df_p.groupby('AB #')[type_col].shift(-1)

    sequences = df_p[df_p['_next_pitch'].notna()].copy()
    if sequences.empty:
        return None, None, None

    transition_counts = sequences.groupby(['_current_pitch', '_next_pitch']).size()
    transition_totals = sequences.groupby('_current_pitch').size()
    transition_pct = (transition_counts / transition_totals * 100).reset_index(name='Percentage')
    transition_matrix = transition_pct.pivot(index='_current_pitch', columns='_next_pitch', values='Percentage').fillna(0)

    effectiveness_data = []
    ev_col = pick_col(df_p, "ExitSpeed", "Exit Velo", "ExitVelocity", "Exit_Velocity", "ExitVel", "EV", "LaunchSpeed")
    play_result_col = pick_col(df_p, "PlayResult", "Play Result", "KorBB", "PAResult")

    for (curr, nxt), grp in sequences.groupby(['_current_pitch', '_next_pitch']):
        n = len(grp)
        if n < 3:
            continue

        if call_col:
            next_pitches = df_p[df_p.index.isin(grp.index + 1)]
            total_count = len(next_pitches)

            is_whiff = next_pitches[call_col].eq('StrikeSwinging')
            whiff_pct = (is_whiff.sum() / total_count * 100) if total_count > 0 else 0

            is_called_strike = next_pitches[call_col].eq('StrikeCalled')
            called_strike_pct = (is_called_strike.sum() / total_count * 100) if total_count > 0 else 0

            is_swing = next_pitches[call_col].isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
            is_strike = next_pitches[call_col].isin(['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
            strike_pct = (is_strike.sum() / total_count * 100) if total_count > 0 else 0

            is_inplay = next_pitches[call_col].eq('InPlay')
            inplay_pct = (is_inplay.sum() / total_count * 100) if total_count > 0 else 0

            weak_contact_pct = 0
            if ev_col and ev_col in next_pitches.columns:
                ev = pd.to_numeric(next_pitches[ev_col], errors="coerce")
                bip_count = is_inplay.sum()
                if bip_count > 0:
                    weak_contacts = ((ev < 80.0) & is_inplay & ev.notna()).sum()
                    weak_contact_pct = (weak_contacts / total_count) * 100

            hardhit_pct = 0
            if ev_col and ev_col in next_pitches.columns:
                ev = pd.to_numeric(next_pitches[ev_col], errors="coerce")
                hard_hits = ((ev >= 95.0) & is_inplay & ev.notna()).sum()
                hardhit_pct = (hard_hits / total_count) * 100

            hit_pct = 0
            if play_result_col and play_result_col in next_pitches.columns:
                is_hit = next_pitches[play_result_col].isin(['Single', 'Double', 'Triple', 'HomeRun'])
                hit_pct = (is_hit.sum() / total_count) * 100

            effectiveness = (
                (whiff_pct * 0.50) + (called_strike_pct * 0.35) + (weak_contact_pct * 0.15) -
                (hit_pct * 0.60) - (hardhit_pct * 0.40)
            )

            effectiveness_data.append({
                'Sequence': f"{curr} -> {nxt}", 'Count': n,
                'Strike%': round(strike_pct, 1), 'Whiff%': round(whiff_pct, 1),
                'Called Strike%': round(called_strike_pct, 1), 'Weak Contact%': round(weak_contact_pct, 1),
                'InPlay%': round(inplay_pct, 1), 'Hit%': round(hit_pct, 1),
                'HardHit%': round(hardhit_pct, 1), 'Effectiveness Score': round(effectiveness, 1),
                '_current': curr, '_next': nxt
            })

    effectiveness_df = pd.DataFrame(effectiveness_data).sort_values('Count', ascending=False)
    top_sequences = effectiveness_df.head(15)

    if len(top_sequences) == 0:
        return transition_matrix, effectiveness_df, None

    pitch_types = list(set(top_sequences['_current'].tolist() + top_sequences['_next'].tolist()))
    source_map = {p: i for i, p in enumerate(pitch_types)}
    target_map = {p: i + len(pitch_types) for i, p in enumerate(pitch_types)}

    sources, targets, values, colors_link = [], [], [], []
    for _, row in top_sequences.iterrows():
        sources.append(source_map[row['_current']])
        targets.append(target_map[row['_next']])
        values.append(row['Count'])
        eff = row['Effectiveness Score']
        if eff > 12:
            colors_link.append('rgba(0, 200, 0, 0.4)')
        elif eff >= 0:
            colors_link.append('rgba(255, 255, 0, 0.4)')
        else:
            colors_link.append('rgba(255, 0, 0, 0.4)')

    node_colors = []
    for p in pitch_types:
        color = get_pitch_color(p)
        node_colors.append(color)
        node_colors.append(color)

    all_labels = [f"{p} (from)" for p in pitch_types] + [f"{p} (to)" for p in pitch_types]

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="white", width=2),
                  label=[label.replace(" (from)", "").replace(" (to)", "") for label in all_labels],
                  color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=colors_link)
    )])
    fig.update_layout(
        title_text=f"Pitch Sequencing Flow: {canonicalize_person_name(pitcher_name)}",
        title_font=dict(size=16, color=DARK_GRAY, family="Arial Black"),
        font_size=11, height=600, margin=dict(l=20, r=20, t=60, b=20)
    )
    return transition_matrix, effectiveness_df, fig

def find_best_sequences(df: pd.DataFrame, pitcher_name: str, min_count: int = 5):
    _, effectiveness_df, _ = analyze_pitch_sequences(df, pitcher_name)
    if effectiveness_df is None or effectiveness_df.empty:
        return pd.DataFrame()
    best = effectiveness_df[effectiveness_df['Count'] >= min_count].copy()
    best = best.sort_values('Effectiveness Score', ascending=False)
    return best[['Sequence', 'Count', 'Strike%', 'Whiff%', 'Called Strike%',
                 'Weak Contact%', 'Hit%', 'HardHit%', 'Effectiveness Score']]

def analyze_sequence_by_count(df: pd.DataFrame, pitcher_name: str):
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

    usage = df_p.groupby([type_col, 'Count_Situation']).size().unstack(fill_value=0)
    usage_pct = usage.div(usage.sum(axis=0), axis=1) * 100
    usage_pct = usage_pct.round(1)
    usage_pct = usage_pct.map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    return usage_pct

# Part 5: Strike Zone, Data Loading, Sidebar, and Filters

# ─── Interactive strike zone (Plotly) ─────────────────────────────────────────
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
    if pa_df is None or pa_df.empty: return None

    type_col = pick_col(pa_df, type_col_in_df(pa_df), "Pitch Type","TaggedPitchType","PitchType")
    speed_col = pick_col(pa_df, "RelSpeed","Relspeed","ReleaseSpeed","RelSpeedMPH")
    ivb_col   = pick_col(pa_df, "InducedVertBreak","IVB","Induced Vert Break")
    hb_col    = pick_col(pa_df, "HorzBreak","HorizontalBreak","HB")
    exit_col  = pick_col(pa_df, "ExitSpeed","Exit Velo","ExitVelocity","EV")
    call_col  = pick_col(pa_df, "PitchCall","Pitch Call","Call")
    pno_col   = pick_col(pa_df, "Pitch # in AB","PitchofPA","PitchOfPA")
    x_col     = pick_col(pa_df, "PlateLocSide","Plate Loc Side","PlateSide","px","PlateLocX")
    y_col     = pick_col(pa_df, "PlateLocHeight","Plate Loc Height","PlateHeight","pz","PlateLocZ")

    xs = pd.to_numeric(pa_df.get(x_col, pd.Series(dtype=float)), errors="coerce")
    ys = pd.to_numeric(pa_df.get(y_col, pd.Series(dtype=float)), errors="coerce")
    if xs.isna().all() or ys.isna().all(): return None

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

    colors_pts = [get_pitch_color(t) for t in pa_df[type_col].astype(str).tolist()] if type_col and type_col in pa_df.columns else [HUSKER_RED] * len(pa_df)

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[str(int(n)) if pd.notna(n) else "" for n in cd[:,6]],
        textposition="top center",
        textfont=dict(size=10, color="black", family="Arial Black"),
        marker=dict(size=14, color=colors_pts, line=dict(width=2, color="white")),
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
        showlegend=False, name=""
    ), row=1, col=1)

    fig.update_xaxes(range=[x_min, x_max], showgrid=False, zeroline=False, showticklabels=False,
                    scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(range=[y_min, y_max], showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)

    x_range = x_max - x_min
    y_range = y_max - y_min
    aspect_ratio = x_range / y_range
    plot_height = 500
    plot_width = int(plot_height * aspect_ratio)

    fig.update_layout(
        height=plot_height, width=plot_width,
        title_text=(title or "Plate Appearance Strike Zone"), title_x=0.5,
        title_font=dict(size=14, color=DARK_GRAY, family="Arial Black"),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    return fig

def build_pitch_by_inning_pa_table(df: pd.DataFrame) -> pd.DataFrame:
    work = add_inning_and_ab(df)

    type_col   = type_col_in_df(work)
    result_col = pick_col(work, "PitchCall","Pitch Call","Call") or "PitchCall"
    velo_col   = pick_col(work, "RelSpeed","Relspeed","ReleaseSpeed")
    spin_col   = pick_col(work, "SpinRate","Spinrate","ReleaseSpinRate")
    ivb_col    = pick_col(work, "InducedVertBreak","IVB")
    hb_col     = pick_col(work, "HorzBreak","HB")
    x_col      = pick_col(work, "PlateLocSide","Plate Loc Side","PlateSide","px")
    y_col      = pick_col(work, "PlateLocHeight","Plate Loc Height","PlateHeight","pz")
    exit_velo_col = pick_col(work, "ExitSpeed","Exit Velo","ExitVelocity","EV","LaunchSpeed")
    batted_ball_type_col = pick_col(work, "AutoHitType","HitType","Batted Ball Type","BattedBallType")
    distance_col = pick_col(work, "Distance","Dist","HitDistance")

    batter_col = "Batter_AB" if "Batter_AB" in work.columns else find_batter_name_col(work)

    col_play_result = pick_col(work, "PlayResult","Result","Event","PAResult")
    col_korbb       = pick_col(work, "KorBB","K_BB","KBB")
    col_pitch_call  = result_col

    for c in [col_play_result, col_korbb, col_pitch_call]:
        if c and c in work.columns:
            work[c] = work[c].fillna("").astype(str)

    def _terminal_row_idx(g: pd.DataFrame) -> int:
        if col_play_result and g[col_play_result].str.strip().ne("").any():
            return g[g[col_play_result].str.strip().ne("")].index[-1]
        if col_korbb and g[col_korbb].str.strip().ne("").any():
            return g[g[col_korbb].str.strip().ne("")].index[-1]
        if "Pitch # in AB" in g.columns and g["Pitch # in AB"].notna().any():
            return g["Pitch # in AB"].astype("Int64").idxmax()
        return g.index[-1]

    def _pa_label(row) -> str:
        pr = row.get(col_play_result, "")
        base_result = ""

        if isinstance(pr, str) and pr.strip() and pr.strip().lower() != "undefined":
            base_result = pr.strip()
        else:
            kb = row.get(col_korbb, "")
            if isinstance(kb, str) and kb.strip():
                low = kb.strip().lower()
                if low in {"k", "so", "strikeout", "strike out"}: return "Strikeout"
                if low in {"bb", "walk", "ibb", "intentional walk"}: return "Walk"
                if low != "undefined": base_result = kb.strip()

        if not base_result:
            return "—"

        result_lower = base_result.lower()
        is_hit = any(h in result_lower for h in ["single", "double", "triple", "home run", "homerun"])
        is_out = result_lower == "out"

        if is_hit or is_out:
            details = []
            if batted_ball_type_col:
                bb_type = row.get(batted_ball_type_col, "")
                if isinstance(bb_type, str) and bb_type.strip():
                    bb_lower = bb_type.strip().lower()
                    if "ground" in bb_lower or bb_lower == "gb": details.append("Groundball")
                    elif "fly" in bb_lower or bb_lower == "fb": details.append("Flyball")
                    elif "line" in bb_lower or bb_lower == "ld": details.append("Line Drive")
                    elif "popup" in bb_lower or "pop" in bb_lower or bb_lower == "pu": details.append("Popup")
                    else: details.append(bb_type.strip())
            if exit_velo_col:
                ev = row.get(exit_velo_col, "")
                ev_num = pd.to_numeric(ev, errors="coerce")
                if pd.notna(ev_num): details.append(f"{ev_num:.1f} mph")
            if is_hit and distance_col:
                dist = row.get(distance_col, "")
                dist_num = pd.to_numeric(dist, errors="coerce")
                if pd.notna(dist_num): details.append(f"{dist_num:.0f} ft")
            if details: return f"{base_result} ({', '.join(details)})"

        return base_result

    idx_by_ab = work.groupby("AB #", sort=True, dropna=False).apply(_terminal_row_idx)
    pa_row = work.loc[idx_by_ab.values].copy()
    pa_row["PA Result"] = pa_row.apply(_pa_label, axis=1)
    work = work.merge(pa_row[["AB #","PA Result"]], on="AB #", how="left")

    ordered = ["Inning #","AB #","Pitch # in AB", batter_col, "PA Result",
               type_col, result_col, velo_col, spin_col, ivb_col, hb_col, x_col, y_col]
    present = [c for c in ordered if c and c in work.columns]
    tbl = work[present].copy()

    rename_map = {
        batter_col: "Batter", type_col: "Pitch Type", result_col: "Result",
        velo_col: "Velo", spin_col: "Spin Rate", ivb_col: "IVB", hb_col: "HB"
    }
    if x_col and x_col != "PlateLocSide": rename_map[x_col] = "PlateLocSide"
    if y_col and y_col != "PlateLocHeight": rename_map[y_col] = "PlateLocHeight"
    for k, v in list(rename_map.items()):
        if k and k in tbl.columns: tbl = tbl.rename(columns={k: v})

    for c in ["Velo","Spin Rate","IVB","HB"]:
        if c in tbl.columns: tbl[c] = pd.to_numeric(tbl[c], errors="coerce")
    if "Velo" in tbl:       tbl["Velo"] = tbl["Velo"].round(1)
    if "Spin Rate" in tbl:  tbl["Spin Rate"] = tbl["Spin Rate"].round(0)
    if "IVB" in tbl:        tbl["IVB"] = tbl["IVB"].round(1)
    if "HB" in tbl:         tbl["HB"] = tbl["HB"].round(1)

    return tbl

def style_pbp_expanders():
    st.markdown(f"""
        <style>
        .pbp-scope div[data-testid="stExpander"] > details > summary {{
            background-color: #FFFFFF !important; color: #2B2B2B !important; border-radius: 6px !important;
            padding: 10px 14px !important; font-weight: 600 !important; border: 1px solid #E0E0E0 !important;
        }}
        .pbp-scope div[data-testid="stExpander"] > details > summary:hover {{
            background-color: #F8F9FA !important; border-color: {HUSKER_RED} !important;
        }}
        </style>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_scrimmage_csv(_correction_version=3):
    if not os.path.exists(DATA_PATH_SCRIM):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH_SCRIM, low_memory=False)
    df = ensure_date_column(df)

    pitcher_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name")
    if pitcher_col:
        df["PitcherDisplay"] = df[pitcher_col].map(canonicalize_person_name)
    else:
        df["PitcherDisplay"] = "Unknown"

    type_col = type_col_in_df(df)
    if type_col and "PitcherDisplay" in df.columns:
        # Auden Pankonin: Convert all Fastballs to Sinkers
        pankonin_mask = df["PitcherDisplay"] == "Auden Pankonin"
        fastball_mask = df[type_col].astype(str).str.lower().str.contains('fastball', na=False)
        df.loc[pankonin_mask & fastball_mask, type_col] = "Sinker"

        # Kevin Mannel: Convert all Fastballs to Sinkers
        mannel_mask = df["PitcherDisplay"] == "Kevin Mannel"
        mannel_fastball_mask = df[type_col].astype(str).str.lower().str.contains('fastball', na=False)
        df.loc[mannel_mask & mannel_fastball_mask, type_col] = "Sinker"

    return df


@st.cache_data
def load_season_csv(_correction_version=1):
    """
    Load 2026 season master CSV (Season2026.csv).
    Each week: replace Season2026.csv with the updated file, then increment
    _correction_version by 1 to force Streamlit to reload the cache.
    """
    if not os.path.exists(DATA_PATH_SEASON):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH_SEASON, low_memory=False)
    df = ensure_date_column(df)

    pitcher_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name")
    if pitcher_col:
        df["PitcherDisplay"] = df[pitcher_col].map(canonicalize_person_name)
    else:
        df["PitcherDisplay"] = "Unknown"

    # ── Add season-specific pitch type corrections here ───────────────────────
    # Example:
    # type_col = type_col_in_df(df)
    # if type_col and "PitcherDisplay" in df.columns:
    #     mask = df["PitcherDisplay"] == "Some Pitcher"
    #     df.loc[mask & df[type_col].str.lower().str.contains('fastball'), type_col] = "Sinker"

    return df


# ── Load both datasets ────────────────────────────────────────────────────────
df_scrim  = load_scrimmage_csv()
df_season = load_season_csv()

_scrim_ok  = not df_scrim.empty
_season_ok = not df_season.empty

if not _scrim_ok and not _season_ok:
    st.error("No data files found. Add Scrimmage(28).csv and/or Season2026.csv to the app folder.")
    st.stop()

scrimmage_pitchers = set()
if _scrim_ok:
    scrimmage_pitchers = set(
        df_scrim[df_scrim.get('PitcherTeam', '') == 'NEB']['PitcherDisplay'].dropna().unique()
    )

season_pitchers = set()
if _season_ok:
    season_pitchers = set(
        df_season[df_season.get('PitcherTeam', '') == 'NEB']['PitcherDisplay'].dropna().unique()
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Data Source + Filters
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=150)

    st.markdown("### Data Source")

    # Build radio options from whichever files exist
    _source_options = []
    if _season_ok:
        _source_options.append("2026 Season")
    if _scrim_ok:
        _source_options.append("2025/26 Scrimmages")
    if _scrim_ok and _season_ok:
        _source_options.append("Both")

    data_source_choice = st.radio(
        "Select Dataset",
        options=_source_options,
        index=0,
        help="Season and Scrimmage data are kept separate. 'Both' merges them."
    )

    # ── Build df_all from selected source ────────────────────────────────────
    if data_source_choice == "2026 Season":
        df_all = df_season.copy()
        _data_label = "2026 Season"
    elif data_source_choice == "2025/26 Scrimmages":
        df_all = df_scrim.copy()
        _data_label = "Fall 2025/26 Scrimmages"
    else:  # Both
        df_all = pd.concat([df_scrim, df_season], ignore_index=True)
        _data_label = "2025/26 Scrimmages + 2026 Season"

    df_all = ensure_date_column(df_all)

    neb_pitchers = sorted(
        df_all[df_all.get('PitcherTeam', '') == 'NEB']['PitcherDisplay'].dropna().unique()
    )

    if len(neb_pitchers) == 0:
        st.warning("No Nebraska pitchers found for the selected data source.")
        st.stop()

    st.markdown("---")
    st.markdown("### Filters")
    st.caption(f"Data: {_data_label}")

    pitcher_choice = st.selectbox(
        "Select Pitcher",
        options=neb_pitchers,
        index=0,
        key="pitcher_choice"
    )

    batter_side_options = ["All", "RHH", "LHH"]
    batter_side_choice = st.selectbox(
        "Batter Handedness",
        options=batter_side_options,
        index=0
    )

    st.markdown("---")
    st.markdown("### Date Filters")

    df_for_date_filter = subset_by_pitcher_if_possible(df_all.copy(), pitcher_choice)

    if "Date" in df_for_date_filter.columns:
        pitcher_dates    = pd.to_datetime(df_for_date_filter["Date"], errors="coerce").dropna()
        unique_dates     = sorted(pitcher_dates.dt.date.unique())
        available_months = sorted(pitcher_dates.dt.month.unique())
        month_options    = [name for num, name in MONTH_CHOICES if num in available_months]
        available_days   = sorted(pitcher_dates.dt.day.unique())
    else:
        unique_dates = []; month_options = []; available_days = []

    game_choices = st.multiselect(
        "Select Game(s)",
        options=[format_date_long(d) for d in unique_dates],
        default=[],
        help="Select specific games by date"
    )

    date_lookup = {format_date_long(d): d for d in unique_dates}
    selected_game_dates = [date_lookup[g] for g in game_choices]

    month_choices = st.multiselect(
        "Month(s)",
        options=month_options,
        default=[],
        help="Only shows months where pitcher has data"
    )
    months_sel = [num for num, name in MONTH_CHOICES if name in month_choices]

    day_choices = st.multiselect(
        "Day(s) of Month",
        options=available_days,
        default=[],
        help="Only shows days where pitcher has data"
    )

    last_n_games = st.number_input(
        "Last N Games (0 = all)",
        min_value=0, max_value=50, value=0, step=1
    )

    st.markdown("---")
    st.caption("Nebraska Baseball Analytics")

# ═══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

df_pitcher_all = df_all.copy()
df_pitcher_all = subset_by_pitcher_if_possible(df_pitcher_all, pitcher_choice)

# Apply specific game date filter
if selected_game_dates and "Date" in df_pitcher_all.columns:
    df_pitcher_all = df_pitcher_all[
        pd.to_datetime(df_pitcher_all["Date"], errors="coerce").dt.date.isin(selected_game_dates)
    ].copy()

# Apply batter handedness filter
if batter_side_choice != "All":
    side_col = pick_col(df_pitcher_all, "BatterSide","Batter Side","Bats","Stand","BatSide")
    if side_col:
        side_norm = df_pitcher_all[side_col].astype(str).str.strip().str[0].str.upper()
        target = "R" if batter_side_choice == "RHH" else "L"
        df_pitcher_all = df_pitcher_all[side_norm == target].copy()

# Apply month/day/lastN filters
df_pitcher_all, season_label_display = apply_month_day_lastN(
    df_pitcher_all, months_sel, day_choices, last_n_games
)

# Build season label from selected games
if selected_game_dates:
    if len(selected_game_dates) == 1:
        season_label_display = format_date_long(selected_game_dates[0])
    else:
        season_label_display = f"{len(selected_game_dates)} selected games"
elif game_choices:
    season_label_display = "Selected games"

if df_pitcher_all.empty:
    st.warning("No data matches the current filter selection.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs(["Post-Game", "Overview", "Pitch Arsenal", "Performance", "Rankings"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: POST-GAME REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("Post-Game Report")
    st.caption("Select a specific game to view detailed pitch-by-pitch analysis")

    df_pitcher_games = subset_by_pitcher_if_possible(df_all.copy(), pitcher_choice)

    if "Date" in df_pitcher_games.columns:
        game_dates = pd.to_datetime(df_pitcher_games["Date"], errors="coerce").dropna()
        unique_game_dates = sorted(game_dates.dt.date.unique(), reverse=True)

        if len(unique_game_dates) == 0:
            st.warning("No games found for this pitcher.")
        else:
            game_date_formatted = [format_date_long(d) for d in unique_game_dates]
            selected_game_str = st.selectbox(
                "Select Game",
                options=game_date_formatted,
                index=0,
                key="postgame_game_selector"
            )

            date_lookup_pg = {format_date_long(d): d for d in unique_game_dates}
            selected_date = date_lookup_pg[selected_game_str]

            df_game = df_pitcher_games[
                pd.to_datetime(df_pitcher_games["Date"], errors="coerce").dt.date == selected_date
            ].copy()

            if df_game.empty:
                st.warning("No data found for selected game.")
            else:
                game_summary = make_outing_overall_summary(df_game)
                st.dataframe(themed_table(game_summary), use_container_width=True, hide_index=True)

                professional_divider()

                st.markdown("### Game Movement Profile")
                logo_img = load_logo_img()
                fig_game_movement, summary_game_movement = combined_pitcher_report(
                    df_game, pitcher_choice, logo_img, season_label=selected_game_str
                )

                if fig_game_movement:
                    show_and_close(fig_game_movement, use_container_width=True)
                else:
                    info_message("Movement profile not available for this game.")

                professional_divider()

                st.markdown("### Pitch-by-Pitch Breakdown")
                st.caption("Detailed pitch data organized by inning and at-bat")

                pbp_table = build_pitch_by_inning_pa_table(df_game)

                if not pbp_table.empty:
                    if "Inning #" in pbp_table.columns:
                        innings = pbp_table["Inning #"].dropna().unique()
                        innings = sorted([int(i) for i in innings if pd.notna(i)])

                        style_pbp_expanders()
                        st.markdown('<div class="pbp-scope">', unsafe_allow_html=True)

                        for inning in innings:
                            inning_data = pbp_table[pbp_table["Inning #"] == inning]
                            pitch_count = len(inning_data)

                            with st.expander(f"Inning {inning} ({pitch_count} pitches)", expanded=(inning == innings[0])):
                                display_cols = [c for c in inning_data.columns
                                               if c not in ["PA Result", "PlateLocSide", "PlateLocHeight"]]
                                st.dataframe(themed_table(inning_data[display_cols]), use_container_width=True)

                                if "AB #" in inning_data.columns:
                                    abs_in_inning = sorted(inning_data["AB #"].unique())

                                    for ab_num in abs_in_inning:
                                        ab_data = inning_data[inning_data["AB #"] == ab_num]
                                        batter_name = ab_data["Batter"].iloc[0] if "Batter" in ab_data.columns else "Unknown Batter"
                                        pa_result = ab_data["PA Result"].iloc[-1] if "PA Result" in ab_data.columns else "—"

                                        st.markdown(f"**AB #{ab_num}: {batter_name}** — Result: {pa_result}")

                                        fig_sz = pa_interactive_strikezone(
                                            ab_data,
                                            title=f"Inning {inning}, AB #{ab_num}: {batter_name}"
                                        )
                                        if fig_sz:
                                            st.plotly_chart(fig_sz, use_container_width=True)

                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        display_cols = [c for c in pbp_table.columns
                                       if c not in ["PA Result", "PlateLocSide", "PlateLocHeight"]]
                        st.dataframe(themed_table(pbp_table[display_cols]), use_container_width=True)
                else:
                    info_message("No pitch-by-pitch data available for this game.")
    else:
        st.warning("No date information available in the data.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("Pitcher Overview")
    st.caption(f"**{season_label_display}** • Dataset: {_data_label} • Batter: {batter_side_choice}")

    summary_table = make_outing_overall_summary(df_pitcher_all)
    st.dataframe(themed_table(summary_table), use_container_width=True)

    professional_divider()

    section_header("Outcomes Summary")
    outcome_table = make_pitcher_outcome_summary_table(df_pitcher_all)
    st.dataframe(themed_table(outcome_table), use_container_width=True)

    professional_divider()

    section_header("Overall Performance Metrics")
    perf_table = create_overall_performance_table(df_pitcher_all)
    if not perf_table.empty:
        def highlight_overall(row):
            if row['Pitch Type'] == 'OVERALL':
                return ['background-color: rgba(230, 0, 38, 0.1); font-weight: 600'] * len(row)
            return [''] * len(row)

        styled_perf = perf_table.style.apply(highlight_overall, axis=1).hide(axis="index").format({
            'Usage%': '{:.1f}', 'Strike%': '{:.1f}', 'Whiff%': '{:.1f}',
            'Zone%': '{:.1f}', 'Chase%': '{:.1f}', 'Zone Contact%': '{:.1f}',
            'InPlay%': '{:.1f}', 'HardHit%': '{:.1f}', 'Avg EV': '{:.1f}'
        }, na_rep="—")
        st.dataframe(styled_perf, use_container_width=True)
    else:
        info_message("Performance metrics not available.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: PITCH ARSENAL
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("Pitch Arsenal")
    st.caption(f"Movement profile and usage patterns • {_data_label}")

    logo_img = load_logo_img()
    fig_movement, summary_movement = combined_pitcher_report(
        df_pitcher_all, pitcher_choice, logo_img, season_label=season_label_display
    )

    if fig_movement:
        show_and_close(fig_movement, use_container_width=True)
    else:
        st.info("Movement profile not available for current selection.")

    professional_divider()

    section_header("Pitch Location Patterns")
    st.caption("Heat maps showing where each pitch type is located")

    type_col = type_col_in_df(df_pitcher_all)
    all_pitch_types = []
    if type_col and type_col in df_pitcher_all.columns:
        all_pitch_types = sorted(df_pitcher_all[type_col].dropna().unique().tolist())

    show_all_pitches = st.checkbox(
        "Show all pitch types",
        value=False,
        key="show_all_pitch_locations",
        help="By default, shows top 3 most-used pitches."
    )

    if show_all_pitches and len(all_pitch_types) > 0:
        pitches_to_display = all_pitch_types
        fig_pitch_locations = create_pitch_type_location_heatmaps(
            df_pitcher_all, pitcher_choice, pitch_types_to_show=pitches_to_display
        )
    else:
        fig_pitch_locations = create_pitch_type_location_heatmaps(
            df_pitcher_all, pitcher_choice, pitch_types_to_show=None, show_top_n=3
        )

    if fig_pitch_locations:
        show_and_close(fig_pitch_locations, use_container_width=True)
    else:
        info_message("Pitch location heatmaps not available. Requires location data.")

    professional_divider()

    section_header("Miss Locations (Called Balls)")
    st.caption("Heat maps showing where pitches miss the zone")

    if show_all_pitches and len(all_pitch_types) > 0:
        fig_miss_locations = create_miss_location_heatmaps(
            df_pitcher_all, pitcher_choice, pitch_types_to_show=pitches_to_display
        )
    else:
        fig_miss_locations = create_miss_location_heatmaps(
            df_pitcher_all, pitcher_choice, pitch_types_to_show=None, show_top_n=3
        )

    if fig_miss_locations:
        show_and_close(fig_miss_locations, use_container_width=True)
    else:
        info_message("Miss location heatmaps not available. Requires location and pitch call data.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("Performance Analysis")
    st.caption(f"Effectiveness metrics and situational performance • {_data_label}")

    st.markdown("### Count Situation Analysis")

    type_col = type_col_in_df(df_pitcher_all)
    available_pitch_types = ["Overall"]
    if type_col and type_col in df_pitcher_all.columns:
        available_pitch_types += sorted(df_pitcher_all[type_col].dropna().unique().tolist())

    count_pitch_filter = st.selectbox(
        "Filter by Pitch Type",
        options=available_pitch_types,
        index=0,
        key="count_pitch_filter",
        help="Filter count analysis by specific pitch type or view overall performance"
    )

    st.markdown("---")

    st.markdown("#### Location by Count Situation")
    filter_display = f" - {count_pitch_filter}" if count_pitch_filter != "Overall" else ""
    st.caption(f"Pitch location density by count situation{filter_display}")

    fig_heatmaps = create_count_leverage_heatmaps(df_pitcher_all, pitcher_choice, pitch_type_filter=count_pitch_filter)
    if fig_heatmaps:
        show_and_close(fig_heatmaps, use_container_width=True)
    else:
        info_message("Count leveraging heatmaps not available. Requires count and location data.")

    st.markdown("---")

    st.markdown("#### Performance by Count Situation")
    if count_pitch_filter != "Overall":
        st.caption(f"Performance metrics for {count_pitch_filter} across different count situations")
    else:
        st.caption("Performance metrics across different count situations")

    count_performance = create_count_situation_comparison(df_pitcher_all, pitch_type_filter=count_pitch_filter)
    if not count_performance.empty:
        st.dataframe(themed_table(count_performance), use_container_width=True, hide_index=True)
    else:
        info_message("Count situation performance data not available.")

    st.markdown("---")

    st.markdown("#### Pitch Usage by Count Situation")
    st.caption("Percentage of each pitch type used in different count situations")

    seq_by_count = analyze_sequence_by_count(df_pitcher_all, pitcher_choice)
    if not seq_by_count.empty:
        st.dataframe(seq_by_count, use_container_width=True)
    else:
        st.info("Count situation data not available.")

    professional_divider()

    section_header("Pitch Sequencing Analysis")

    trans_matrix, effectiveness, sankey_fig = analyze_pitch_sequences(df_pitcher_all, pitcher_choice)

    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)

        st.markdown("#### Most Effective Sequences")
        st.caption("Effectiveness = (Whiff%×0.50) + (Called Strike%×0.35) + (Weak Contact%×0.15) - (Hit%×0.60) - (HardHit%×0.40) | Green > 12 | Yellow 0-12 | Red < 0")
        best_sequences = find_best_sequences(df_pitcher_all, pitcher_choice, min_count=5)

        if not best_sequences.empty:
            def style_effectiveness(val):
                try:
                    v = float(val)
                    if v > 12: return 'background-color: rgba(0, 200, 0, 0.3)'
                    elif v < 0: return 'background-color: rgba(255, 0, 0, 0.3)'
                    else: return 'background-color: rgba(255, 255, 0, 0.3)'
                except: pass
                return ''

            styled = best_sequences.style.applymap(
                style_effectiveness, subset=['Effectiveness Score']
            ).hide(axis="index").format({
                'Strike%': '{:.1f}', 'Whiff%': '{:.1f}', 'Called Strike%': '{:.1f}',
                'Weak Contact%': '{:.1f}', 'Hit%': '{:.1f}', 'HardHit%': '{:.1f}',
                'Effectiveness Score': '{:.1f}'
            }, na_rep="—")
            st.dataframe(styled, use_container_width=True)
        else:
            info_message("Insufficient data for sequence effectiveness analysis (minimum 5 occurrences required).")
    else:
        info_message("Pitch sequencing requires multiple pitches per at-bat. Not enough sequence data available.")

    professional_divider()

    st.markdown("### Outcome Heatmaps")
    st.caption("Location patterns for different pitch outcomes")

    available_pitch_types_outcomes = ["Overall"]
    if type_col and type_col in df_pitcher_all.columns:
        available_pitch_types_outcomes += sorted(df_pitcher_all[type_col].dropna().unique().tolist())

    outcome_pitch_filter = st.selectbox(
        "Filter by Pitch Type",
        options=available_pitch_types_outcomes,
        index=0,
        key="outcome_pitch_filter",
        help="Filter outcome heatmaps by specific pitch type or view overall outcomes"
    )

    fig_outcomes = create_outcome_heatmaps(df_pitcher_all, pitcher_choice, pitch_type_filter=outcome_pitch_filter)
    if fig_outcomes:
        show_and_close(fig_outcomes, use_container_width=True)
    else:
        info_message("Outcome heatmaps not available. Requires location and result data.")

    professional_divider()

    section_header("Spray Chart: Hits Allowed")
    st.caption("Note: Spray chart shows hits from opponent batter's perspective")

    fig_spray, summary_spray = create_spray_chart(df_pitcher_all, pitcher_choice, season_label_display)
    if fig_spray:
        show_and_close(fig_spray, use_container_width=True)
        if summary_spray is not None and not summary_spray.empty:
            st.markdown("#### Batted Ball Summary")
            st.dataframe(themed_table(summary_spray), use_container_width=True, hide_index=True)
    else:
        info_message("Spray chart not available. Requires batted ball location data (Bearing/Distance).")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section_header("Team Rankings")
    st.caption(_data_label)

    st.markdown("### Filter Options")

    df_rank_base = df_all.copy()

    type_col_rank = type_col_in_df(df_rank_base)
    available_types = sorted(df_rank_base[type_col_rank].dropna().unique()) if type_col_rank in df_rank_base.columns else []

    pitch_types_filter = st.multiselect(
        "Pitch Type(s)",
        options=available_types,
        default=[],
        help="Filter rankings by specific pitch type(s)"
    )

    professional_divider()

    rankings_df = make_pitcher_rankings(
        df_rank_base,
        pitch_types_filter if pitch_types_filter else None
    )

    if not rankings_df.empty:
        st.markdown("### Team Averages")
        team_avg = make_team_averages(
            df_rank_base,
            pitch_types_filter if pitch_types_filter else None
        )
        st.dataframe(themed_table(team_avg), use_container_width=True, hide_index=True)

        professional_divider()

        st.markdown("### Individual Pitcher Rankings")
        st.caption("D1 Average shown in gray for comparison")

        D1_AVERAGES = {
            'WHIP': 1.64, 'H9': 9.90, 'Strike%': 60.7, 'Whiff%': 23.0,
            'Zone%': 45.5, 'Zwhiff%': 15.7, 'Chase%': 24.3, 'HH%': 36.0,
            'Barrel%': 17.3, 'BB%': 11.3, 'SO%': 19.3,
        }

        rankings_display = rankings_df.sort_values("WHIP", ascending=True, na_position="last")
        rankings_display = rankings_display.drop(columns=["_IP_num"])

        if pitch_types_filter:
            cols_to_drop = [c for c in ["App", "IP"] if c in rankings_display.columns]
            if cols_to_drop:
                rankings_display = rankings_display.drop(columns=cols_to_drop)

        d1_row = pd.DataFrame([{
            'Pitcher': 'D1 Average',
            'App': '—' if 'App' in rankings_display.columns else None,
            'IP': '—' if 'IP' in rankings_display.columns else None,
            'H': '—', 'HR': '—', 'BB': '—', 'HBP': '—', 'SO': '—',
            'WHIP': D1_AVERAGES.get('WHIP', np.nan),
            'H9': D1_AVERAGES.get('H9', np.nan),
            'BB%': D1_AVERAGES.get('BB%', np.nan),
            'SO%': D1_AVERAGES.get('SO%', np.nan),
            'Strike%': D1_AVERAGES.get('Strike%', np.nan),
            'HH%': D1_AVERAGES.get('HH%', np.nan),
            'Barrel%': D1_AVERAGES.get('Barrel%', np.nan),
            'Zone%': D1_AVERAGES.get('Zone%', np.nan),
            'Zwhiff%': D1_AVERAGES.get('Zwhiff%', np.nan),
            'Chase%': D1_AVERAGES.get('Chase%', np.nan),
            'Whiff%': D1_AVERAGES.get('Whiff%', np.nan),
        }])

        d1_row = d1_row[[c for c in d1_row.columns if c in rankings_display.columns]]
        rankings_display = pd.concat([d1_row, rankings_display], ignore_index=True)

        styled_rankings = rankings_display.style.hide(axis="index")

        def highlight_d1_row(row):
            if row['Pitcher'] == 'D1 Average':
                return ['background-color: rgba(200, 200, 200, 0.3); font-weight: 600'] * len(row)
            return [''] * len(row)

        styled_rankings = styled_rankings.apply(highlight_d1_row, axis=1)

        format_dict = {}
        for col in rankings_display.columns:
            if col in ['WHIP']: format_dict[col] = '{:.2f}'
            elif col in ['H9']: format_dict[col] = '{:.1f}'
            elif col.endswith('%'): format_dict[col] = '{:.1f}'

        styled_rankings = styled_rankings.format(format_dict, na_rep="—")
        st.dataframe(styled_rankings, use_container_width=True)

        csv = rankings_display.to_csv(index=False)
        st.download_button(
            label="Download Rankings CSV",
            data=csv,
            file_name=f"nebraska_pitcher_rankings_{data_source_choice.replace(' ', '_').replace('/', '-')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No ranking data available for the selected filters.")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; padding: 20px; font-size: 14px;'>
        <strong>Nebraska Baseball Pitcher Analytics Platform</strong><br>
        Built with Streamlit • Data updated through {date.today().strftime('%B %d, %Y')}<br>
        Active dataset: <strong>{_data_label}</strong><br>
        <span style='color: {HUSKER_RED}; font-weight: 600;'>Go Big Red!</span>
    </div>
    """,
    unsafe_allow_html=True
)
