# pitcher_app.py — COMPLETE VERSION WITH ALL FEATURES
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
            text-shadow: 0 4px 12px rgba(0,0,0,0.6); margin: 0; text-transform: uppercase;
        }}
        .hero-sub {{ 
            font-size: 20px; font-weight: 500; opacity: .95; margin-top: 8px;
            text-shadow: 0 2px 8px rgba(0,0,0,0.5); letter-spacing: 0.5px;
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
    return out

def type_col_in_df(df: pd.DataFrame) -> str:
    seg = st.session_state.get("segment_choice", "")
    if "Scrimmages" in str(seg):
        return (pick_col(df, "TaggedPitchType","Tagged Pitch Type","PitchType") or "TaggedPitchType")
    return (pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType") or "AutoPitchType")

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
    return pick_col(df, "PitchofPA","PitchOfPA","Pitch_of_PA","Pitch of PA","Pitch # in AB")

def find_inning_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "Inning","inning","InningNumber","Inning #")

def _to_num(s): return pd.to_numeric(s, errors="coerce")

def _normalize_inning_series(series: pd.Series) -> pd.Series:
    txt = series.astype(str); num = txt.str.extract(r'(\d+)')[0]
    return pd.to_numeric(num, errors="coerce").astype(pd.Int64Dtype())

def add_inning_and_ab(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    inn_c = find_inning_col(out)
    po_c = find_pitch_of_pa_col(out)
    
    out["Inning #"] = _normalize_inning_series(out[inn_c]) if inn_c else pd.Series([pd.NA]*len(out), dtype="Int64")
    
    if po_c is None:
        out["AB #"] = 1
        out["Pitch # in AB"] = np.arange(1, len(out) + 1)
    else:
        is_start = (_to_num(out[po_c]) == 1)
        ab_id = is_start.cumsum()
        if (ab_id == 0).any(): ab_id = ab_id.replace(0, np.nan).ffill().fillna(1)
        out["AB #"] = ab_id.astype(int)
        out["Pitch # in AB"] = _to_num(out[po_c]).astype(pd.Int64Dtype())
    
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

# End of Part 1# Part 2 of 6: Helper Functions, Outcome Summaries, and Rankings

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
            "HardHits": 0, "HardHit% (BIP)": np.nan
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
        "HardHits": hard_cnt, "HardHit% (BIP)": round(float(hard_pct_bip), 1) if pd.notna(hard_pct_bip) else np.nan,
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

# End of Part 2# Part 3 of 6: Rankings and New Advanced Features (Performance & Count Leveraging)

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

# ═══════════════════════════════════════════════════════════════════════════════
# NEW ADVANCED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── NEW: Overall Performance Metrics ─────────────────────────────────────────
def create_overall_performance_table(df: pd.DataFrame):
    """Creates overall performance metrics by pitch type AND overall."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    type_col = type_col_in_df(df)
    call_col = pick_col(df, "PitchCall", "Pitch Call", "Call", "PitchResult")
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")
    ev_col = pick_col(df, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV", "LaunchSpeed")
    
    if not call_col:
        return pd.DataFrame()
    
    def calc_metrics(subset):
        n = len(subset)
        if n == 0:
            return {}
        
        row = {'Pitches': n}
        
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
    
    cols = ['Pitch Type', 'Pitches', 'Strike%', 'Whiff%', 'Zone%', 'Chase%', 
            'Zone Contact%', 'InPlay%', 'HardHit%', 'Avg EV']
    cols = [c for c in cols if c in result.columns]
    result = result[cols]
    
    result = result.sort_values('Pitches', ascending=False)
    overall_row = result[result['Pitch Type'] == 'OVERALL']
    other_rows = result[result['Pitch Type'] != 'OVERALL']
    result = pd.concat([other_rows, overall_row], ignore_index=True)
    
    return result

# ─── NEW: Count Leveraging Analysis ───────────────────────────────────────────
def create_count_leverage_heatmap(df: pd.DataFrame, metric: str = "Strike%"):
    """Creates a heatmap showing pitcher effectiveness by count (balls-strikes)."""
    if df is None or df.empty:
        return None, pd.DataFrame()
    
    balls_col = pick_col(df, "Balls", "Ball Count", "BallCount", "balls")
    strikes_col = pick_col(df, "Strikes", "Strike Count", "StrikeCount", "strikes")
    call_col = pick_col(df, "PitchCall", "Pitch Call", "Call", "PitchResult")
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")
    ev_col = pick_col(df, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV", "LaunchSpeed")
    
    if not balls_col or not strikes_col:
        return None, pd.DataFrame()
    
    balls = pd.to_numeric(df[balls_col], errors="coerce")
    strikes = pd.to_numeric(df[strikes_col], errors="coerce")
    
    valid = balls.between(0, 3) & strikes.between(0, 2)
    work = df[valid].copy()
    work["Balls"] = balls[valid].astype(int)
    work["Strikes"] = strikes[valid].astype(int)
    
    if work.empty:
        return None, pd.DataFrame()
    
    matrix = np.full((4, 3), np.nan)
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
    
    # Create figure with single heatmap (larger)
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(['0', '1', '2'], fontsize=12, fontweight='600')
    ax.set_yticklabels(['0', '1', '2', '3'], fontsize=12, fontweight='600')
    
    ax.set_xlabel('Strikes', fontsize=14, fontweight='700')
    ax.set_ylabel('Balls', fontsize=14, fontweight='700')
    ax.set_title(f'{metric} by Count', fontsize=16, fontweight='bold', pad=20)
    
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
                   color=color, fontsize=11, fontweight='600')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, rotation=270, labelpad=25, fontsize=13, fontweight='700')
    cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
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
    """Compare pitcher effectiveness across count situations."""
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
    
    return pd.DataFrame(rows).sort_values('Pitches', ascending=False)

# Part 4 of 6: Spray Charts, Pitch Sequencing, and Count Heatmaps (Hitter App Style)

# ─── DENSITY COMPUTATION ──────────────────────────────────────────────────────
def compute_density_pitcher(x, y, xi_m, yi_m):
    """Compute density for pitcher location heatmaps"""
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

# ─── SPRAY CHART HELPER FUNCTIONS ─────────────────────────────────────────────
def draw_dirt_diamond(
    ax,
    origin=(0.0, 0.0),
    size: float = 80,
    base_size: float = 8,
    grass_scale: float = 0.4,
    custom_wall_distances: list = None
):
    """Draw a baseball diamond matching the hitter app style"""
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
        ax.add_patch(Polygon(outfield_points, closed=True, facecolor='#228B22', 
                            edgecolor='black', linewidth=2))
        
        outfield_radius = max(distances)
    else:
        outfield_radius = size * 1.7
        ax.add_patch(Wedge(home, outfield_radius, 45, 135, facecolor='#228B22', 
                          edgecolor='black', linewidth=2))
    
    # Dirt infield
    ax.add_patch(Wedge(home, size, 45, 135, facecolor='#ED8B00', 
                      edgecolor='black', linewidth=2))
    
    # Grass cutout
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
    
    # Bases
    for pos in [gfirst, gsecond, gthird]:
        ax.add_patch(Rectangle((pos[0] - base_size/2, pos[1] - base_size/2), 
                              base_size, base_size,
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
    for angle in [45, 135]:
        rad = math.radians(angle)
        end = home + np.array([outfield_radius * 1.1 * math.cos(rad),
                               outfield_radius * 1.1 * math.sin(rad)])
        ax.plot([home[0], end[0]], [home[1], end[1]], color='white', linewidth=2)

    ax.set_xlim(-outfield_radius, outfield_radius)
    ax.set_ylim(-base_size * 1.5, outfield_radius)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax

def bearing_distance_to_xy(bearing, distance):
    """Convert bearing and distance to x,y coordinates"""
    angle_rad = np.radians(90 - bearing)
    x = distance * np.cos(angle_rad)
    y = distance * np.sin(angle_rad)
    return x, y

# ─── NEW: Count Leveraging Heatmaps (3-panel matching hitter app) ────────────
def create_count_leverage_heatmaps(df: pd.DataFrame, pitcher_name: str):
    """
    Creates 3-panel heatmaps showing pitcher performance in different count situations.
    Matches the visual style of the hitter app heatmaps.
    """
    if df is None or df.empty:
        return None
    
    balls_col = pick_col(df, "Balls", "Ball Count", "BallCount", "balls")
    strikes_col = pick_col(df, "Strikes", "Strike Count", "StrikeCount", "strikes")
    call_col = pick_col(df, "PitchCall", "Pitch Call", "Call", "PitchResult")
    x_col = pick_col(df, "PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX")
    y_col = pick_col(df, "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ")
    
    if not balls_col or not strikes_col or not x_col or not y_col:
        return None
    
    balls = pd.to_numeric(df[balls_col], errors="coerce")
    strikes = pd.to_numeric(df[strikes_col], errors="coerce")
    
    work = df.copy()
    work["Balls"] = balls
    work["Strikes"] = strikes
    
    # Define count situations
    ahead_mask = strikes > balls  # Pitcher ahead
    behind_mask = balls > strikes  # Hitter ahead
    two_strike_mask = strikes == 2
    
    # Get location data
    xs = pd.to_numeric(work[x_col], errors="coerce")
    ys = pd.to_numeric(work[y_col], errors="coerce")
    
    # Create figure with 3 panels (matching hitter app layout)
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.25)
    
    def _panel(ax, title, mask):
        """Draw a single heatmap panel"""
        # Draw strike zone
        l, b, w, h = get_zone_bounds()
        ax.add_patch(Rectangle((l, b), w, h, fill=False, linewidth=2, color='black'))
        
        # Draw thirds
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
            # Plot individual pitches if too few
            ax.plot(subset_x, subset_y, 'o', color='deepskyblue', alpha=0.8, markersize=6)
        else:
            # Create density heatmap
            xi = np.linspace(-3, 3, 200)
            yi = np.linspace(0, 5, 200)
            xi_m, yi_m = np.meshgrid(xi, yi)
            zi = compute_density_pitcher(subset_x, subset_y, xi_m, yi_m)
            
            ax.imshow(zi, origin='lower', extent=[-3, 3, 0, 5], 
                     aspect='equal', cmap=custom_cmap, alpha=0.8)
            
            # Redraw strike zone on top
            ax.add_patch(Rectangle((l, b), w, h, fill=False, linewidth=2, color='black'))
            for i in (1, 2):
                ax.add_line(Line2D([l+i*dx]*2, [b, b+h], linestyle='--', color='gray', linewidth=1))
                ax.add_line(Line2D([l, l+w], [b+i*dy]*2, linestyle='--', color='gray', linewidth=1))
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal', 'box')
        ax.set_title(title, fontsize=12, pad=8, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add pitch count to title
        n_pitches = mask.sum()
        ax.text(0.5, 0.02, f"n = {n_pitches}", transform=ax.transAxes,
               ha='center', va='bottom', fontsize=10, style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Panel 1: Pitcher Ahead
    ax1 = fig.add_subplot(gs[0, 0])
    _panel(ax1, "Pitcher Ahead in Count", ahead_mask & balls.notna() & strikes.notna())
    
    # Panel 2: Hitter Ahead  
    ax2 = fig.add_subplot(gs[0, 1])
    _panel(ax2, "Hitter Ahead in Count", behind_mask & balls.notna() & strikes.notna())
    
    # Panel 3: Two Strikes
    ax3 = fig.add_subplot(gs[0, 2])
    _panel(ax3, "Two Strike Counts", two_strike_mask & balls.notna() & strikes.notna())
    
    plt.tight_layout()
    return fig

def create_spray_chart(df: pd.DataFrame, pitcher_name: str, season_label: str = "Season"):
    """
    Creates a spray chart showing hits allowed (from batter's perspective).
    Matches the visual style of the hitter app spray charts.
    """
    df_p = subset_by_pitcher_if_possible(df, pitcher_name)
    
    bearing_col = pick_col(df_p, "Bearing", "HitBearing", "Hit Bearing", "Direction", "Angle")
    distance_col = pick_col(df_p, "Distance", "HitDistance", "Hit Distance", "Dist")
    result_col = pick_col(df_p, "PlayResult", "Result", "Event", "PAResult")
    type_col = pick_col(df_p, "HitType", "Hit Type", "BattedBallType", "BBType", "TaggedHitType")
    ev_col = pick_col(df_p, "ExitSpeed", "Exit Velo", "ExitVelocity", "EV")
    call_col = pick_col(df_p, "PitchCall", "Pitch Call", "Call")
    
    if not bearing_col or not distance_col:
        return None, pd.DataFrame()
    
    # Get balls in play
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
    
    # Convert to x,y coordinates
    coords = [bearing_distance_to_xy(b, d) for b, d in zip(bearing, distance)]
    bip['x'] = [c[0] for c in coords]
    bip['y'] = [c[1] for c in coords]
    
    # Create figure - SMALLER SIZE
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define wall distances (matching hitter app)
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
    
    # Draw field - SMALLER SIZE (reduced from 100 to 70)
    draw_dirt_diamond(ax, origin=(0.0, 0.0), size=70, custom_wall_distances=wall_data)
    
    # Draw outfield wall - THINNER LINE
    wall_x = [dist * np.cos(np.radians(ang)) for ang, dist in wall_data]
    wall_y = [dist * np.sin(np.radians(ang)) for ang, dist in wall_data]
    ax.plot(wall_x, wall_y, 'k-', linewidth=2, zorder=10)  # Changed from 3 to 2
    
    # Add distance markers - SMALLER TEXT
    for angle, dist, label in [(45, 335, '335'), (90, 395, '395'), (135, 325, '325')]:
        rad = np.radians(angle)
        x = dist * np.cos(rad)
        y = dist * np.sin(rad)
        ax.text(x, y, label, ha='center', va='center', fontsize=9,  # Changed from 11 to 9
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',  # Changed from 0.4 to 0.3
                facecolor='yellow', edgecolor='black', linewidth=1.5, alpha=0.9), zorder=11)  # Changed from 2 to 1.5
    
    # Categorize hit types and determine colors
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
    
    bip['HitCategory'] = bip.get(type_col, pd.Series(dtype=object)).apply(categorize_hit_type)
    
    # Color scheme matching hitter app
    hit_type_colors = {
        'GroundBall': '#DC143C',  # Crimson red
        'LineDrive': '#FFD700',   # Gold
        'FlyBall': '#1E90FF',     # Dodger blue
        'Popup': '#FF69B4',       # Hot pink
        'Other': '#A9A9A9'        # Dark gray
    }
    
    # Plot each batted ball - SMALLER MARKERS
    for idx, row in bip.iterrows():
        hit_cat = row['HitCategory']
        play_result = str(row.get(result_col, ''))
        
        marker_size = 80  # Changed from 120 to 80
        
        # Thicker edge for hits
        if play_result in ['Single', 'Double', 'Triple', 'HomeRun']:
            edgecolor = 'black'
            linewidth = 1.5  # Changed from 2 to 1.5
        else:
            edgecolor = 'black'
            linewidth = 0.8  # Changed from 1 to 0.8
        
        ax.scatter(row['x'], row['y'], 
                  c=hit_type_colors.get(hit_cat, '#A9A9A9'), 
                  s=marker_size, 
                  marker='o',
                  edgecolors=edgecolor, 
                  linewidths=linewidth,
                  alpha=0.85,
                  zorder=20)
    
    # Create legend - SMALLER
    legend_elements = []
    
    for hit_type in ['GroundBall', 'LineDrive', 'FlyBall', 'Popup']:
        count = (bip['HitCategory'] == hit_type).sum()
        if count > 0:
            label = hit_type.replace('GroundBall', 'Ground Ball').replace('LineDrive', 'Line Drive').replace('FlyBall', 'Fly Ball')
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=hit_type_colors[hit_type], 
                       markersize=7,  # Changed from 10 to 7
                       markeredgecolor='black', 
                       markeredgewidth=1,  # Changed from 1.5 to 1
                       label=f'{label} ({count})')
            )
    
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=7, markeredgecolor='black', markeredgewidth=1.5,  # Changed from 10/2 to 7/1.5
               label='Hit (thick edge)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=7, markeredgecolor='black', markeredgewidth=0.8,  # Changed from 10/1 to 7/0.8
               label='Out (thin edge)')
    ])
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(0.02, 0.98), frameon=True, 
             fancybox=True, shadow=True, fontsize=8)  # Changed from 10 to 8
    
    # Set axis limits
    max_dist = max(bip['Distance'].max(), 400)
    ax.set_xlim(-max_dist * 0.85, max_dist * 0.85)
    ax.set_ylim(-30, max_dist * 1.1)
    ax.set_aspect('equal')
    
    # SMALLER TITLE
    ax.set_title(f"{canonicalize_person_name(pitcher_name)} — Hits Allowed (Batter's View)\n{season_label}", 
                fontsize=13, fontweight='bold', pad=15)  # Changed from 16 to 13, pad from 20 to 15
    
    plt.tight_layout()
    
    # Create summary table
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

# ─── NEW: Pitch Sequencing Analysis ───────────────────────────────────────────
def analyze_pitch_sequences(df: pd.DataFrame, pitcher_name: str):
    """Analyzes pitch-to-pitch sequences and their effectiveness."""
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
    
    transition_matrix = transition_pct.pivot(index='_current_pitch', 
                                            columns='_next_pitch', 
                                            values='Percentage').fillna(0)
    
    effectiveness_data = []
    
    for (curr, nxt), grp in sequences.groupby(['_current_pitch', '_next_pitch']):
        n = len(grp)
        if n < 3:
            continue
        
        if call_col:
            next_pitches = df_p[df_p.index.isin(grp.index + 1)]
            
            is_strike = next_pitches[call_col].isin(['StrikeCalled','StrikeSwinging',
                                                     'FoulBallNotFieldable','FoulBallFieldable','InPlay'])
            strike_pct = is_strike.mean() * 100
            
            is_swing = next_pitches[call_col].isin(['StrikeSwinging','FoulBallNotFieldable',
                                                    'FoulBallFieldable','InPlay'])
            is_whiff = next_pitches[call_col].eq('StrikeSwinging')
            swings = is_swing.sum()
            whiff_pct = (is_whiff.sum() / swings * 100) if swings > 0 else 0
            
            is_inplay = next_pitches[call_col].eq('InPlay')
            inplay_pct = is_inplay.mean() * 100
            
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
    
    top_sequences = effectiveness_df.head(15)
    
    if len(top_sequences) == 0:
        return transition_matrix, effectiveness_df, None
    
    pitch_types = list(set(top_sequences['_current'].tolist() + top_sequences['_next'].tolist()))
    
    source_labels = [f"{p} (from)" for p in pitch_types]
    target_labels = [f"{p} (to)" for p in pitch_types]
    all_labels = source_labels + target_labels
    
    source_map = {p: i for i, p in enumerate(pitch_types)}
    target_map = {p: i + len(pitch_types) for i, p in enumerate(pitch_types)}
    
    sources = []
    targets = []
    values = []
    colors_link = []
    
    for _, row in top_sequences.iterrows():
        sources.append(source_map[row['_current']])
        targets.append(target_map[row['_next']])
        values.append(row['Count'])
        
        eff = row['Effectiveness Score']
        if eff > 60:
            color = 'rgba(0, 200, 0, 0.4)'
        elif eff > 40:
            color = 'rgba(255, 255, 0, 0.4)'
        else:
            color = 'rgba(255, 0, 0, 0.4)'
        colors_link.append(color)
    
    node_colors = []
    for p in pitch_types:
        color = get_pitch_color(p)
        node_colors.append(color)
        node_colors.append(color)
    
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
            color=colors_link
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
    """Returns the most effective pitch sequences."""
    _, effectiveness_df, _ = analyze_pitch_sequences(df, pitcher_name)
    
    if effectiveness_df is None or effectiveness_df.empty:
        return pd.DataFrame()
    
    best = effectiveness_df[effectiveness_df['Count'] >= min_count].copy()
    best = best.sort_values('Effectiveness Score', ascending=False)
    
    return best[['Sequence', 'Count', 'Strike%', 'Whiff%', 'InPlay%', 'Effectiveness Score']]

def analyze_sequence_by_count(df: pd.DataFrame, pitcher_name: str):
    """Show pitch usage % by count situation."""
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
    
    return usage_pct.round(1)

# End of Part 4# Part 5 of 6: Data Loading, Sidebar Filters, and Main App Structure

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
    """Plot a single-PA interactive strike zone."""
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

    if type_col and type_col in pa_df.columns:
        colors_pts = [get_pitch_color(t) for t in pa_df[type_col].astype(str).tolist()]
    else:
        colors_pts = [HUSKER_RED] * len(pa_df)

    fig.add_trace(
        go.Scattergl(
            x=xs, y=ys, mode="markers+text",
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
            showlegend=False, name=""
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

def build_pitch_by_inning_pa_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build pitch-by-pitch table with PA results."""
    work = add_inning_and_ab(df)

    type_col   = type_col_in_df(work)
    result_col = pick_col(work, "PitchCall","Pitch Call","Call") or "PitchCall"
    velo_col   = pick_col(work, "RelSpeed","Relspeed","ReleaseSpeed")
    spin_col   = pick_col(work, "SpinRate","Spinrate","ReleaseSpinRate")
    ivb_col    = pick_col(work, "InducedVertBreak","IVB")
    hb_col     = pick_col(work, "HorzBreak","HB")
    x_col      = pick_col(work, "PlateLocSide","Plate Loc Side","PlateSide","px")
    y_col      = pick_col(work, "PlateLocHeight","Plate Loc Height","PlateHeight","pz")

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
        if isinstance(pr, str) and pr.strip(): return pr.strip()
        kb = row.get(col_korbb, "")
        if isinstance(kb, str):
            low = kb.strip().lower()
            if low in {"k","so","strikeout"}: return "Strikeout"
            if "walk" in low or low in {"bb","ibb"}: return "Walk"
        return "—"

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
def load_main_csv():
    if not os.path.exists(DATA_PATH_MAIN):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH_MAIN, low_memory=False)
    df = ensure_date_column(df)
    
    pitcher_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name")
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
    
    pitcher_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name")
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

df_all = pd.concat([df_main, df_scrim], ignore_index=True) if not df_scrim.empty else df_main.copy()
df_all = ensure_date_column(df_all)

neb_pitchers = sorted(df_all[df_all.get('PitcherTeam','') == 'NEB']['PitcherDisplay'].dropna().unique())

if len(neb_pitchers) == 0:
    st.error("No Nebraska pitchers found in data.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=150)
    
    st.markdown("### 📊 Data Filters")
    
    segment_choice = st.selectbox(
        "Season/Segment",
        options=["All Data"] + list(SEGMENT_DEFS.keys()),
        index=0,
        key="segment_choice"
    )
    
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
    st.markdown("### 📅 Date Filters")
    
    # Filter data by segment and pitcher to get available dates
    if segment_choice == "All Data":
        df_for_date_filter = df_all.copy()
    else:
        df_for_date_filter = filter_by_segment(df_all, segment_choice)
    
    df_for_date_filter = subset_by_pitcher_if_possible(df_for_date_filter, pitcher_choice)
    
    # Get unique dates for this pitcher
    if "Date" in df_for_date_filter.columns:
        pitcher_dates = pd.to_datetime(df_for_date_filter["Date"], errors="coerce").dropna()
        unique_dates = sorted(pitcher_dates.dt.date.unique())
        
        # Get available months (only those where pitcher has data)
        available_months = sorted(pitcher_dates.dt.month.unique())
        month_options = [name for num, name in MONTH_CHOICES if num in available_months]
        
        # Get available days (only those where pitcher has data)
        available_days = sorted(pitcher_dates.dt.day.unique())
    else:
        unique_dates = []
        month_options = []
        available_days = []
    
    # Game selector (by date)
    game_choices = st.multiselect(
        "Select Game(s)",
        options=[format_date_long(d) for d in unique_dates],
        default=[],
        help="Select specific games by date"
    )
    
    # Convert selected game strings back to dates
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
        min_value=0,
        max_value=50,
        value=0,
        step=1
    )
    
    st.markdown("---")
    st.caption("Nebraska Baseball Analytics")
# ═══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

if segment_choice == "All Data":
    df_pitcher_all = df_all.copy()
else:
    df_pitcher_all = filter_by_segment(df_all, segment_choice)

df_pitcher_all = subset_by_pitcher_if_possible(df_pitcher_all, pitcher_choice)

# Apply game filter (specific dates selected)
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

# Build better season label if games were selected
if selected_game_dates:
    if len(selected_game_dates) == 1:
        season_label_display = format_date_long(selected_game_dates[0])
    else:
        season_label_display = f"{len(selected_game_dates)} selected games"
elif game_choices:  # Games were selected but resulted in no data
    season_label_display = "Selected games"

if df_pitcher_all.empty:
    st.warning("No data matches the current filter selection.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS SETUP
# ═══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs(["📈 Standard", "👤 Profiles", "🏆 Rankings", "🍂 Fall Summary"])

# End of Part 5# Part 6 of 6: Main Tabs Implementation (Standard, Profiles, Rankings, Fall Summary)

# Part 6 of 6: Main Tabs Implementation (Standard, Profiles, Rankings, Fall Summary)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1: STANDARD
# ───────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    section_header("Season Overview")
    
    summary_table = make_outing_overall_summary(df_pitcher_all)
    st.dataframe(themed_table(summary_table), use_container_width=True)
    
    professional_divider()
    
    section_header("Movement Profile & Metrics")
    
    logo_img = load_logo_img()
    fig_movement, summary_movement = combined_pitcher_report(
        df_pitcher_all, pitcher_choice, logo_img, season_label=season_label_display
    )
    
    if fig_movement:
        show_and_close(fig_movement, use_container_width=True)
    else:
        st.info("Movement profile not available for current selection.")
    
    professional_divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PITCH-BY-PITCH BREAKDOWN (MOVED FROM PROFILES TAB)
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("Pitch-by-Pitch Breakdown")
    
    pbp_df = build_pitch_by_inning_pa_table(df_pitcher_all)
    
    if not pbp_df.empty and "Inning #" in pbp_df.columns:
        style_pbp_expanders()
        st.markdown('<div class="pbp-scope">', unsafe_allow_html=True)
        
        innings = sorted(pbp_df["Inning #"].dropna().unique())
        
        for inn in innings:
            inn_data = pbp_df[pbp_df["Inning #"] == inn]
            
            with st.expander(f"⚾ Inning {inn}", expanded=False):
                if "AB #" in inn_data.columns:
                    abs = sorted(inn_data["AB #"].dropna().unique())
                    
                    for ab_num in abs:
                        pa_data = inn_data[inn_data["AB #"] == ab_num]
                        
                        batter_name = pa_data["Batter"].iloc[0] if "Batter" in pa_data.columns else "Unknown"
                        pa_result = pa_data["PA Result"].iloc[0] if "PA Result" in pa_data.columns else "—"
                        
                        with st.expander(f"AB #{int(ab_num)}: {batter_name} ({pa_result})", expanded=False):
                            fig_pa = pa_interactive_strikezone(pa_data, title=f"AB #{int(ab_num)}: {batter_name}")
                            if fig_pa:
                                st.plotly_chart(fig_pa, use_container_width=True)
                            
                            display_cols = [c for c in pa_data.columns 
                                          if c not in ["PlateLocSide", "Plate Loc Side", "PlateSide", "px", "PlateLocX",
                                                      "PlateLocHeight", "Plate Loc Height", "PlateHeight", "pz", "PlateLocZ"]]
                            pa_display = pa_data[display_cols]
                            st.dataframe(themed_table(pa_display), use_container_width=True)
                else:
                    st.dataframe(themed_table(inn_data), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No pitch-by-pitch data available for current selection.")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2: PROFILES (PITCH-BY-PITCH REMOVED)
# ───────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    section_header(f"Pitcher Profile: {canonicalize_person_name(pitcher_choice)}")
    st.caption(f"**{season_label_display}** • Batter: {batter_side_choice}")
    
    # Outcome summary
    outcome_table = make_pitcher_outcome_summary_table(df_pitcher_all)
    st.dataframe(themed_table(outcome_table), use_container_width=True)
    
    professional_divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW: OVERALL PERFORMANCE METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("Overall Performance Metrics")
    
    perf_table = create_overall_performance_table(df_pitcher_all)
    if not perf_table.empty:
        def highlight_overall(row):
            if row['Pitch Type'] == 'OVERALL':
                return ['background-color: rgba(230, 0, 38, 0.1); font-weight: 600'] * len(row)
            return [''] * len(row)
        
        styled_perf = perf_table.style.apply(highlight_overall, axis=1).hide(axis="index").format({
            'Strike%': '{:.1f}',
            'Whiff%': '{:.1f}',
            'Zone%': '{:.1f}',
            'Chase%': '{:.1f}',
            'Zone Contact%': '{:.1f}',
            'InPlay%': '{:.1f}',
            'HardHit%': '{:.1f}',
            'Avg EV': '{:.1f}'
        }, na_rep="—")
        
        st.dataframe(styled_perf, use_container_width=True)
    else:
        info_message("Performance metrics not available.")
    
    professional_divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW: COUNT LEVERAGING HEATMAPS (3-panel matching hitter app)
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("Count Leveraging Heatmaps")
    st.caption("Pitch location density by count situation")
    
    fig_heatmaps = create_count_leverage_heatmaps(df_pitcher_all, pitcher_choice)
    
    if fig_heatmaps:
        show_and_close(fig_heatmaps, use_container_width=True)
    else:
        info_message("Count leveraging heatmaps not available. Requires count and location data.")
    
    # ADD PERFORMANCE BY COUNT TABLE
    st.markdown("#### Performance by Count Situation")
    count_performance = create_count_situation_comparison(df_pitcher_all)
    
    if not count_performance.empty:
        st.dataframe(themed_table(count_performance), use_container_width=True, hide_index=True)
    else:
        info_message("Count situation performance data not available.")
    
    professional_divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW: SPRAY CHART (matching hitter app style)
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("Spray Chart: Hits Allowed")
    st.caption("⚠️ Note: Spray chart shows hits from opponent batter's perspective")
    
    fig_spray, summary_spray = create_spray_chart(df_pitcher_all, pitcher_choice, season_label_display)
    
    if fig_spray:
        show_and_close(fig_spray, use_container_width=True)
        
        if summary_spray is not None and not summary_spray.empty:
            st.markdown("#### Batted Ball Summary")
            st.dataframe(themed_table(summary_spray), use_container_width=True, hide_index=True)
    else:
        info_message("Spray chart not available. Requires batted ball location data (Bearing/Distance).")
    
    professional_divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW: PITCH SEQUENCING
    # ═══════════════════════════════════════════════════════════════════════════
    section_header("Pitch Sequencing Analysis")
    
    trans_matrix, effectiveness, sankey_fig = analyze_pitch_sequences(df_pitcher_all, pitcher_choice)
    
    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)
        
        st.markdown("#### Most Effective Sequences")
        best_sequences = find_best_sequences(df_pitcher_all, pitcher_choice, min_count=5)
        
        if not best_sequences.empty:
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
        
        with st.expander("📊 Sequencing Strategy by Count"):
            seq_by_count = analyze_sequence_by_count(df_pitcher_all, pitcher_choice)
            if not seq_by_count.empty:
                st.dataframe(themed_table(seq_by_count), use_container_width=True)
            else:
                st.info("Count situation data not available.")
        
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
        
        sort_col = st.selectbox(
            "Sort by",
            options=["WHIP", "SO", "Strike%", "Whiff%", "HH%", "Barrel%", "IP"],
            index=0
        )
        
        if sort_col == "IP":
            rankings_display = rankings_df.sort_values("_IP_num", ascending=False)
        elif sort_col in ["WHIP", "HH%", "Barrel%"]:
            rankings_display = rankings_df.sort_values(sort_col, ascending=True, na_position="last")
        else:
            rankings_display = rankings_df.sort_values(sort_col, ascending=False, na_position="last")
        
        rankings_display = rankings_display.drop(columns=["_IP_num"])
        
        st.dataframe(themed_table(rankings_display), use_container_width=True, hide_index=True)
        
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
    
    df_fall = filter_by_segment(df_all, "2025/26 Scrimmages")
    
    if df_fall.empty:
        st.info("No fall scrimmage data available.")
    else:
        st.markdown(f"### Data Summary: {len(df_fall)} pitches from fall scrimmages")
        
        fall_pitchers = sorted(df_fall[df_fall.get('PitcherTeam','') == 'NEB']['PitcherDisplay'].dropna().unique())
        
        if fall_pitchers:
            selected_fall = st.selectbox(
                "Select Pitcher",
                options=fall_pitchers,
                key="fall_pitcher"
            )
            
            df_fall_p = subset_by_pitcher_if_possible(df_fall, selected_fall)
            
            professional_divider()
            
            logo_img = load_logo_img()
            fig_fall, summary_fall = combined_pitcher_report(
                df_fall_p, selected_fall, logo_img, season_label="Fall 2025/26"
            )
            
            if fig_fall:
                show_and_close(fig_fall, use_container_width=True)
            
            professional_divider()
            
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

# ═══════════════════════════════════════════════════════════════════════════════
# END OF APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
