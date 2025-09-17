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
from matplotlib.patches import Rectangle
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
DATA_PATH_SCRIM  = "Scrimmage(8).csv"  # file, directory, or glob pattern
DATA_PATH_2026   = "B10C26_hitter_app_columns.csv"  # placeholder; update when ready

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
    barrel = ((angle[inplay].between(10,30, inclusive="both")) & (exitv[inplay] >= 95)).mean()*100 if inplay.any() else 0.0

    # Discipline
    swing = isswing.mean()*100 if len(isswing) else 0.0
    whiff = (iswhiff.sum() / max(isswing.sum(),1) * 100) if len(isswing) else 0.0
    chase = (isswing[~isinzone].mean()*100) if (~isinzone).sum() else 0.0
    z_swing = (isswing[isinzone].mean()*100) if isinzone.sum() else 0.0
    z_contact = ((iscontact & isinzone).sum() / max(isswing[isinzone].sum(),1) * 100) if isinzone.sum() else 0.0
    z_whiff = ((iswhiff & isinzone).sum() / max(isswing[isinzone].sum(),1) * 100) if isinzone.sum() else 0.0

    return {
        "PA": pa, "AB": ab, "SO": so, "BB": bb, "Hits": hits,
        "2B": doubles, "3B": triples, "HR": hrs,
        "AVG": ba, "OBP": obp, "SLG": slg, "OPS": ops,
        "Avg EV": avg_ev, "Max EV": max_ev, "Avg LA": avg_la,
        "HardHit%": hard, "Barrel%": barrel,
        "Swing%": swing, "Whiff%": whiff, "Chase%": chase,
        "ZSwing%": z_swing, "ZContact%": z_contact, "ZWhiff%": z_whiff,
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
      t1_counts  -> Total + by pitch: PA AB SO BB Hits 2B 3B HR AVG OBP SLG OPS
      t2_rates   -> Total + by pitch: Avg EV Max EV Avg LA HardHit% Barrel% Swing% Whiff% Chase% ZSwing% ZContact% ZWhiff%
      t3_batted  -> Totals only: LD% GB% FB% Pull% Middle% Oppo%
    """
    if df_profiles.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Total rows
    core_total = _compute_split_core(df_profiles)
    row_total_counts = {"Split": "Total", **{k: core_total[k] for k in ["PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS"]}}
    row_total_rates  = {"Split": "Total", **{k: core_total[k] for k in ["Avg EV","Max EV","Avg LA","HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]}}

    t1_rows = [row_total_counts]
    t2_rows = [row_total_rates]

    # By pitch
    if "AutoPitchType" in df_profiles.columns:
        order = _sorted_unique_pitches(df_profiles["AutoPitchType"])
        for raw_pitch in order:
            sub = df_profiles[df_profiles["AutoPitchType"].astype(str) == raw_pitch]
            if sub.empty:
                continue
            core = _compute_split_core(sub)
            nice = _pretty_pitch_name(raw_pitch)
            t1_rows.append({"Split": nice, **{k: core[k] for k in ["PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS"]}})
            t2_rows.append({"Split": nice, **{k: core[k] for k in ["Avg EV","Max EV","Avg LA","HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]}})

    # DataFrames
    t1 = pd.DataFrame(t1_rows, columns=["Split","PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS"])
    t2 = pd.DataFrame(t2_rows, columns=["Split","Avg EV","Max EV","Avg LA","HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"])

    # Format t1 batting rates as .xxx (for tables)
    for c in ["AVG","OBP","SLG","OPS"]:
        t1[c] = t1[c].apply(lambda v: "—" if pd.isna(v) else (f"{float(v):.3f}"[1:] if float(v) < 1.0 else f"{float(v):.3f}"))

    # Format t2: EV/LA with 2 decimals, percents 1 decimal
    for c in ["Avg EV","Max EV","Avg LA"]:
        t2[c] = t2[c].apply(lambda v: "—" if pd.isna(v) else f"{float(v):.2f}")
    for c in ["HardHit%","Barrel%","Swing%","Whiff%","Chase%","ZSwing%","ZContact%","ZWhiff%"]:
        t2[c] = t2[c].apply(lambda v: "—" if pd.isna(v) else f"{round(float(v),1)}%")

    # T3 batted ball totals only
    t3 = create_batted_ball_profile(df_profiles).copy()
    for c in t3.columns:
        t3[c] = t3[c].apply(lambda v: "—" if pd.isna(v) else f"{float(v):.1f}%")

    return t1, t2, t3

# ──────────────────────────────────────────────────────────────────────────────
# RANKINGS HELPERS
# ──────────────────────────────────────────────────────────────────────────────
RANKABLE_COLS = ["PA","AB","SO","BB","Hits","2B","3B","HR","AVG","OBP","SLG","OPS"]

def build_rankings_numeric(df_player_scope: pd.DataFrame, display_name_by_key: dict) -> pd.DataFrame:
    """
    Build one NUMERIC row per BatterKey for rankings (click-column sorting).
    Slash stats remain numeric (0.xxx) to enable correct sorting.
    """
    rows = []
    for key, g in df_player_scope.groupby("BatterKey"):
        if not key:
            continue
        core = _compute_split_core(g)
        rows.append({
            "Player": display_name_by_key.get(key, key),
            **{k: core[k] for k in RANKABLE_COLS}
        })
    out = pd.DataFrame(rows, columns=["Player"] + RANKABLE_COLS)
    for c in RANKABLE_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def style_rankings(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply Husker red header + conditional fill for leaders (green) and last (red).
    Works with st.dataframe for click-to-sort.
    """
    numeric_cols = [c for c in RANKABLE_COLS if c in df.columns]

    def color_leader_last(col: pd.Series):
        if col.name not in numeric_cols:
            return [''] * len(col)
        s = pd.to_numeric(col, errors="coerce")
        if s.dropna().empty:
            return [''] * len(col)
        max_val = s.max()
        min_val = s.min()
        styles = []
        for v in s:
            if pd.isna(v):
                styles.append('')
            elif v == max_val and max_val != min_val:
                styles.append('background-color: #b6f2b0;')  # green
            elif v == min_val and max_val != min_val:
                styles.append('background-color: #f9b0b0;')  # red
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
          }, na_rep="—")
    )
    return sty

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD HITTER REPORT (single game) — with boxed legends bottom
# ──────────────────────────────────────────────────────────────────────────────
def create_hitter_report(df, batter_display_name, ncols=3):
    bdf = df  # already filtered to this batter upstream
    pa_groups = list(bdf.groupby(['GameID','Inning','Top/Bottom','PAofInning']))
    n_pa = len(pa_groups)
    nrows = max(1, math.ceil(n_pa / ncols))

    # helper to pretty-print TaggedHitType like "GroundBall" -> "Ground Ball"
    def _pretty_hit_type(s):
        if pd.isna(s) or s is None:
            return None
        t = str(s)
        t = t.replace("_", " ")
        t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
        return t.strip().title()

    # textual descriptions per PA
    descriptions = []
    for _, pa_df in pa_groups:
        lines = []
        for _, p in pa_df.iterrows():
            velo = p.get('EffectiveVelo', np.nan)
            velo_str = f"{float(velo):.1f}" if pd.notna(velo) else "—"
            lines.append(f"{int(p.get('PitchofPA', 0))} / {p.get('AutoPitchType', '—')}  {velo_str} MPH / {p.get('PitchCall', '—')}")
        inplay = pa_df[pa_df.get('PitchCall')=='InPlay']
        if not inplay.empty:
            last = inplay.iloc[-1]
            res = last.get('PlayResult', 'InPlay') or 'InPlay'
            es  = last.get('ExitSpeed', np.nan)
            tag = _pretty_hit_type(last.get('TaggedHitType'))
            result_bits = [res]
            if pd.notna(es):
                result_bits[-1] = f"{result_bits[-1]} ({float(es):.1f} MPH)"
            if tag:
                result_bits.append(f"— {tag}")
            lines.append(f"  ▶ PA Result: {' '.join(result_bits)}")
        else:
            balls = (pa_df.get('PitchCall')=='BallCalled').sum()
            strikes = pa_df.get('PitchCall').isin(['StrikeCalled','StrikeSwinging']).sum()
            if balls >= 4:
                lines.append("  ▶ PA Result: Walk 🚶")
            elif strikes >= 3:
                lines.append("  ▶ PA Result: Strikeout 💥")
        descriptions.append(lines)

    # Figure + grid (leave extra space at bottom for legends)
    fig = plt.figure(figsize=(3 + 4*ncols + 1, 4*nrows))
    gs = GridSpec(nrows, ncols+1, width_ratios=[0.8] + [1]*ncols, wspace=0.15, hspace=0.55)

    # Small top-right label: Name — Date   (no suptitle)
    date_str = ""
    if pa_groups:
        d0 = pa_groups[0][1].get('Date').iloc[0]
        date_str = format_date_long(d0)
    if batter_display_name or date_str:
        fig.text(0.985, 0.985, f"{batter_display_name} — {date_str}".strip(" —"),
            ha='right', va='top', fontsize=9, fontweight='normal')

    # summary line
    gd = pd.concat([grp for _, grp in pa_groups]) if pa_groups else pd.DataFrame()
    whiffs = (gd.get('PitchCall')=='StrikeSwinging').sum() if not gd.empty else 0
    hardhits = (pd.to_numeric(gd.get('ExitSpeed'), errors="coerce") > 95).sum() if not gd.empty else 0
    chases = 0
    if not gd.empty:
        pls = pd.to_numeric(gd.get('PlateLocSide'), errors='coerce')
        plh = pd.to_numeric(gd.get('PlateLocHeight'), errors='coerce')
        is_swing = gd.get('PitchCall').eq('StrikeSwinging')
        chases = (is_swing & ((pls<-0.83)|(pls>0.83)|(plh<1.5)|(plh>3.5))).sum()
    fig.text(0.55, 0.965, f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",
            ha='center', va='top', fontsize=12)

    # panels
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
                   'InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.get(str(p.get('PitchCall')), 'black')
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

    # left descriptions column
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

    # Legends: boxed, side-by-side at the bottom (Result LEFT, Pitches RIGHT)
    res_handles = [Line2D([0],[0], marker='o', color='w', label=k,
                          markerfacecolor=v, markersize=10, markeredgecolor='k')
                   for k,v in {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan',
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
# HITTER HEATMAPS — 3 panels (Contact, Whiffs, Damage)
# ──────────────────────────────────────────────────────────────────────────────
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
            "PlateLocSide", "PlateLocHeight", "TaggedHitType", "Bearing", "BatterSide"]:
    if col not in df_all.columns:
        df_all[col] = pd.NA

df_all = ensure_date_column(df_all)

df_neb_bat = df_all[df_all["BatterTeam"].astype(str).str.upper().eq("NEB")].copy()
df_neb_bat["PitchofPA"] = pd.to_numeric(df_neb_bat["PitchofPA"], errors="coerce")
df_neb_bat["BatterKey"]  = df_neb_bat["Batter"].map(normalize_name)
df_neb_bat["BatterDisp"] = df_neb_bat["BatterKey"]

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
# Top section selector (includes "Rankings")
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

        st.markdown("**Totals & By Pitch — PA/AB/SO/BB/Hits/2B/3B/HR & AVG/OBP/SLG/OPS**")
        st.table(themed_styler(t1_counts, nowrap=True))

        st.markdown("**Totals & By Pitch — EV/LA & Discipline**")
        st.table(themed_styler(t2_rates, nowrap=True))

        st.markdown("**Batted Ball Distribution (Totals only)**")
        st.table(themed_styler(t3_batted, nowrap=True))

        st.markdown("#### Hitter Heatmaps")
        fig_hm = hitter_heatmaps(df_profiles, batter_key)
        if fig_hm:
            st.pyplot(fig_hm)

# ──────────────────────────────────────────────────────────────────────────────
# MODE: RANKINGS (team-wide, click-to-sort, red headers, leader/last coloring)
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("### Rankings")

    # Filters apply team-wide
    st.markdown("#### Filters")
    colM, colD2, colN, colH = st.columns([1.2, 1.2, 0.9, 1.9])

    df_scope = df_neb_bat.copy()

    # months / days present in NEB batting set
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

    # Apply filters
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

    # Build numeric rankings table and optional Min PA filter
    rankings_df = build_rankings_numeric(df_scope, display_name_by_key)
    min_pa = int(st.number_input("Min PA", min_value=0, value=0, step=1, key="rk_min_pa"))
    if min_pa > 0:
        rankings_df = rankings_df[rankings_df["PA"] >= min_pa]

    # Styled (headers red + conditional leader/last cell coloring). Still sortable by clicking headers.
    styled = style_rankings(rankings_df)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=520
    )
