# pitcher_app.py (UPDATED)
# - 2025 Season uses pitcher_columns.csv
# - 2025/26 Scrimmages uses Fall_WinterScrimmages (1).csv (tries common variants)
# - STANDARD tab Movement plot prefers TaggedPitchType (fallbacks kept)
# - Release Points + Extensions now prefer TaggedPitchType (both segments)
# - Removed "Scrimmage data source" caption
# - Added "(FB Only)" label for Sept 4, 2025
# - Compare: Movement → Release Points → Extensions → Heatmaps; expand toggle
# - Profiles: Strike % = Total first then by pitch type; centered headers; cleaned titles

import os
import gc
import re
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from matplotlib.patches import Rectangle, Ellipse, Polygon, Circle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2, gaussian_kde
from numpy.linalg import LinAlgError
from matplotlib import colors
from datetime import date

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & PATHS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nebraska Baseball — Pitcher Reports",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.set_option("client.showErrorDetails", True)

DATA_PATH_MAIN = "pitcher_columns.csv"  # 2025 Season

# 2025/26 Scrimmages: prefer uploaded path & variants
SCRIM_PATHS = [
    "/mnt/data/Fall_WinterScrimmages (1).csv",  # uploaded location (with space)
    "/mnt/data/Fall_WinterScrimmages(1).csv",   # uploaded location (no space)
    "Fall_WinterScrimmages (1).csv",            # project dir (with space)
    "Fall_WinterScrimmages(1).csv",             # project dir (no space)
]

LOGO_PATH   = "Nebraska-Cornhuskers-Logo.png"
BANNER_IMG  = "NebraskaChampions.jpg"
HUSKER_RED  = "#E60026"
EXT_VIS_WIDTH = 480  # default on-screen width for Extensions

# ──────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS & RENDER HELPERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_banner_b64() -> str | None:
    if not os.path.exists(BANNER_IMG):
        return None
    with open(BANNER_IMG, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@st.cache_resource
def load_logo_img():
    if os.path.exists(LOGO_PATH):
        return mpimg.imread(LOGO_PATH)
    return None

def show_and_close(fig, *, use_container_width: bool = False):
    try:
        st.pyplot(fig=fig, clear_figure=False, use_container_width=use_container_width)
    finally:
        plt.close(fig)
        gc.collect()

def show_image_scaled(fig, *, width_px: int = EXT_VIS_WIDTH, dpi: int = 200, pad_inches: float = 0.1):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    buf.seek(0)
    st.image(buf, width=width_px)
    plt.close(fig)
    gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────────────────────────────
def hero_banner(title: str, *, subtitle: str | None = None, height_px: int = 260):
    from streamlit.components.v1 import html as _html
    b64 = load_banner_b64()
    bg_url = f"data:image/jpeg;base64,{b64}" if b64 else ""
    sub_html = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    _html(
        f"""
        <style>
        .hero-wrap {{
            position: relative; width: 100%; height: {height_px}px; border-radius: 10px;
            overflow: hidden; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }}
        .hero-bg {{
            position: absolute; inset: 0;
            background: linear-gradient(to bottom, rgba(0,0,0,0.45), rgba(0,0,0,0.60)), url('{bg_url}');
            background-size: cover; background-position: center; filter: saturate(105%);
        }}
        .hero-text {{
            position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
            flex-direction: column; color: #fff; text-align: center;
        }}
        .hero-title {{ font-size: 40px; font-weight: 800; letter-spacing: .5px; text-shadow: 0 2px 8px rgba(0,0,0,.45); margin: 0; }}
        .hero-sub   {{ font-size: 18px; font-weight: 600; opacity: .95; margin-top: 6px; }}
        </style>
        <div class="hero-wrap">
          <div class="hero-bg"></div>
          <div class="hero-text">
            <h1 class="hero-title">{title}</h1>
            {sub_html}
          </div>
        </div>
        """,
        height=height_px + 28,
    )

hero_banner("Nebraska Baseball", subtitle=None, height_px=260)

# ──────────────────────────────────────────────────────────────────────────────
# DATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
DATE_CANDIDATES = [
    "Date","date","GameDate","GAME_DATE","Game Date","date_game","Datetime",
    "DateTime","game_datetime","GameDateTime"
]

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    found = None
    lower = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand.lower() in lower:
            found = lower[cand.lower()]
            break
    if found is None:
        df["Date"] = pd.NaT
        return df
    dt = pd.to_datetime(df[found], errors="coerce")
    df["Date"] = pd.to_datetime(dt.dt.date, errors="coerce")
    return df

def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"

def format_date_long(d) -> str:
    if d is None or pd.isna(d):
        return ""
    d = pd.to_datetime(d).date()
    return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"

# Scrimmage "(FB Only)" labels
from datetime import date as _date
FB_ONLY_DATES = { _date(2025, 9, 3), _date(2025, 9, 4) }  # added Sept 4, 2025

def is_fb_only(d) -> bool:
    try:
        return pd.to_datetime(d).date() in FB_ONLY_DATES
    except Exception:
        return False

def label_date_with_fb(d) -> str:
    base = format_date_long(d)
    return f"{base} (FB Only)" if is_fb_only(d) else base

def summarize_dates_range(series_like) -> str:
    if series_like is None:
        return ""
    if not isinstance(series_like, pd.Series):
        series_like = pd.Series(series_like)
    ser = pd.to_datetime(series_like, errors="coerce").dropna()
    if ser.empty:
        return ""
    uniq = ser.dt.date.unique()
    if len(uniq) == 1:
        return format_date_long(uniq[0])
    dmin, dmax = min(uniq), max(uniq)
    return f"{format_date_long(dmin)} – {format_date_long(dmax)}"

def filter_by_month_day(df, date_col="Date", months=None, days=None):
    if date_col not in df.columns or df.empty:
        return df
    s = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if months:
        mask &= s.dt.month.isin(months)
    if days:
        mask &= s.dt.day.isin(days)
    return df[mask]

MONTH_CHOICES = [
    (1,"January"), (2,"February"), (3,"March"), (4,"April"), (5,"May"), (6,"June"),
    (7,"July"), (8,"August"), (9,"September"), (10,"October"), (11,"November"), (12,"December")
]
MONTH_NAME_BY_NUM = {n: name for n, name in MONTH_CHOICES}

def build_pitcher_season_label(months_sel, days_sel, selected_df: pd.DataFrame) -> str:
    if (not months_sel) and (not days_sel):
        return "Season"
    if months_sel and not days_sel and len(months_sel) == 1:
        return MONTH_NAME_BY_NUM.get(months_sel[0], "Season")
    if selected_df is None or selected_df.empty or "Date" not in selected_df.columns:
        return "Season"
    rng = summarize_dates_range(selected_df["Date"])
    return rng if rng else "Season"

# ──────────────────────────────────────────────────────────────────────────────
# SEGMENTS
# ──────────────────────────────────────────────────────────────────────────────
SESSION_TYPE_CANDIDATES = [
    "SessionType","Session Type","GameType","Game Type","EventType","Event Type",
    "Context","context","Type","type","Environment","Env"
]

def pick_col(df: pd.DataFrame, *cands) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c and c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def find_session_type_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, *SESSION_TYPE_CANDIDATES)

def _norm_session_type(val: str) -> str:
    s = str(val).strip().lower()
    if not s or s == "nan":
        return ""
    if any(k in s for k in ["scrim", "intra", "fall ball", "exhib"]):
        return "scrimmage"
    if any(k in s for k in ["bullpen", "pen", "bp"]):
        return "bullpen"
    if any(k in s for k in ["game", "regular", "season", "conf", "non-conf", "ncaa"]):
        return "game"
    return ""

SEGMENT_DEFS = {
    "2025 Season":        {"start": "2025-02-01", "end": "2025-08-01", "types": ["game"]},
    "2025/26 Scrimmages": {"start": "2025-08-01", "end": "2026-02-01", "types": ["scrimmage"]},
    "2025/26 Bullpens":   {"start": "2025-08-01", "end": "2026-02-01", "types": ["bullpen"]},
    "2026 Season":        {"start": "2026-02-01", "end": "2026-08-01", "types": ["game"]},
}

def filter_by_segment(df: pd.DataFrame, segment_name: str) -> pd.DataFrame:
    spec = SEGMENT_DEFS.get(segment_name)
    if spec is None or df.empty:
        return df
    out = df.copy()
    if "Date" in out.columns:
        d = pd.to_datetime(out["Date"], errors="coerce")
        start = pd.to_datetime(spec["start"]); end = pd.to_datetime(spec["end"])
        out = out[(d >= start) & (d < end)]
    st_col = find_session_type_col(out)
    if st_col and len(spec.get("types", [])) > 0:
        st_norm = out[st_col].apply(_norm_session_type)
        out = out[st_norm.isin(spec["types"])]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# STRIKE ZONE & COLORS
# ──────────────────────────────────────────────────────────────────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
    N=256,
)

def get_zone_bounds():
    left, bottom = -0.83, 1.17
    width, height = 1.66, 2.75
    return left, bottom, width, height

def get_view_bounds():
    left, bottom, width, height = get_zone_bounds()
    mx, my = width * 0.8, height * 0.6
    return left - mx, left + width + mx, bottom - my, bottom + height + my

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
    if isinstance(ptype, str) and (ptype.lower().startswith("four-seam fastball") or ptype.lower() == "fastball"):
        return "#E60026"
    savant = {
        "sinker": "#FF9300","cutter": "#800080","changeup": "#008000","curveball": "#0033CC",
        "slider": "#CCCC00","splitter": "#00CCCC","knuckle curve": "#000000","screwball": "#CC0066","eephus": "#666666",
    }
    return savant.get(str(ptype).lower(), "#E60026")

def format_name(name):
    if isinstance(name, str) and ',' in name:
        last, first = [s.strip() for s in name.split(',', 1)]
        return f"{first} {last}"
    return str(name)

# ──────────────────────────────────────────────────────────────────────────────
# DENSITY / UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def compute_density(x, y, grid_coords, mesh_shape):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.zeros(mesh_shape)
    try:
        kde = gaussian_kde(np.vstack([x, y]))
        return kde(grid_coords).reshape(mesh_shape)
    except LinAlgError:
        return np.zeros(mesh_shape)

def strike_rate(df):
    if len(df) == 0 or "PitchCall" not in df.columns:
        return np.nan
    strike_calls = ['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']
    return df['PitchCall'].isin(strike_calls).mean() * 100

def find_batter_side_col(df: pd.DataFrame) -> str | None:
    return pick_col(
        df, "BatterSide", "Batter Side", "Batter_Bats", "BatterBats", "Bats", "Stand", "BatSide", "BatterBatSide", "BatterBatHand"
    )

def normalize_batter_side(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str[0].str.upper()
    return s.replace({"L":"L","R":"R","S":"S","B":"S"})

def parse_hand_filter_to_LR(hand_filter: str) -> str | None:
    s = str(hand_filter).strip().lower()
    s = s.replace("vs", "").replace("batters", "").replace("hitters", "").strip()
    if s in {"l", "lhh", "lhb", "left", "left-handed", "left handed"}:  return "L"
    if s in {"r", "rhh", "rhb", "right", "right-handed", "right handed"}: return "R"
    return None

# ──────────────────────────────────────────────────────────────────────────────
# UI: grid of per-pitch checkboxes
# ──────────────────────────────────────────────────────────────────────────────
def _safe_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s))

def pitchtype_checkbox_grid(label: str, options: list[str], key_prefix: str, default_all=True, columns_per_row=6) -> list[str]:
    options = list(dict.fromkeys([str(o) for o in options]))
    if not options:
        st.caption("No pitch types available.")
        return []
    opt_keys = [f"{key_prefix}_{_safe_key(o)}" for o in options]
    for k in opt_keys:
        if k not in st.session_state:
            st.session_state[k] = bool(default_all)
    st.write(f"**{label}**")
    col_a, col_b = st.columns([0.12, 0.12])
    if col_a.button("Select all", key=f"{key_prefix}_select_all"):
        for k in opt_keys: st.session_state[k] = True
    if col_b.button("Clear all", key=f"{key_prefix}_clear_all"):
        for k in opt_keys: st.session_state[k] = False
    cols = st.columns(columns_per_row)
    for i, (o, k) in enumerate(zip(options, opt_keys)):
        cols[i % columns_per_row].checkbox(o, value=st.session_state[k], key=k)
    return [o for o, k in zip(options, opt_keys) if st.session_state[k]]

# ──────────────────────────────────────────────────────────────────────────────
# PITCHER REPORT (movement + summary)
# ──────────────────────────────────────────────────────────────────────────────
def combined_pitcher_report(
    df, pitcher_name, logo_img, coverage=0.8, season_label="Season", *, prefer_tagged: bool = False
):
    """
    prefer_tagged=True ⇒ use TaggedPitchType (fallback to others if missing).
    """
    if prefer_tagged:
        type_col = (
            pick_col(df, "TaggedPitchType", "Tagged Pitch Type", "Tagged_Type", "PitchType",
                     "AutoPitchType", "Auto Pitch Type")
            or "TaggedPitchType"
        )
    else:
        type_col = (
            pick_col(df, "AutoPitchType", "Auto Pitch Type", "PitchType", "TaggedPitchType")
            or "AutoPitchType"
        )

    pitch_col = pick_col(df, "PitchCall","Pitch Call","Call") or "PitchCall"
    speed_col = pick_col(df, "RelSpeed","Relspeed","ReleaseSpeed","RelSpeedMPH","release_speed")
    spin_col  = pick_col(df, "SpinRate","Spinrate","ReleaseSpinRate","Spin")
    ivb_col   = pick_col(df, "InducedVertBreak","IVB","Induced Vert Break","IndVertBreak")
    hb_col    = pick_col(df, "HorzBreak","HorizontalBreak","HB","HorizBreak")
    rh_col    = pick_col(df, "RelHeight","Relheight","ReleaseHeight","Release_Height","release_pos_z")
    vaa_col   = pick_col(df, "VertApprAngle","VAA","VerticalApproachAngle")
    ext_col   = pick_col(df, "Extension","Ext","ReleaseExtension","ExtensionInFt","Extension(ft)")

    df_p = df[df.get('Pitcher', '') == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' with the current filters.")
        return None

    try:
        grp = df_p.groupby(type_col, dropna=False)
    except KeyError:
        st.error(f"Pitch type column not found (tried '{type_col}').")
        return None

    counts = grp.size()
    total = int(len(df_p))

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
        nonlocal summary
        if col_name and col_name in df_p.columns:
            vals = grp[col_name].mean().values
            summary[label] = np.round(vals, r)

    add_mean(speed_col, 'Rel Speed', r=1)
    add_mean(spin_col,  'Spin Rate', r=1)
    add_mean(ivb_col,   'IVB',       r=1)
    add_mean(hb_col,    'HB',        r=1)
    add_mean(rh_col,    'Rel Height',r=2)
    add_mean(vaa_col,   'VAA',       r=1)
    add_mean(ext_col,   'Extension', r=2)

    summary = summary.sort_values('Pitches', ascending=False)

    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.7], hspace=0.3)

    axm = fig.add_subplot(gs[0, 0]); axm.set_title('Movement Plot', fontweight='bold')
    axm.axhline(0, ls='--', color='grey'); axm.axvline(0, ls='--', color='grey')
    chi2v = chi2.ppf(coverage, df=2)

    for ptype, g in df_p.groupby(type_col, dropna=False):
        clr = get_pitch_color(ptype)
        x = pd.to_numeric(g[hb_col], errors='coerce') if hb_col in g.columns else pd.Series([np.nan]*len(g))
        y = pd.to_numeric(g[ivb_col], errors='coerce') if ivb_col in g.columns else pd.Series([np.nan]*len(g))
        mask = x.notna() & y.notna()
        if mask.any():
            axm.scatter(x[mask], y[mask], label=str(ptype), color=clr, alpha=0.7)
            if mask.sum() > 1:
                X = np.vstack((x[mask], y[mask])); cov = np.cov(X)
                try:
                    vals, vecs = np.linalg.eigh(cov)
                    ord_ = vals.argsort()[::-1]; vals, vecs = vals[ord_], vecs[:, ord_]
                    ang = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                    w, h = 2*np.sqrt(vals*chi2v)
                    axm.add_patch(Ellipse((x[mask].mean(), y[mask].mean()), w, h, angle=ang,
                                          edgecolor=clr, facecolor=clr, alpha=0.2, ls='--', lw=1.5))
                except Exception:
                    pass
        else:
            axm.scatter([], [], label=str(ptype), color=clr, alpha=0.7)

    axm.set_xlim(-30,30); axm.set_ylim(-30,30); axm.set_aspect('equal','box')
    axm.set_xlabel('Horizontal Break'); axm.set_ylabel('Induced Vertical Break')
    axm.legend(title='Pitch Type', fontsize=8, title_fontsize=9, loc='upper right')

    axt = fig.add_subplot(gs[1, 0]); axt.axis('off')
    tbl = axt.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Summary Metrics', fontweight='bold', y=0.87)

    logo_img = load_logo_img()
    if logo_img is not None:
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10); axl.imshow(logo_img); axl.axis('off')

    fig.suptitle(f"{format_name(pitcher_name)} Metrics\n({season_label})", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, summary

# ──────────────────────────────────────────────────────────────────────────────
# HEATMAPS (split & combined) — titles cleaned
# ──────────────────────────────────────────────────────────────────────────────
def _heatmap_panel(ax, sub, title, *, x_min, x_max, y_min, y_max, grid_coords, xi_mesh,
                   z_left, z_bottom, z_w, z_h, color='deepskyblue', threshold=12):
    n = len(sub)
    x = sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy()
    y = sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy()
    if n < threshold:
        ax.scatter(x, y, s=30, alpha=0.7, color=color, edgecolors='black')
    else:
        zi = compute_density(x, y, grid_coords, xi_mesh.shape)
        ax.imshow(zi, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax, z_left, z_bottom, z_w, z_h)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal','box')
    ax.set_title(title, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])

def heatmaps_top3_pitch_types(df, pitcher_name, hand_filter="Both", grid_size=100, season_label="Season"):
    df_p = df[df.get('Pitcher','') == pitcher_name].copy()
    if df_p.empty:
        st.info("No data for the selected filters.")
        return None

    type_col = pick_col(df_p, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"

    side_col = find_batter_side_col(df_p)
    if side_col is not None:
        sides = normalize_batter_side(df_p[side_col])
        want = parse_hand_filter_to_LR(hand_filter)
        if   want == "L": df_p = df_p[sides == "L"]
        elif want == "R": df_p = df_p[sides == "R"]

    if df_p.empty:
        st.info("No pitches for the selected batter-side filter.")
        return None

    x_min, x_max, y_min, y_max = get_view_bounds()
    xi = np.linspace(x_min, x_max, grid_size); yi = np.linspace(y_min, y_max, grid_size)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi); grid_coords = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()])
    z_left, z_bottom, z_w, z_h = get_zone_bounds()

    try:
        top3 = list(df_p[type_col].value_counts().index[:3])
    except KeyError:
        top3 = []

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        if i < len(top3):
            pitch = top3[i]
            sub = df_p[df_p[type_col] == pitch]
            _heatmap_panel(ax, sub, f"{pitch} (n={len(sub)})",
                           x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                           grid_coords=grid_coords, xi_mesh=xi_mesh,
                           z_left=z_left, z_bottom=z_bottom, z_w=z_w, z_h=z_h)
        else:
            draw_strikezone(ax, z_left, z_bottom, z_w, z_h)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal','box')
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title("—", fontweight='bold')

    fig.suptitle(f"{format_name(pitcher_name)} — Top 3 Pitches ({season_label})", fontsize=16, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
def heatmaps_outcomes(df, pitcher_name, hand_filter="Both", grid_size=100, season_label="Season", outcome_pitch_types=None):
    df_p = df[df.get('Pitcher','') == pitcher_name].copy()
    if df_p.empty:
        st.info("No data for the selected filters.")
        return None

    type_col = pick_col(df_p, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"

    side_col = find_batter_side_col(df_p)
    if side_col is not None:
        sides = normalize_batter_side(df_p[side_col])
        want = parse_hand_filter_to_LR(hand_filter)
        if   want == "L": df_p = df_p[sides == "L"]
        elif want == "R": df_p = df_p[sides == "R"]

    if df_p.empty:
        st.info("No pitches for the selected batter-side filter.")
        return None

    df_out = df_p
    if outcome_pitch_types is not None:
        if len(outcome_pitch_types) == 0:
            df_out = df_p.iloc[0:0].copy()
        else:
            if type_col in df_p.columns:
                df_out = df_p[df_p[type_col].astype(str).isin(list(outcome_pitch_types))].copy()

    x_min, x_max, y_min, y_max = get_view_bounds()
    xi = np.linspace(x_min, x_max, grid_size); yi = np.linspace(y_min, y_max, grid_size)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi); grid_coords = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()])
    z_left, z_bottom, z_w, z_h = get_zone_bounds()

    sub_wh = df_out[df_out.get('PitchCall','') == 'StrikeSwinging']
    sub_ks = df_out[df_out.get('KorBB','') == 'Strikeout']
    sub_dg = df_out[pd.to_numeric(df_out.get('ExitSpeed', pd.Series(dtype=float)), errors='coerce') >= 95]

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0]); _heatmap_panel(ax, sub_wh, f"Whiffs (n={len(sub_wh)})",
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        grid_coords=grid_coords, xi_mesh=xi_mesh, z_left=z_left, z_bottom=z_bottom, z_w=z_w, z_h=z_h)
    ax = fig.add_subplot(gs[0, 1]); _heatmap_panel(ax, sub_ks, f"Strikeouts (n={len(sub_ks)})",
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        grid_coords=grid_coords, xi_mesh=xi_mesh, z_left=z_left, z_bottom=z_bottom, z_w=z_w, z_h=z_h)
    ax = fig.add_subplot(gs[0, 2]); _heatmap_panel(ax, sub_dg, f"Damage (n={len(sub_dg)})",
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        grid_coords=grid_coords, xi_mesh=xi_mesh, z_left=z_left, z_bottom=z_bottom, z_w=z_w, z_h=z_h, color='orange')

    fig.suptitle(f"{format_name(pitcher_name)} — Outcomes ({season_label})", fontsize=16, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def combined_pitcher_heatmap_report(
    df, pitcher_name, hand_filter="Both", grid_size=100, season_label="Season", outcome_pitch_types=None,
):
    df_p = df[df.get('Pitcher','') == pitcher_name].copy()
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' with the current filters.")
        return None
    type_col = pick_col(df_p, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"

    side_col = find_batter_side_col(df_p)
    hand_label = "Both"
    if side_col is not None:
        sides = normalize_batter_side(df_p[side_col])
        want = parse_hand_filter_to_LR(hand_filter)
        if want == "L": df_p = df_p[sides == "L"]; hand_label = "LHH"
        elif want == "R": df_p = df_p[sides == "R"]; hand_label = "RHH"
    else:
        st.caption("Batter-side column not found; showing Both.")
    if df_p.empty:
        st.info("No pitches for the selected batter-side filter.")
        return None

    x_min, x_max, y_min, y_max = get_view_bounds()
    xi = np.linspace(x_min, x_max, grid_size); yi = np.linspace(y_min, y_max, grid_size)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi); grid_coords = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()])
    z_left, z_bottom, z_w, z_h = get_zone_bounds()
    threshold = 12

    def panel(ax, sub, title, color='deepskyblue'):
        _heatmap_panel(ax, sub, title, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                       grid_coords=grid_coords, xi_mesh=xi_mesh, z_left=z_left, z_bottom=z_bottom, z_w=z_w, z_h=z_h,
                       color=color, threshold=threshold)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.6], hspace=0.35, wspace=0.3)

    try:
        top3 = list(df_p[type_col].value_counts().index[:3])
    except KeyError:
        top3 = []

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        if i < len(top3):
            pitch = top3[i]
            sub = df_p[df_p[type_col] == pitch]
            panel(ax, sub, f"{pitch} (n={len(sub)})")
        else:
            draw_strikezone(ax, z_left, z_bottom, z_w, z_h)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal','box')
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title("—", fontweight='bold')

    df_out = df_p
    if outcome_pitch_types is not None:
        if len(outcome_pitch_types) == 0:
            df_out = df_p.iloc[0:0].copy()
        else:
            if type_col in df_p.columns:
                df_out = df_p[df_p[type_col].astype(str).isin(list(outcome_pitch_types))].copy()

    sub_wh = df_out[df_out.get('PitchCall','') == 'StrikeSwinging']
    sub_ks = df_out[df_out.get('KorBB','') == 'Strikeout']
    sub_dg = df_out[pd.to_numeric(df_out.get('ExitSpeed', pd.Series(dtype=float)), errors='coerce') >= 95]
    ax = fig.add_subplot(gs[1, 0]); panel(ax, sub_wh, f"Whiffs (n={len(sub_wh)})")
    ax = fig.add_subplot(gs[1, 1]); panel(ax, sub_ks, f"Strikeouts (n={len(sub_ks)})")
    ax = fig.add_subplot(gs[1, 2]); panel(ax, sub_dg, f"Damage (n={len(sub_dg)})", color='orange')

    # Strike % table row (legacy combined figure)
    axt = fig.add_subplot(gs[2, :]); axt.axis('off')
    def _safe_mask(q):
        for col in ("Balls","Strikes"):
            if col not in df_p.columns: return df_p.iloc[0:0]
        return q
    fp  = strike_rate(_safe_mask(df_p[(df_p.get('Balls',0)==0) & (df_p.get('Strikes',0)==0)]))
    mix = strike_rate(_safe_mask(df_p[((df_p.get('Balls',0)==1)&(df_p.get('Strikes',0)==0)) |
                                      ((df_p.get('Balls',0)==0)&(df_p.get('Strikes',0)==1)) |
                                      ((df_p.get('Balls',0)==1)&(df_p.get('Strikes',0)==1))]))
    hp  = strike_rate(_safe_mask(df_p[((df_p.get('Balls',0)==2)&(df_p.get('Strikes',0)==0)) |
                                      ((df_p.get('Balls',0)==2)&(df_p.get('Strikes',0)==1)) |
                                      ((df_p.get('Balls',0)==3)&(df_p.get('Strikes',0)==1))]))
    two = strike_rate(_safe_mask(df_p[(df_p.get('Strikes',0)==2) & (df_p.get('Balls',0)<3)]))
    metrics = pd.DataFrame({'1st Pitch %':[fp],'Mix Count %':[mix],'Hitter+ %':[hp],'2-Strike %':[two]}).round(1)
    tbl = axt.table(cellText=metrics.values, colLabels=metrics.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Strike Percentage by Count', y=0.75, fontweight='bold')

    fig.suptitle(f"{format_name(pitcher_name)}\n({season_label}) ({hand_label})", fontsize=18, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# RELEASE POINTS  (now prefers TaggedPitchType)
# ──────────────────────────────────────────────────────────────────────────────
ARM_BASE_HALF_WIDTH = 0.24
ARM_TIP_HALF_WIDTH  = 0.08
SHOULDER_RADIUS_OUT = 0.20
HAND_RING_OUTER_R   = 0.26
HAND_RING_INNER_R   = 0.15
ARM_FILL_COLOR      = "#111111"

def norm_text(x: str) -> str: return str(x).strip().lower()
def norm_type(x: str) -> str:
    s = norm_text(x)
    return {"four seam":"four-seam","4 seam":"4-seam","two seam":"two-seam","2 seam":"2-seam"}.get(s, s)

def canonicalize_type(raw: str) -> str:
    s = norm_text(raw)
    if "sinker" in s or s in {"si","snk"}: return "Fastball"
    n = norm_type(raw)
    return {
        "four-seam":"Fastball","4-seam":"Fastball","fastball":"Fastball",
        "two-seam":"Two-Seam Fastball","2-seam":"Two-Seam Fastball",
        "cutter":"Cutter","changeup":"Changeup","splitter":"Splitter",
        "curveball":"Curveball","knuckle curve":"Knuckle Curve","slider":"Slider",
        "sweeper":"Sweeper","screwball":"Screwball","eephus":"Eephus",
    }.get(n, "Unknown")

def color_for_release(canon_label: str) -> str:
    key = str(canon_label).lower()
    palette = {
        "fastball": "#E60026","two-seam fastball": "#FF9300","cutter": "#800080","changeup": "#008000",
        "splitter": "#00CCCC","curveball": "#0033CC","knuckle curve": "#000000","slider": "#CCCC00",
        "sweeper": "#B5651D","screwball": "#CC0066","eephus": "#666666",
    }
    return palette.get(key, "#7F7F7F")

def release_points_figure(df: pd.DataFrame, pitcher_name: str, include_types=None):
    pitcher_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name","PitcherLastFirst") or "Pitcher"
    x_col = pick_col(df, "Relside","RelSide","ReleaseSide","Release_Side","release_pos_x")
    y_col = pick_col(df, "Relheight","RelHeight","ReleaseHeight","Release_Height","release_pos_z")
    # Prefer TaggedPitchType, then fall back
    type_col = pick_col(df, "TaggedPitchType","Tagged Pitch Type","Tagged_Type","PitchType","AutoPitchType","Auto Pitch Type") or "TaggedPitchType"
    speed_col = pick_col(df, "Relspeed","RelSpeed","ReleaseSpeed","RelSpeedMPH","release_speed")

    missing = [lbl for lbl, col in [("Relside",x_col), ("Relheight",y_col)] if col is None]
    if missing:
        st.error(f"Missing required column(s) for release plot: {', '.join(missing)}")
        return None

    sub = df[df[pitcher_col] == pitcher_name].copy()
    if sub.empty:
        st.error(f"No rows found for pitcher '{pitcher_name}'.")
        return None

    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    if speed_col:
        sub[speed_col] = pd.to_numeric(sub[speed_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col])

    sub["_type_canon"] = sub[type_col].apply(canonicalize_type)
    sub = sub[sub["_type_canon"] != "Unknown"].copy()

    if include_types is not None and len(include_types) > 0:
        sub = sub[sub["_type_canon"].isin(include_types)]
    if sub.empty:
        st.warning("No pitches after applying the selected pitch-type filter.")
        return None

    sub["_color"] = sub["_type_canon"].apply(color_for_release)

    agg = {"mean_x": (x_col, "mean"), "mean_y": (y_col, "mean")}
    if speed_col:
        agg["mean_speed"] = (speed_col, "mean")
    means = sub.groupby("_type_canon", as_index=False).agg(**agg)
    means["color"] = means["_type_canon"].apply(color_for_release)
    if "mean_speed" in means.columns:
        means = (means.sort_values("mean_speed", ascending=False, na_position="last").reset_index(drop=True))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.0), sharey=True)

    ax1.scatter(sub[x_col], sub[y_col], s=12, alpha=0.75, c=sub["_color"], edgecolors="none")
    ax1.set_xlim(-5, 5); ax1.set_ylim(0, 8); ax1.set_aspect("equal")
    ax1.axhline(0, color="black", linewidth=1); ax1.axvline(0, color="black", linewidth=1)
    ax1.set_xlabel(x_col); ax1.set_ylabel(y_col)
    ax1.set_title(f"All Releases (n={len(sub)})", fontweight="bold")

    for _, row in means.iterrows():
        x0, y0 = 0.0, 0.0
        x1, y1 = float(row["mean_x"]), float(row["mean_y"])
        dx, dy = x1 - x0, y1 - y0
        L = float(np.hypot(dx, dy))
        if L <= 1e-6: continue
        ux, uy = dx / L, dy / L
        px, py = -uy, ux
        sLx, sLy = x0 + px*ARM_BASE_HALF_WIDTH, y0 + py*ARM_BASE_HALF_WIDTH
        sRx, sRy = x0 - px*ARM_BASE_HALF_WIDTH, y0 - py*ARM_BASE_HALF_WIDTH
        eLx, eLy = x1 + px*ARM_TIP_HALF_WIDTH, y1 + py*ARM_TIP_HALF_WIDTH
        eRx, eRy = x1 - px*ARM_TIP_HALF_WIDTH, y1 - py*ARM_TIP_HALF_WIDTH
        arm_poly = Polygon([(sLx, sLy), (eLx, eLy), (eRx, eRy), (sRx, sRy)], closed=True,
                           facecolor=ARM_FILL_COLOR, edgecolor=ARM_FILL_COLOR, zorder=1)
        ax2.add_patch(arm_poly)
        ax2.add_patch(Circle((x0, y0), radius=0.20, facecolor="#0d0d0d", edgecolor="#0d0d0d", zorder=2))
        outer = Circle((x1, y1), radius=0.26, facecolor=row["color"], edgecolor=row["color"], zorder=4)
        ax2.add_patch(outer)
        inner_face = ax2.get_facecolor()
        inner = Circle((x1, y1), radius=0.15, facecolor=inner_face, edgecolor=inner_face, zorder=5)
        ax2.add_patch(inner)

    ax2.set_xlim(-5, 5); ax2.set_ylim(0, 8); ax2.set_aspect("equal")
    ax2.axhline(0, color="black", linewidth=1); ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel(x_col); ax2.set_title("Average Releases", fontweight="bold")

    handles = []
    for _, row in means.iterrows():
        label = row["_type_canon"]
        if "mean_speed" in means.columns and not pd.isna(row.get("mean_speed", None)):
            label = f"{label} ({row['mean_speed']:.1f})"
        handles.append(Line2D([0],[0], marker="o", linestyle="none", markersize=6, label=label, color=row["color"]))
    if handles:
        ax2.legend(handles=handles, title="Pitch Type", loc="upper right")

    fig.suptitle(f"{format_name(pitcher_name)} Release Points", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# EXTENSIONS (now prefers TaggedPitchType)
# ──────────────────────────────────────────────────────────────────────────────
RELEASE_X_OFFSET_FT = 2.0
SHOULDER_X_R = 0.7
SHOULDER_X_L = -0.7
SHOULDER_Y = 1.6
ARM_THICK_BASE = 0.28
ARM_THICK_TIP  = 0.05
MOUND_COLOR    = "#7B4B24"
MOUND_EDGE     = "#3D2A1A"
MOUND_ALPHA    = 0.90
LEGEND_LOC     = "upper center"
LEGEND_ANCHOR_FRAC = (0.50, 0.98)
XLIM_PAD_DEFAULT = 11.0
YLIM_PAD_DEFAULT = 4.0
YLIM_BOTTOM     = -10.0

def _decide_pitcher_hand(sub: pd.DataFrame) -> str:
    hand_col = pick_col(sub, "PitcherThrows","Throws","PitcherHand","Pitcher_Hand","ThrowHand")
    relside_col = pick_col(sub, "Relside","RelSide","ReleaseSide","Release_Side","release_pos_x")
    if hand_col and hand_col in sub.columns:
        v = sub[hand_col].dropna().astype(str).str[0].str.upper()
        if not v.empty:
            return "R" if v.mode().iloc[0] == "R" else "L"
    if relside_col and relside_col in sub.columns:
        med = pd.to_numeric(sub[relside_col], errors="coerce").dropna().median()
        if pd.notna(med): return "R" if med >= 0 else "L"
    return "R"

def extensions_topN_figure(
    df: pd.DataFrame,
    pitcher_name: str,
    include_types=None,
    top_n: int = 3,
    figsize=(5.2, 7.0),
    title_size=14,
    show_plate: bool = False,
    xlim_pad: float = XLIM_PAD_DEFAULT,
    ylim_pad: float = YLIM_PAD_DEFAULT
):
    pitcher_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name","PitcherLastFirst") or "Pitcher"
    # Prefer TaggedPitchType, then fall back
    type_col    = pick_col(df, "TaggedPitchType","Tagged Pitch Type","Tagged_Type","PitchType","AutoPitchType","Auto Pitch Type") or "TaggedPitchType"
    ext_col     = pick_col(df, "Extension","Ext","ReleaseExtension","ExtensionInFt","Extension(ft)")
    if ext_col is None:
        st.warning("No Extension column found in data; skipping extensions plot.")
        return None

    sub = df[df[pitcher_col] == pitcher_name].copy()
    if sub.empty:
        st.warning("No data for selected pitcher.")
        return None

    sub[ext_col] = pd.to_numeric(sub[ext_col], errors="coerce")
    sub = sub.dropna(subset=[ext_col])
    if sub.empty:
        st.warning("No valid extension values for selected filters.")
        return None

    sub["_type_canon"] = sub[type_col].apply(canonicalize_type)
    sub = sub[sub["_type_canon"] != "Unknown"].copy()

    if include_types is not None and len(include_types) > 0:
        sub = sub[sub["_type_canon"].isin(include_types)]
    if sub.empty:
        st.warning("No pitches after applying pitch-type filter (extensions).")
        return None

    usage_top = sub["_type_canon"].value_counts().head(max(1, top_n)).index.tolist()
    mean_ext  = sub.groupby("_type_canon")[ext_col].mean().dropna()
    mean_sel  = mean_ext[mean_ext.index.isin(usage_top)].sort_values(ascending=False)
    entries   = list(mean_sel.items())
    if not entries:
        st.warning("Not enough data to compute top extensions.")
        return None

    hand = _decide_pitcher_hand(sub)

    fig, ax = plt.subplots(figsize=figsize)
    mound_radius  = 9.0
    rubber_len, rubber_wid = 2.0, 0.5

    ax.add_patch(Circle((0, 0), mound_radius, facecolor=MOUND_COLOR, edgecolor=MOUND_EDGE,
                        linewidth=1.0, alpha=MOUND_ALPHA, zorder=1))
    ax.add_patch(Rectangle((-rubber_len/2, -rubber_wid), rubber_len, rubber_wid,
                           linewidth=1.5, edgecolor="black", facecolor="white", zorder=5))

    throw_right = hand.upper() == "R"
    shoulder_x  = SHOULDER_X_R if throw_right else SHOULDER_X_L
    shoulder_y  = SHOULDER_Y

    handles, labels = [], []
    for i, (canon_name, ext_val) in enumerate(entries[:top_n]):
        clr   = color_for_release(canon_name)
        x_end = RELEASE_X_OFFSET_FT if throw_right else -RELEASE_X_OFFSET_FT
        y_end = float(ext_val)

        v     = np.array([x_end - shoulder_x, y_end - shoulder_y], dtype=float)
        v_len = np.hypot(v[0], v[1]) + 1e-9
        n     = v / v_len
        p     = np.array([-n[1], n[0]])
        sL    = (shoulder_x + p[0]*ARM_THICK_BASE, shoulder_y + p[1]*ARM_THICK_BASE)
        sR    = (shoulder_x - p[0]*ARM_THICK_BASE, shoulder_y - p[1]*ARM_THICK_BASE)
        tL    = (x_end + p[0]*ARM_THICK_TIP, y_end + p[1]*ARM_THICK_TIP)
        tR    = (x_end - p[0]*ARM_THICK_TIP, y_end - p[1]*ARM_THICK_TIP)
        ax.add_patch(Polygon([sL, sR, tR, tL], closed=True,
                             facecolor="#111111", edgecolor="#111111", alpha=0.9, zorder=6+i))

        ax.add_patch(Circle((x_end, y_end), radius=0.30, facecolor=clr, edgecolor=clr, zorder=7+i))
        ax.add_patch(Circle((x_end, y_end), radius=0.165, facecolor="#2b2b2b", edgecolor="#2b2b2b", zorder=8+i))

        handles.append(Line2D([0],[0], marker='o', linestyle='none', markersize=8,
                              markerfacecolor=clr, markeredgecolor=clr))
        labels.append(f"{canon_name} ({ext_val:.2f} ft)")

    max_ext = max([v for _, v in entries]) if entries else 7.0
    top_y   = max(9.0 + ylim_pad, max_ext + 2.0)
    ax.set_xlim(-xlim_pad, xlim_pad)
    ax.set_ylim(YLIM_BOTTOM, top_y)
    ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle(f"{format_name(pitcher_name)} Extension", fontsize=title_size, fontweight="bold", y=0.98)
    if handles:
        ax.legend(
            handles, labels, title="Pitch Type", loc=LEGEND_LOC,
            bbox_to_anchor=LEGEND_ANCHOR_FRAC, borderaxespad=0.0,
            frameon=True, fancybox=False, edgecolor="#d0d0d0"
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# PROFILES TABLES
# ──────────────────────────────────────────────────────────────────────────────
def _assign_spray_category_row(row):
    ang = row.get('Bearing', np.nan)
    side = str(row.get('BatterSide', "")).upper()[:1]
    if not pd.notna(ang): return np.nan
    ang = float(ang)
    if -15 <= ang <= 15: return "Straight"
    if ang < -15:        return "Pull" if side == "R" else "Opposite"
    return "Opposite" if side == "R" else "Pull"

def _bb_metrics_for_subset(sub_all: pd.DataFrame) -> dict:
    s_call = sub_all.get('PitchCall', pd.Series(dtype=object))
    inplay = sub_all[s_call == 'InPlay'].copy()
    if 'TaggedHitType' not in inplay.columns: inplay['TaggedHitType'] = pd.NA
    if 'Bearing' not in inplay.columns:       inplay['Bearing'] = np.nan
    if 'BatterSide' not in inplay.columns:    inplay['BatterSide'] = ""
    inplay['spray_cat'] = inplay.apply(_assign_spray_category_row, axis=1)
    tt = inplay['TaggedHitType'].astype(str).str.lower()
    def pct(mask):
        try:    return round(100 * float(np.nanmean(mask.astype(float))), 1) if len(mask) else 0.0
        except: return 0.0
    return {
        'Pitches': int(len(sub_all)),
        'Ground ball %': pct(tt.str.contains('groundball',   na=False)),
        'Fly ball %':    pct(tt.str.contains('flyball',      na=False)),
        'Line drive %':  pct(tt.str.contains('linedrive',    na=False)),
        'Popup %':       pct(tt.str.contains('popup',        na=False)),
        'Pull %':        pct(inplay['spray_cat'].astype(str).eq('Pull')),
        'Straight %':    pct(inplay['spray_cat'].astype(str).eq('Straight')),
        'Opposite %':    pct(inplay['spray_cat'].astype(str).eq('Opposite')),
    }

def make_pitcher_batted_ball_by_type(df: pd.DataFrame) -> pd.DataFrame:
    type_col = pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
    rows = [{'Pitch Type': 'Total', **_bb_metrics_for_subset(df)}]
    if type_col in df.columns:
        for ptype, sub in df.groupby(type_col, dropna=False):
            metrics = _bb_metrics_for_subset(sub)
            if metrics['Pitches'] >= 20:
                rows.append({'Pitch Type': str(ptype), **metrics})
    out = pd.DataFrame(rows)
    if len(out) > 1:
        total_row = out.iloc[[0]]
        others = out.iloc[1:].sort_values('Pitches', ascending=False)
        out = pd.concat([total_row, others], ignore_index=True)
    for c in out.columns:
        if c.endswith('%'): out[c] = out[c].astype(float).round(1)
    out['Pitches'] = out['Pitches'].astype(int)
    return out

def _plate_metrics(sub: pd.DataFrame) -> dict:
    s_call = sub.get('PitchCall', pd.Series(dtype=object))
    lside = pd.to_numeric(sub.get('PlateLocSide',   pd.Series(dtype=float)), errors="coerce")
    lht   = pd.to_numeric(sub.get('PlateLocHeight', pd.Series(dtype=float)), errors="coerce")
    isswing   = s_call.isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallNotFieldable','InPlay'])
    isswing   = s_call.isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
    iswhiff   = s_call.eq('StrikeSwinging')
    iscontact = s_call.isin(['InPlay','FoulBallNotFieldable','FoulBallFieldable'])
    isinzone  = lside.between(-0.83, 0.83) & lht.between(1.5, 3.5)
    total_pitches = int(len(sub))
    total_swings  = int(isswing.sum())
    z_count       = int(isinzone.sum())
    def pct(val):
        try:    return round(float(val) * 100, 1)
        except: return 0.0
    zone_pct  = pct(isinzone.mean()) if total_pitches else 0.0
    zone_sw   = pct(isswing[isinzone].mean()) if z_count else 0.0
    zone_ct   = pct((iscontact & isinzone).sum() / max(isswing[isinzone].sum(), 1)) if z_count else 0.0
    chase     = pct(isswing[~isinzone].mean()) if (~isinzone).sum() else 0.0
    swing_all = pct(total_swings / max(total_pitches, 1)) if total_pitches else 0.0
    whiff_pct = pct(iswhiff.sum() / max(total_swings, 1)) if total_swings else 0.0
    return {
        'Pitches': total_pitches,
        'Zone Pitches': z_count,
        'Zone %': zone_pct,
        'Zone Swing %': zone_sw,
        'Zone Contact %': zone_ct,
        'Chase %': chase,
        'Swing %': swing_all,
        'Whiff %': whiff_pct,
    }

def make_pitcher_plate_discipline_by_type(df: pd.DataFrame) -> pd.DataFrame:
    type_col = pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
    rows = [{'Pitch Type': 'Total', **_plate_metrics(df)}]
    if type_col in df.columns:
        for ptype, sub in df.groupby(type_col, dropna=False):
            metrics = _plate_metrics(sub)
            if metrics['Pitches'] >= 20:
                rows.append({'Pitch Type': str(ptype), **metrics})
    out = pd.DataFrame(rows)
    if len(out) > 1:
        total_row = out.iloc[[0]]
        others = out.iloc[1:].sort_values('Pitches', ascending=False)
        out = pd.concat([total_row, others], ignore_index=True)
    for c in out.columns:
        if c.endswith('%'): out[c] = out[c].astype(float).round(1)
    out['Pitches'] = out['Pitches'].astype(int)
    out['Zone Pitches'] = out['Zone Pitches'].astype(int)
    return out

def _strike_metrics(sub_df: pd.DataFrame) -> dict:
    def _safe_mask(q):
        for col in ("Balls","Strikes"):
            if col not in sub_df.columns:
                return sub_df.iloc[0:0]
        return q
    fp  = strike_rate(_safe_mask(sub_df[(sub_df.get('Balls',0)==0) & (sub_df.get('Strikes',0)==0)]))
    mix = strike_rate(_safe_mask(sub_df[((sub_df.get('Balls',0)==1)&(sub_df.get('Strikes',0)==0)) |
                                        ((sub_df.get('Balls',0)==0)&(sub_df.get('Strikes',0)==1)) |
                                        ((sub_df.get('Balls',0)==1)&(sub_df.get('Strikes',0)==1))]))
    hp  = strike_rate(_safe_mask(sub_df[((sub_df.get('Balls',0)==2)&(sub_df.get('Strikes',0)==0)) |
                                        ((sub_df.get('Balls',0)==2)&(sub_df.get('Strikes',0)==1)) |
                                        ((sub_df.get('Balls',0)==3)&(sub_df.get('Strikes',0)==1))]))
    two = strike_rate(_safe_mask(sub_df[(sub_df.get('Strikes',0)==2) & (sub_df.get('Balls',0)<3)]))
    return {'Pitches': int(len(sub_df)),
            '1st Pitch %': fp, 'Mix Count %': mix, 'Hitter+ %': hp, '2-Strike %': two}

def make_strike_percentage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Total first, then by pitch type (>=20 pitches), sorted by Pitches desc."""
    type_col = pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
    rows = [{'Pitch Type': 'Total', **_strike_metrics(df)}]
    if type_col in df.columns:
        for ptype, sub in df.groupby(type_col, dropna=False):
            metrics = _strike_metrics(sub)
            if metrics['Pitches'] >= 20:
                rows.append({'Pitch Type': str(ptype), **metrics})
    out = pd.DataFrame(rows)
    if len(out) > 1:
        total_row = out.iloc[[0]]
        others = out.iloc[1:].sort_values('Pitches', ascending=False)
        out = pd.concat([total_row, others], ignore_index=True)
    for c in out.columns:
        if c.endswith('%'): out[c] = out[c].astype(float).round(1)
    out['Pitches'] = out['Pitches'].astype(int)
    return out

def themed_table(df: pd.DataFrame):
    float_cols = df.select_dtypes(include=["float", "float64"]).columns.tolist()
    percent_cols = [c for c in df.columns if c.strip().endswith('%')]
    fmt_map = {c: "{:.1f}" for c in set(float_cols) | set(percent_cols)}
    styles = [
        {'selector': 'thead th', 'props': f'background-color: {HUSKER_RED}; color: white; white-space: nowrap; text-align: center;'},
        {'selector': 'th',        'props': f'background-color: {HUSKER_RED}; color: white; white-space: nowrap; text-align: center;'},
        {'selector': 'td',        'props': 'white-space: nowrap;'},
    ]
    return (df.style.hide(axis="index").format(fmt_map, na_rep="—").set_table_styles(styles))

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_csv_norm(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    return ensure_date_column(df)

def _load_scrimmage_df():
    for p in SCRIM_PATHS:
        if p and os.path.exists(p):
            return load_csv_norm(p), p
    return None, None

df_main = load_csv_norm(DATA_PATH_MAIN) if os.path.exists(DATA_PATH_MAIN) else None
df_scrim, SCRIM_USED_PATH = _load_scrimmage_df()

# ──────────────────────────────────────────────────────────────────────────────
# DATA SEGMENT PICKER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Data Segment")
segment_choice = st.selectbox(
    "Choose time period",
    list(SEGMENT_DEFS.keys()),
    index=0,
    key="segment_choice"
)

# Base dataset selection by segment
if segment_choice == "2025/26 Scrimmages":
    if df_scrim is None or df_scrim.empty:
        looked_for = ", ".join(SCRIM_PATHS)
        st.error(
            "Scrimmage data file not found or empty.\n\n"
            f"Looked for any of:\n{looked_for}"
        )
        st.stop()
    base_df = df_scrim
else:
    if df_main is None:
        st.error(f"Main data file not found at '{DATA_PATH_MAIN}'.")
        st.stop()
    base_df = df_main

df_segment = filter_by_segment(base_df, segment_choice)
if df_segment.empty:
    st.info(f"No rows found for **{segment_choice}** with the current dataset.")
    st.stop()

SEG_TYPES = SEGMENT_DEFS.get(segment_choice, {}).get("types", [])
is_bullpen_segment = "bullpen" in SEG_TYPES

# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────────
neb_df_all = df_segment[df_segment.get('PitcherTeam','') == 'NEB'].copy()
pitchers_all = sorted(neb_df_all.get('Pitcher', pd.Series(dtype=object)).dropna().unique().tolist())

st.markdown("### Pitcher Report")
player = st.selectbox("Pitcher", pitchers_all, key="neb_player_main") if pitchers_all else None

if not player:
    st.info("Select a pitcher to begin.")
    st.stop()

df_pitcher_all = neb_df_all[neb_df_all['Pitcher'] == player].copy()
df_pitcher_all['Date'] = pd.to_datetime(df_pitcher_all['Date'], errors="coerce")
appearances = int(df_pitcher_all['Date'].dropna().dt.date.nunique())
st.subheader(f"{format_name(player)} ({appearances} Appearances)")

tabs = st.tabs(["Standard", "Compare", "Profiles"])

# ── STANDARD TAB ───────────────────────────────────────────────────────────────
with tabs[0]:
    if segment_choice == "2025/26 Scrimmages":
        dates_all = sorted(df_pitcher_all['Date'].dropna().dt.date.unique().tolist())
        if not dates_all:
            st.info("No scrimmage dates available for this pitcher.")
            st.stop()
        default_idx = len(dates_all) - 1
        date_labels = [label_date_with_fb(d) for d in dates_all]
        sel_label = st.selectbox("Scrimmage Date", options=date_labels, index=default_idx, key="scrim_std_date")
        sel_date = dates_all[date_labels.index(sel_label)]
        neb_df = df_pitcher_all[pd.to_datetime(df_pitcher_all['Date']).dt.date == sel_date].copy()
        season_label = label_date_with_fb(sel_date)
    else:
        present_months = sorted(df_pitcher_all['Date'].dropna().dt.month.unique().tolist())
        col_m, col_d, _col_side = st.columns([1,1,1.6])
        months_sel = col_m.multiselect(
            "Months (optional)",
            options=present_months,
            format_func=lambda n: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][n-1],
            default=[],
            key="std_months",
        )
        dser = df_pitcher_all['Date'].dropna()
        if months_sel:
            dser = dser[dser.dt.month.isin(months_sel)]
        present_days = sorted(dser.dt.day.unique().tolist())
        days_sel = col_d.multiselect("Days (optional)", options=present_days, default=[], key="std_days")

        neb_df = filter_by_month_day(df_pitcher_all, months=months_sel, days=days_sel)
        season_label_base = build_pitcher_season_label(months_sel, days_sel, neb_df)
        season_label = f"{segment_choice} — {season_label_base}" if season_label_base else segment_choice

    if neb_df.empty:
        st.info("No rows for the selected filters.")
    else:
        logo_img = load_logo_img()

        # 1) Metrics (movement + summary) — prefer TaggedPitchType in STANDARD tab
        out = combined_pitcher_report(
            neb_df, player, logo_img, coverage=0.8, season_label=season_label, prefer_tagged=True
        )
        if out:
            fig_m, _ = out
            show_and_close(fig_m)

        # 2) Release Points (with its own filter) — build options from TaggedPitchType
        type_col_all = pick_col(neb_df, "TaggedPitchType","Tagged Pitch Type","Tagged_Type","PitchType","AutoPitchType","Auto Pitch Type") or "TaggedPitchType"
        types_available = (
            neb_df.get(type_col_all, pd.Series(dtype=object))
                  .dropna().map(canonicalize_type)
                  .replace("Unknown", np.nan).dropna().unique().tolist()
        )
        types_available = sorted(types_available)
        st.markdown("### Release Points")
        rel_selected = pitchtype_checkbox_grid(
            "Pitch Types (Release Points)",
            options=types_available,
            key_prefix="std_release_types",
            default_all=True,
            columns_per_row=6,
        )
        rel_fig = release_points_figure(neb_df, player, include_types=rel_selected)
        if rel_fig:
            show_and_close(rel_fig)

        # 3) Extensions (own pitch-type filter) — build options from TaggedPitchType
        st.markdown("### Extensions")
        ext_selected = pitchtype_checkbox_grid(
            "Pitch Types (Extensions)",
            options=types_available,
            key_prefix="std_ext_types",
            default_all=True,
            columns_per_row=6,
        )
        ext_fig = extensions_topN_figure(
            neb_df, player, include_types=ext_selected, top_n=3,
            figsize=(5.2, 7.0), title_size=14, show_plate=False
        )
        if ext_fig:
            show_image_scaled(ext_fig, width_px=EXT_VIS_WIDTH, dpi=200, pad_inches=0.1)

# ── COMPARE TAB ────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Compare Appearances")
    cmp_n = st.selectbox("Number of windows", [2,3], index=0, key="cmp_n")
    expand_view = st.checkbox("Expand compare view (full-width)", value=False, key="cmp_expand")

    # Define windows (date selection for scrimmages; month/day ranges otherwise)
    if segment_choice == "2025/26 Scrimmages":
        dates_all_cmp = sorted(df_pitcher_all['Date'].dropna().dt.date.unique().tolist())
        if not dates_all_cmp:
            st.info("No scrimmage dates available.")
            st.stop()
        date_labels_all = [label_date_with_fb(d) for d in dates_all_cmp]
        default_idx = len(dates_all_cmp) - 1

        cols_filters = st.columns(cmp_n)
        windows = []
        for i in range(cmp_n):
            with cols_filters[i]:
                lab = st.selectbox(
                    f"Scrimmage Date (Window {'ABC'[i]})",
                    options=date_labels_all, index=default_idx, key=f"cmp_scrim_date_{i}"
                )
                chosen = dates_all_cmp[date_labels_all.index(lab)]
                df_win = df_pitcher_all[pd.to_datetime(df_pitcher_all['Date']).dt.date == chosen]
                season_lab = label_date_with_fb(chosen)
                windows.append((season_lab, df_win))
    else:
        date_ser_all = df_pitcher_all['Date'].dropna()
        month_options = sorted(date_ser_all.dt.month.unique().tolist())
        cols_filters = st.columns(cmp_n)
        windows = []
        for i in range(cmp_n):
            with cols_filters[i]:
                st.markdown(f"**Window {'ABC'[i]} Filters**")
                mo_sel = st.multiselect(
                    f"Months (Window {'ABC'[i]})",
                    options=month_options,
                    format_func=lambda n: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][n-1],
                    key=f"cmp_months_{i}"
                )
                dser = date_ser_all
                if mo_sel:
                    dser = dser[dser.dt.month.isin(mo_sel)]
                day_opts = sorted(dser.dt.day.unique().tolist())
                dy_sel = st.multiselect(f"Days (Window {'ABC'[i]})", options=day_opts, key=f"cmp_days_{i}")
                df_win = filter_by_month_day(df_pitcher_all, months=mo_sel, days=dy_sel)
                season_lab = build_pitcher_season_label(mo_sel, dy_sel, df_win)
                season_lab = f"{segment_choice} — {season_lab}" if season_lab else segment_choice
                windows.append((season_lab, df_win))

    # Common type options for filters (prefer TaggedPitchType)
    type_col_all = pick_col(df_pitcher_all, "TaggedPitchType","Tagged Pitch Type","Tagged_Type","PitchType","AutoPitchType","Auto Pitch Type") or "TaggedPitchType"
    types_avail_canon = (
        df_pitcher_all.get(type_col_all, pd.Series(dtype=object))
                      .dropna().map(canonicalize_type)
                      .replace("Unknown", np.nan).dropna().unique().tolist()
    )
    types_avail_canon = sorted(types_avail_canon)

    # 1) MOVEMENT
    st.markdown("### Movement")
    cols_out = st.columns(cmp_n)
    logo_img = load_logo_img()
    for i, (season_lab, df_win) in enumerate(windows):
        with cols_out[i]:
            st.markdown(f"**Window {'ABC'[i]} — {season_lab}**")
            if df_win.empty:
                st.info("No data for this window."); continue
            out_win = combined_pitcher_report(df_win, player, logo_img, coverage=0.8, season_label=season_lab)
            if out_win:
                fig_m, _ = out_win
                show_and_close(fig_m, use_container_width=expand_view)

    # 2) RELEASE POINTS (filter above)
    st.markdown("### Release Points")
    cmp_types_selected = pitchtype_checkbox_grid(
        "Pitch Types (Release Points)",
        options=types_avail_canon,
        key_prefix="cmp_rel_types",
        default_all=True,
        columns_per_row=6,
    )
    cols_rel = st.columns(cmp_n)
    for i, (season_lab, df_win) in enumerate(windows):
        with cols_rel[i]:
            if df_win.empty:
                st.info("No data for this window."); continue
            fig_r = release_points_figure(df_win, player, include_types=cmp_types_selected)
            if fig_r:
                show_and_close(fig_r, use_container_width=expand_view)

    # 3) EXTENSIONS (filter above)
    st.markdown("### Extensions")
    cmp_ext_types_selected = pitchtype_checkbox_grid(
        "Pitch Types (Extensions)",
        options=types_avail_canon,
        key_prefix="cmp_ext_types",
        default_all=True,
        columns_per_row=6,
    )
    cols_ext = st.columns(cmp_n)
    ext_width = 1000 if expand_view else EXT_VIS_WIDTH
    for i, (season_lab, df_win) in enumerate(windows):
        with cols_ext[i]:
            if df_win.empty:
                st.info("No data for this window."); continue
            ext_fig = extensions_topN_figure(
                df_win, player, include_types=cmp_ext_types_selected, top_n=3,
                figsize=(5.2, 7.0), title_size=14, show_plate=False
            )
            if ext_fig:
                show_image_scaled(ext_fig, width_px=ext_width, dpi=200, pad_inches=0.1)

    # 4) HEATMAPS (filters above)
    st.markdown("### Heatmaps")
    cmp_hand = st.radio("Batter Side (Heatmaps)", ["Both","LHH","RHH"], index=0, horizontal=True, key="cmp_hand")
    types_avail_outcomes = sorted(df_pitcher_all.get(type_col_all, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
    cmp_types_out_selected = pitchtype_checkbox_grid(
        "Pitch Types (Whiffs/Strikeouts/Damage)",
        options=types_avail_outcomes,
        key_prefix="cmp_types_outcomes",
        default_all=True,
        columns_per_row=6,
    )
    cols_hm = st.columns(cmp_n)
    for i, (season_lab, df_win) in enumerate(windows):
        with cols_hm[i]:
            if df_win.empty:
                st.info("No data for this window."); continue
            fig_h = combined_pitcher_heatmap_report(
                df_win, player, hand_filter=cmp_hand, season_label=season_lab,
                outcome_pitch_types=cmp_types_out_selected,
            )
            if fig_h:
                show_and_close(fig_h, use_container_width=expand_view)

# ── PROFILES TAB ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### Pitcher Profiles")
    if is_bullpen_segment:
        st.info("Profiles are not available for **Bullpens** (no batting occurs).")
    else:
        if segment_choice == "2025/26 Scrimmages":
            dates_all_prof = sorted(df_pitcher_all['Date'].dropna().dt.date.unique().tolist())
            if not dates_all_prof:
                st.info("No scrimmage dates available."); st.stop()
            default_idx = len(dates_all_prof) - 1
            date_labels_prof = [label_date_with_fb(d) for d in dates_all_prof]
            lab = st.selectbox("Scrimmage Date (Profiles)", options=date_labels_prof, index=default_idx, key="prof_scrim_date")
            chosen = dates_all_prof[date_labels_prof.index(lab)]
            df_prof = df_pitcher_all[pd.to_datetime(df_pitcher_all['Date']).dt.date == chosen].copy()
            season_label_prof = label_date_with_fb(chosen)
            prof_hand = st.radio("Batter Side", ["Both","LHH","RHH"], index=0, horizontal=True, key="prof_hand")
            side_col = find_batter_side_col(df_prof)
            if prof_hand in ("LHH","RHH") and side_col is not None and not df_prof.empty:
                sides = normalize_batter_side(df_prof[side_col])
                target = "L" if prof_hand == "LHH" else "R"
                df_prof = df_prof[sides == target].copy()
        else:
            prof_months_all = sorted(df_pitcher_all['Date'].dropna().dt.month.unique().tolist())
            col_pm, col_pd, col_ln, col_side = st.columns([1,1,1,1.4])
            prof_months = col_pm.multiselect(
                "Months (optional)",
                options=prof_months_all,
                format_func=lambda n: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][n-1],
                default=[],
                key="prof_months"
            )
            dser_prof = df_pitcher_all['Date'].dropna()
            if prof_months:
                dser_prof = dser_prof[dser_prof.dt.month.isin(prof_months)]
            prof_days_all = sorted(dser_prof.dt.day.unique().tolist())
            prof_days = col_pd.multiselect("Days (optional)", options=prof_days_all, default=[], key="prof_days")
            last_n_games = int(col_ln.number_input("Last N games", min_value=0, max_value=50, value=0, step=1, format="%d", key="prof_lastn"))
            prof_hand = col_side.radio("Batter Side", ["Both","LHH","RHH"], index=0, horizontal=True, key="prof_hand")
            df_prof = filter_by_month_day(df_pitcher_all, months=prof_months, days=prof_days).copy()
            if last_n_games and not df_prof.empty:
                uniq_dates = pd.to_datetime(df_prof['Date'], errors="coerce").dt.date.dropna().unique()
                uniq_dates = sorted(uniq_dates)
                last_dates = set(uniq_dates[-last_n_games:])
                df_prof = df_prof[pd.to_datetime(df_prof['Date'], errors="coerce").dt.date.isin(last_dates)].copy()
            season_label_prof_base = build_pitcher_season_label(prof_months, prof_days, df_prof)
            season_label_prof = f"{segment_choice} — {season_label_prof_base}" if season_label_prof_base else segment_choice
            side_col = find_batter_side_col(df_prof)
            if prof_hand in ("LHH","RHH") and side_col is not None and not df_prof.empty:
                sides = normalize_batter_side(df_prof[side_col])
                target = "L" if prof_hand == "LHH" else "R"
                df_prof = df_prof[sides == target].copy()

        if df_prof.empty:
            st.info("No rows for the selected profile filters.")
        else:
            # 1) Strike Percentage by Count — Total first, then by pitch type
            st.markdown("### Strike Percentage by Count")
            strike_df = make_strike_percentage_table(df_prof).round(1)
            st.table(themed_table(strike_df))

            # 2) Batted Ball Profile (title cleaned)
            bb_df_typed = make_pitcher_batted_ball_by_type(df_prof)
            st.markdown(f"### Batted Ball Profile — {segment_choice}")
            st.table(themed_table(bb_df_typed))

            # 3) Plate Discipline Profile (title cleaned)
            pd_df_typed = make_pitcher_plate_discipline_by_type(df_prof)
            st.markdown(f"### Plate Discipline Profile — {segment_choice}")
            st.table(themed_table(pd_df_typed))

            # 4) Top-3 Pitches (title cleaned)
            st.markdown("### Top 3 Pitches")
            fig_top3 = heatmaps_top3_pitch_types(
                df_prof, player, hand_filter=prof_hand, season_label=season_label_prof
            )
            if fig_top3:
                show_and_close(fig_top3)

            # 5) Whiffs / Strikeouts / Damage (title cleaned)
            st.markdown("### Whiffs / Strikeouts / Damage")
            type_col_for_hm = pick_col(df_prof, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
            types_available_hm = (
                df_prof.get(type_col_for_hm, pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
            )
            types_available_hm = sorted(types_available_hm)
            hm_selected = pitchtype_checkbox_grid(
                "Filter Whiffs / Strikeouts / Damage by Pitch Type",
                options=types_available_hm,
                key_prefix="prof_hm_types",
                default_all=True,
                columns_per_row=6,
            )
            fig_outcomes = heatmaps_outcomes(
                df_prof, player, hand_filter=prof_hand, season_label=season_label_prof,
                outcome_pitch_types=hm_selected,
            )
            if fig_outcomes:
                show_and_close(fig_outcomes)
