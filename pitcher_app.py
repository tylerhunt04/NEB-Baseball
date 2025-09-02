# pitcher_app.py (UPDATED)
# - Heatmaps moved to Profiles tab
# - New Extensions figure added under Release Points in Standard tab
# - Extensions plot respects the same pitch-type filter as Release Points

import os
import gc
import re
import base64
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive, memory-safe backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from matplotlib.patches import Rectangle, Ellipse, Polygon, Circle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2, gaussian_kde
from numpy.linalg import LinAlgError
from matplotlib import colors

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & ERROR DETAILS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nebraska Baseball — Pitcher Reports",
    layout="wide",  # wide mode ON
    initial_sidebar_state="collapsed",
)
st.set_option("client.showErrorDetails", True)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "pitcher_columns.csv"  # <-- your CSV
LOGO_PATH = "Nebraska-Cornhuskers-Logo.png"
BANNER_IMG = "NebraskaChampions.jpg"  # uploaded earlier
HUSKER_RED = "#E60026"

# ──────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS & FIGURE CLEANUP
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

def show_and_close(fig):
    """Display a Matplotlib figure and free its memory immediately."""
    try:
        st.pyplot(fig=fig, clear_figure=False)
    finally:
        plt.close(fig)
        gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# SIMPLE HERO BANNER (image + overlayed title)
# ──────────────────────────────────────────────────────────────────────────────

def hero_banner(title: str, *, subtitle: str | None = None, height_px: int = 260):
    b64 = load_banner_b64()
    bg_url = f"data:image/jpeg;base64,{b64}" if b64 else ""
    sub_html = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
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
        .hero-text {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
            flex-direction: column; color: #fff; text-align: center; }}
        .hero-title {{ font-size: 40px; font-weight: 800; letter-spacing: .5px; text-shadow: 0 2px 8px rgba(0,0,0,.45); margin: 0; }}
        .hero-sub {{ font-size: 18px; font-weight: 600; opacity: .95; margin-top: 6px; }}
        </style>
        <div class="hero-wrap">
            <div class="hero-bg"></div>
            <div class="hero-text">
                <h1 class="hero-title">{title}</h1>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render banner (ONLY “Nebraska Baseball” — no subtitle)
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
# SEGMENTS (Season/Scrimmage/Bullpen) — detection + filtering
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
    """Return one of: 'game', 'scrimmage', 'bullpen', or ''."""
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

# Adjust these windows/types to your calendar as needed
SEGMENT_DEFS = {
    "2025 Season": {
        "start": "2025-02-01",
        "end": "2025-08-01",
        "types": ["game"],
    },
    "2025/26 Scrimmages": {
        "start": "2025-08-01",
        "end": "2026-02-01",
        "types": ["scrimmage"],
    },
    "2025/26 Bullpens": {
        "start": "2025-08-01",
        "end": "2026-02-01",
        "types": ["bullpen"],
    },
    "2026 Season": {
        "start": "2026-02-01",
        "end": "2026-08-01",
        "types": ["game"],
    },
}

def filter_by_segment(df: pd.DataFrame, segment_name: str) -> pd.DataFrame:
    """Filter by date window and (if available) session type."""
    spec = SEGMENT_DEFS.get(segment_name)
    if spec is None or df.empty:
        return df
    out = df.copy()
    # Date filter
    if "Date" in out.columns:
        d = pd.to_datetime(out["Date"], errors="coerce")
        start = pd.to_datetime(spec["start"])
        end = pd.to_datetime(spec["end"])
        out = out[(d >= start) & (d < end)]
    # Session-type filter (optional)
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
    sz_left = l if sz_left is None else sz_left
    sz_bottom = b if sz_bottom is None else sz_bottom
    sz_width = w if sz_width is None else sz_width
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
# DENSITY / UTILS
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
    if len(df) == 0:
        return np.nan
    if "PitchCall" not in df.columns:
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
    if s in {"l", "lhh", "lhb", "left", "left-handed", "left handed"}:
        return "L"
    if s in {"r", "rhh", "rhb", "right", "right-handed", "right handed"}:
        return "R"
    return None

# ──────────────────────────────────────────────────────────────────────────────
# UI HELPER: grid of per-pitch checkboxes (no dropdown) + Select/Clear all
# ──────────────────────────────────────────────────────────────────────────────

def _safe_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s))

def pitchtype_checkbox_grid(label: str, options: list[str], key_prefix: str, default_all=True, columns_per_row=6) -> list[str]:
    """
    Renders a label, then a 'Select all' and 'Clear all' button row, followed by a grid of checkboxes (no dropdown).
    Returns the list of selected pitch types.
    """
    options = list(dict.fromkeys([str(o) for o in options]))  # unique, stable order
    if not options:
        st.caption("No pitch types available.")
        return []
    # init per-option states
    opt_keys = [f"{key_prefix}_{_safe_key(o)}" for o in options]
    for k in opt_keys:
        if k not in st.session_state:
            st.session_state[k] = bool(default_all)

    st.write(f"**{label}**")
    col_a, col_b = st.columns([0.12, 0.12])
    if col_a.button("Select all", key=f"{key_prefix}_select_all"):
        for k in opt_keys:
            st.session_state[k] = True
    if col_b.button("Clear all", key=f"{key_prefix}_clear_all"):
        for k in opt_keys:
            st.session_state[k] = False

    cols = st.columns(columns_per_row)
    for i, (o, k) in enumerate(zip(options, opt_keys)):
        col = cols[i % columns_per_row]
        col.checkbox(o, value=st.session_state[k], key=k)

    selected = [o for o, k in zip(options, opt_keys) if st.session_state[k]]
    return selected

# ──────────────────────────────────────────────────────────────────────────────
# PITCHER REPORT (movement + summary)
# ──────────────────────────────────────────────────────────────────────────────

def combined_pitcher_report(df, pitcher_name, logo_img, coverage=0.8, season_label="Season"):
    # Resolve core columns
    type_col = pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
    pitch_col = pick_col(df, "PitchCall","Pitch Call","Call") or "PitchCall"
    speed_col = pick_col(df, "RelSpeed","Relspeed","ReleaseSpeed","RelSpeedMPH","release_speed")
    spin_col = pick_col(df, "SpinRate","Spinrate","ReleaseSpinRate","Spin")
    ivb_col = pick_col(df, "InducedVertBreak","IVB","Induced Vert Break","IndVertBreak")
    hb_col = pick_col(df, "HorzBreak","HorizontalBreak","HB","HorizBreak")
    rh_col = pick_col(df, "RelHeight","Relheight","ReleaseHeight","Release_Height","release_pos_z")
    vaa_col = pick_col(df, "VertApprAngle","VAA","VerticalApproachAngle")
    ext_col = pick_col(df, "Extension","Ext","ReleaseExtension","ExtensionInFt","Extension(ft)")

    df_p = df[df.get('Pitcher', '') == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' with the current filters.")
        return None

    # group
    try:
        grp = df_p.groupby(type_col, dropna=False)
    except KeyError:
        st.error(f"Pitch type column not found (tried '{type_col}').")
        return None

    counts = grp.size()
    total = int(len(df_p))

    # Base summary frame
    summary = pd.DataFrame({
        'Pitch Type': counts.index.astype(str),
        'Pitches': counts.values,
        'Usage %': np.round((counts.values / max(total, 1)) * 100, 1),
    })

    # Strike %
    if pitch_col in df_p.columns:
        is_strike = df_p[pitch_col].isin(['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
        strike_pct = grp.apply(lambda g: is_strike.loc[g.index].mean() * 100 if len(g) else np.nan).values
        summary['Strike %'] = np.round(strike_pct, 1)

    # Helper to add mean column if present
    def add_mean(col_name, label, r=1):
        nonlocal summary
        if col_name and col_name in df_p.columns:
            vals = grp[col_name].mean().values
            summary[label] = np.round(vals, r)

    add_mean(speed_col, 'Rel Speed', r=1)
    add_mean(spin_col, 'Spin Rate', r=1)
    add_mean(ivb_col, 'IVB', r=1)
    add_mean(hb_col, 'HB', r=1)
    add_mean(rh_col, 'Rel Height', r=2)
    add_mean(vaa_col, 'VAA', r=1)
    add_mean(ext_col, 'Extension', r=2)

    # sort
    summary = summary.sort_values('Pitches', ascending=False)

    # ── Figure: movement + table
    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.7], hspace=0.3)

    # Movement plot
    axm = fig.add_subplot(gs[0, 0]); axm.set_title('Movement Plot', fontweight='bold')
    axm.axhline(0, ls='--', color='grey'); axm.axvline(0, ls='--', color='grey')
    chi2v = chi2.ppf(coverage, df=2)

    for ptype, g in df_p.groupby(type_col, dropna=False):
        clr = get_pitch_color(ptype)
        x_series = g[hb_col] if hb_col in g.columns else pd.Series([np.nan]*len(g))
        y_series = g[ivb_col] if ivb_col in g.columns else pd.Series([np.nan]*len(g))
        x = pd.to_numeric(x_series, errors='coerce')
        y = pd.to_numeric(y_series, errors='coerce')
        mask = x.notna() & y.notna()
        if mask.any():
            axm.scatter(x[mask], y[mask], label=str(ptype), color=clr, alpha=0.7)
            if mask.sum() > 1:
                X = np.vstack((x[mask], y[mask]))
                cov = np.cov(X)
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

    # Summary table
    axt = fig.add_subplot(gs[1, 0]); axt.axis('off')
    tbl = axt.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Summary Metrics', fontweight='bold', y=0.87)

    # Logo
    logo_img = load_logo_img()
    if logo_img is not None:
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10); axl.imshow(logo_img); axl.axis('off')

    fig.suptitle(f"{format_name(pitcher_name)} Metrics\n({season_label})", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, summary

# ──────────────────────────────────────────────────────────────────────────────
# PITCHER HEATMAPS (top 3 + whiffs/K/damage)
# ──────────────────────────────────────────────────────────────────────────────

def combined_pitcher_heatmap_report(
    df, pitcher_name, hand_filter="Both", grid_size=100, season_label="Season", outcome_pitch_types=None,
):
    df_p = df[df.get('Pitcher','') == pitcher_name].copy()
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' with the current filters.")
        return None

    type_col = pick_col(df_p, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"

    # Batter-side filter
    side_col = find_batter_side_col(df_p)
    hand_label = "Both"
    if side_col is not None:
        sides = normalize_batter_side(df_p[side_col])
        want = parse_hand_filter_to_LR(hand_filter)
        if want == "L":
            df_p = df_p[sides == "L"]; hand_label = "LHH"
        elif want == "R":
            df_p = df_p[sides == "R"]; hand_label = "RHH"
        else:
            hand_label = "Both"
    else:
        st.caption("Batter-side column not found; showing Both.")

    if df_p.empty:
        st.info("No pitches for the selected batter-side filter.")
        return None

    # heatmap canvas
    x_min, x_max, y_min, y_max = get_view_bounds()
    xi = np.linspace(x_min, x_max, grid_size); yi = np.linspace(y_min, y_max, grid_size)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi); grid_coords = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()])
    z_left, z_bottom, z_w, z_h = get_zone_bounds()
    threshold = 12

    def panel(ax, sub, title, color='deepskyblue'):
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
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.6], hspace=0.35, wspace=0.3)

    # Top 3 pitch types
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
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("—", fontweight='bold')

    # Limit outcome panels (whiffs/K/damage) to selected pitch types if provided (list may be empty)
    df_out = df_p
    if outcome_pitch_types is not None:
        if len(outcome_pitch_types) == 0:
            df_out = df_p.iloc[0:0].copy()
        else:
            if type_col in df_p.columns:
                df_out = df_p[df_p[type_col].astype(str).isin(list(outcome_pitch_types))].copy()

    # Outcome panels
    sub_wh = df_out[df_out.get('PitchCall','') == 'StrikeSwinging']
    sub_ks = df_out[df_out.get('KorBB','') == 'Strikeout']
    sub_dg = df_out[pd.to_numeric(df_out.get('ExitSpeed', pd.Series(dtype=float)), errors='coerce') >= 95]
    ax = fig.add_subplot(gs[1, 0]); panel(ax, sub_wh, f"Whiffs (n={len(sub_wh)})")
    ax = fig.add_subplot(gs[1, 1]); panel(ax, sub_ks, f"Strikeouts (n={len(sub_ks)})")
    ax = fig.add_subplot(gs[1, 2]); panel(ax, sub_dg, f"Damage (n={len(sub_dg)})", color='orange')

    # Strike % by count
    axt = fig.add_subplot(gs[2, :]); axt.axis('off')

    def _safe_mask(q):
        for col in ("Balls","Strikes"):
            if col not in df_p.columns:
                return df_p.iloc[0:0]
        return q

    fp  = strike_rate(_safe_mask(df_p[(df_p.get('Balls',0)==0) & (df_p.get('Strikes',0)==0)]))
    mix = strike_rate(_safe_mask(df_p[((df_p.get('Balls',0)==1)&(df_p.get('Strikes',0)==0)) | ((df_p.get('Balls',0)==0)&(df_p.get('Strikes',0)==1)) | ((df_p.get('Balls',0)==1)&(df_p.get('Strikes',0)==1))]))
    hp  = strike_rate(_safe_mask(df_p[((df_p.get('Balls',0)==2)&(df_p.get('Strikes',0)==0)) | ((df_p.get('Balls',0)==2)&(df_p.get('Strikes',0)==1)) | ((df_p.get('Balls',0)==3)&(df_p.get('Strikes',0)==1))]))
    two = strike_rate(_safe_mask(df_p[(df_p.get('Strikes',0)==2) & (df_p.get('Balls',0)<3)]))

    metrics = pd.DataFrame({'1st Pitch %':[fp],'Mix Count %':[mix],'Hitter+ %':[hp],'2-Strike %':[two]}).round(1)
    tbl = axt.table(cellText=metrics.values, colLabels=metrics.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Strike Percentage by Count', y=0.75, fontweight='bold')

    logo_img = load_logo_img()
    if logo_img is not None:
        axl = fig.add_axes([0.88, 0.92, 0.10, 0.10], anchor='NE', zorder=10); axl.imshow(logo_img); axl.axis('off')

    fig.suptitle(
        f"{format_name(pitcher_name)} Heatmaps\n({season_label}) ({hand_label})",
        fontsize=18, y=0.98, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# RELEASE POINTS (with pitch-type checkbox grid filter)
# ──────────────────────────────────────────────────────────────────────────────
ARM_BASE_HALF_WIDTH = 0.24
ARM_TIP_HALF_WIDTH  = 0.08
SHOULDER_RADIUS_OUT = 0.20
HAND_RING_OUTER_R   = 0.26
HAND_RING_INNER_R   = 0.15
ARM_FILL_COLOR      = "#111111"

def norm_text(x: str) -> str:
    return str(x).strip().lower()

def norm_type(x: str) -> str:
    s = norm_text(x)
    replacements = {"four seam":"four-seam","4 seam":"4-seam","two seam":"two-seam","2 seam":"2-seam"}
    return replacements.get(s, s)

def canonicalize_type(raw: str) -> str:
    s = norm_text(raw)
    if "sinker" in s or s in {"si","snk"}:
        return "Fastball"
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
    type_col = pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
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
        if L <= 1e-6:
            continue
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
    ax2.set_xlabel(x_col)
    ax2.set_title("Average Releases", fontweight="bold")

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
# NEW: EXTENSIONS FIGURE (Top-3 by usage, ordered by mean extension desc)
# ──────────────────────────────────────────────────────────────────────────────

# Visual config
RELEASE_X_OFFSET_FT = 2.0
SHOULDER_X_R = 0.7
SHOULDER_X_L = -0.7
SHOULDER_Y = 1.6
ARM_THICK_BASE = 0.28
ARM_THICK_TIP  = 0.05
ANGLE_BASE_DEG = 12.0
ANGLE_FAN_DEG  = 10.0
MOUND_COLOR    = "#7B4B24"
MOUND_EDGE     = "#3D2A1A"
MOUND_ALPHA    = 0.90
LEGEND_ANCHOR_FRAC = (0.50, 0.22)  # x_frac, y_frac
LEGEND_LOC     = "lower center"

def _decide_pitcher_hand(sub: pd.DataFrame) -> str:
    hand_col = pick_col(sub, "PitcherThrows","Throws","PitcherHand","Pitcher_Hand","ThrowHand")
    relside_col = pick_col(sub, "Relside","RelSide","ReleaseSide","Release_Side","release_pos_x")
    if hand_col and hand_col in sub.columns:
        v = sub[hand_col].dropna().astype(str).str[0].str.upper()
        if not v.empty:
            return "R" if v.mode().iloc[0] == "R" else "L"
    if relside_col and relside_col in sub.columns:
        med = pd.to_numeric(sub[relside_col], errors="coerce").dropna().median()
        if pd.notna(med):
            return "R" if med >= 0 else "L"
    return "R"

def extensions_topN_figure(df: pd.DataFrame, pitcher_name: str, include_types=None, top_n: int = 3):
    pitcher_col = pick_col(df, "Pitcher","PitcherName","Pitcher Full Name","Name","PitcherLastFirst") or "Pitcher"
    type_col = pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
    ext_col  = pick_col(df, "Extension","Ext","ReleaseExtension","ExtensionInFt","Extension(ft)")

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

    # Respect pitch-type filter (same as release figure)
    if include_types is not None and len(include_types) > 0:
        sub = sub[sub["_type_canon"].isin(include_types)]
    if sub.empty:
        st.warning("No pitches after applying pitch-type filter (extensions).")
        return None

    # Pick top-N by usage, THEN order by mean extension desc
    usage_top = sub["_type_canon"].value_counts().head(max(1, top_n)).index.tolist()
    mean_ext  = sub.groupby("_type_canon")[ext_col].mean().dropna()
    mean_sel  = mean_ext[mean_ext.index.isin(usage_top)].sort_values(ascending=False)
    entries   = list(mean_sel.items())
    if len(entries) == 0:
        st.warning("Not enough data to compute top extensions.")
        return None

    hand = _decide_pitcher_hand(sub)

    # Draw
    fig, ax = plt.subplots(figsize=(7.5, 10))
    mound_radius = 9.0
    rubber_len, rubber_wid = 2.0, 0.5
    distance_plate = 60 + 6/12

    # Mound (darker brown) & rubber
    ax.add_patch(Circle((0, 0), mound_radius, facecolor=MOUND_COLOR, edgecolor=MOUND_EDGE,
                        linewidth=1.0, alpha=MOUND_ALPHA, zorder=1))
    ax.add_patch(Rectangle((-rubber_len/2, -rubber_wid), rubber_len, rubber_wid,
                           linewidth=1.5, edgecolor="black", facecolor="white", zorder=5))

    # Home plate — point AWAY from mound (flat edge toward mound)
    plate_width = 17/12
    point_depth = 8.5/12
    plate_y = distance_plate
    plate = [
        (-plate_width/2, plate_y),
        ( plate_width/2, plate_y),
        ( plate_width/2, plate_y + point_depth),
        ( 0, plate_y + 2*point_depth),  # point toward backstop
        (-plate_width/2, plate_y + point_depth),
    ]
    ax.add_patch(Polygon(plate, closed=True, facecolor="white", edgecolor="black", zorder=3))

    # Shoulder anchor
    throw_right = hand.upper() == "R"
    shoulder_x = SHOULDER_X_R if throw_right else SHOULDER_X_L
    shoulder_y = SHOULDER_Y

    handles, labels = [], []
    for i, (canon_name, ext_val) in enumerate(entries[:top_n]):
        clr = color_for_release(canon_name)
        x_end = RELEASE_X_OFFSET_FT if throw_right else -RELEASE_X_OFFSET_FT
        y_end = float(ext_val)

        # Arm polygon oriented from shoulder → (x_end, y_end)
        v = np.array([x_end - shoulder_x, y_end - shoulder_y], dtype=float)
        v_len = np.hypot(v[0], v[1]) + 1e-9
        n = v / v_len
        p = np.array([-n[1], n[0]])
        sL = (shoulder_x + p[0]*ARM_THICK_BASE, shoulder_y + p[1]*ARM_THICK_BASE)
        sR = (shoulder_x - p[0]*ARM_THICK_BASE, shoulder_y - p[1]*ARM_THICK_BASE)
        tL = (x_end + p[0]*ARM_THICK_TIP, y_end + p[1]*ARM_THICK_TIP)
        tR = (x_end - p[0]*ARM_THICK_TIP, y_end - p[1]*ARM_THICK_TIP)
        ax.add_patch(Polygon([sL, sR, tR, tL], closed=True, facecolor="#111111", edgecolor="#111111", alpha=0.9, zorder=6+i))

        # Colored ball at the tip
        ax.add_patch(Circle((x_end, y_end), radius=0.30, facecolor=clr, edgecolor=clr, zorder=7+i))
        ax.add_patch(Circle((x_end, y_end), radius=0.30*0.55, facecolor="#2b2b2b", edgecolor="#2b2b2b", zorder=8+i))

        # Legend
        handles.append(Line2D([0],[0], marker='o', linestyle='none', markersize=8, markerfacecolor=clr, markeredgecolor=clr))
        labels.append(f"{canon_name} ({ext_val:.2f} ft)")

    # Clean diagram
    ax.set_xlim(-10, 12)
    ax.set_ylim(-5, distance_plate+5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Suptitle
    fig.suptitle(f"{format_name(pitcher_name)} Extension", fontsize=16, fontweight="bold", y=0.98)

    # Legend ABOVE mound
    if handles:
        ax.legend(handles, labels, title="Pitch Type", loc=LEGEND_LOC,
                  bbox_to_anchor=LEGEND_ANCHOR_FRAC, borderaxespad=0.0,
                  frameon=True, fancybox=False, edgecolor="#d0d0d0")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# PITCHER PROFILES (Batted Ball & Plate Discipline by Pitch Type)
# ──────────────────────────────────────────────────────────────────────────────

def _assign_spray_category_row(row):
    ang = row.get('Bearing', np.nan)
    side = str(row.get('BatterSide', "")).upper()[:1]
    if not pd.notna(ang):
        return np.nan
    ang = float(ang)
    if -15 <= ang <= 15:
        return "Straight"
    if ang < -15:
        return "Pull" if side == "R" else "Opposite"
    return "Opposite" if side == "R" else "Pull"

def _bb_metrics_for_subset(sub_all: pd.DataFrame) -> dict:
    """Compute batted-ball % metrics for a subset; returns dict with 1-decimal rounding."""
    s_call = sub_all.get('PitchCall', pd.Series(dtype=object))
    inplay = sub_all[s_call == 'InPlay'].copy()
    if 'TaggedHitType' not in inplay.columns:
        inplay['TaggedHitType'] = pd.NA
    if 'Bearing' not in inplay.columns:
        inplay['Bearing'] = np.nan
    if 'BatterSide' not in inplay.columns:
        inplay['BatterSide'] = ""
    inplay['spray_cat'] = inplay.apply(_assign_spray_category_row, axis=1)
    tt = inplay['TaggedHitType'].astype(str).str.lower()

    def pct(mask):
        try:
            return round(100 * float(np.nanmean(mask.astype(float))), 1) if len(mask) else 0.0
        except Exception:
            return 0.0

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
    """
    Build a batted-ball table:
    - First column: Pitch Type
    - First row: 'Total' (overall across all pitches)
    - Then each pitch type (≥ 20 pitches only), sorted by Pitches desc
    - All percentage metrics rounded to 1 decimal
    """
    type_col = pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"

    # Total row
    rows = [{'Pitch Type': 'Total', **_bb_metrics_for_subset(df)}]

    # Per-type rows (≥ 20 pitches)
    if type_col in df.columns:
        for ptype, sub in df.groupby(type_col, dropna=False):
            metrics = _bb_metrics_for_subset(sub)
            if metrics['Pitches'] >= 20:
                rows.append({'Pitch Type': str(ptype), **metrics})

    out = pd.DataFrame(rows)

    # Keep Total first, then sort others by Pitches
    if len(out) > 1:
        total_row = out.iloc[[0]]
        others = out.iloc[1:].sort_values('Pitches', ascending=False)
        out = pd.concat([total_row, others], ignore_index=True)

    # Dtypes & rounding (ensure 1-decimal floats, ints for counts)
    for c in out.columns:
        if c.endswith('%'):
            out[c] = out[c].astype(float).round(1)
    out['Pitches'] = out['Pitches'].astype(int)
    return out

def _plate_metrics(sub: pd.DataFrame) -> dict:
    """Compute plate discipline metrics for a subset; return rounded (1-decimal) dict."""
    s_call = sub.get('PitchCall', pd.Series(dtype=object))
    lside = pd.to_numeric(sub.get('PlateLocSide',   pd.Series(dtype=float)), errors="coerce")
    lht   = pd.to_numeric(sub.get('PlateLocHeight', pd.Series(dtype=float)), errors="coerce")

    isswing  = s_call.isin(['StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
    iswhiff  = s_call.eq('StrikeSwinging')
    iscontact= s_call.isin(['InPlay','FoulBallNotFieldable','FoulBallFieldable'])
    isinzone = lside.between(-0.83, 0.83) & lht.between(1.5, 3.5)

    total_pitches = int(len(sub))
    total_swings  = int(isswing.sum())
    z_count       = int(isinzone.sum())

    def pct(val):
        try:
            return round(float(val) * 100, 1)
        except Exception:
            return 0.0

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
    """
    Build a table:
    - First column: Pitch Type
    - First row: 'Total' (overall across all pitches)
    - Then each pitch type (≥ 20 pitches only), sorted by Pitches desc
    - All percentage metrics rounded to 1 decimal
    """
    type_col = pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"

    rows = [{'Pitch Type': 'Total', **_plate_metrics(df)}]  # Total row FIRST

    if type_col in df.columns:
        for ptype, sub in df.groupby(type_col, dropna=False):
            metrics = _plate_metrics(sub)
            if metrics['Pitches'] >= 20:
                rows.append({'Pitch Type': str(ptype), **metrics})

    out = pd.DataFrame(rows)

    # sort after keeping 'Total' first
    if len(out) > 1:
        total_row = out.iloc[[0]]
        others = out.iloc[1:].sort_values('Pitches', ascending=False)
        out = pd.concat([total_row, others], ignore_index=True)

    # enforce 1-decimal percentages & int counts
    for c in out.columns:
        if c.endswith('%'):
            out[c] = out[c].astype(float).round(1)
    out['Pitches'] = out['Pitches'].astype(int)
    out['Zone Pitches'] = out['Zone Pitches'].astype(int)
    return out

def themed_table(df: pd.DataFrame):
    """Styled table with Husker red headers, white text, nowrap headers & cells, 1-decimal floats."""
    # Build format map: all float columns and any column ending in '%' -> 1 decimal
    float_cols = df.select_dtypes(include=["float", "float64"]).columns.tolist()
    percent_cols = [c for c in df.columns if c.strip().endswith('%')]
    fmt_map = {c: "{:.1f}" for c in set(float_cols) | set(percent_cols)}
    styles = [
        {'selector': 'thead th', 'props': f'background-color: {HUSKER_RED}; color: white; white-space: nowrap;'},
        {'selector': 'th',        'props': f'background-color: {HUSKER_RED}; color: white; white-space: nowrap;'},
        {'selector': 'td',        'props': 'white-space: nowrap;'},
    ]
    return (
        df.style
          .hide(axis="index")
          .format(fmt_map, na_rep="—")
          .set_table_styles(styles)
    )
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

if not os.path.exists(DATA_PATH):
    st.error(f"Data not found at {DATA_PATH}")
    st.stop()

df_all = load_csv_norm(DATA_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# DATA SEGMENT PICKER (Season / Scrimmages / Bullpens / Next Season)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Data Segment")
segment_choice = st.selectbox(
    "Choose time period",
    list(SEGMENT_DEFS.keys()),
    index=0,  # default to 2025 Season for now
    key="segment_choice"
)
df_segment = filter_by_segment(df_all, segment_choice)
if df_segment.empty:
    st.info(f"No rows found for **{segment_choice}** with the current dataset.")
    st.stop()

# Helper flag for bullpen behavior
SEG_TYPES = SEGMENT_DEFS.get(segment_choice, {}).get("types", [])
is_bullpen_segment = "bullpen" in SEG_TYPES

# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI (no sidebar; filters live in tabs/sections)
# ──────────────────────────────────────────────────────────────────────────────
# Nebraska pitchers (apply team filter after segment)
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

# Tabs: Standard / Compare / Profiles
tabs = st.tabs(["Standard", "Compare", "Profiles"])

# ── STANDARD TAB ───────────────────────────────────────────────────────────────
with tabs[0]:
    # Filters for this pitcher only (below the header)
    present_months = sorted(df_pitcher_all['Date'].dropna().dt.month.unique().tolist())
    col_m, col_d, col_side = st.columns([1,1,1.6])
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
    days_sel = col_d.multiselect(
        "Days (optional)",
        options=present_days,
        default=[],
        key="std_days",
    )
    hand_choice = col_side.radio("Batter Side (for heatmaps — now in Profiles)", ["Both","LHH","RHH"], index=0, horizontal=True, key="std_hand")

    # Filtered slice
    neb_df = filter_by_month_day(df_pitcher_all, months=months_sel, days=days_sel)
    season_label_base = build_pitcher_season_label(months_sel, days_sel, neb_df)
    season_label = f"{segment_choice} — {season_label_base}" if season_label_base else segment_choice

    if neb_df.empty:
        st.info("No rows for the selected filters.")
    else:
        logo_img = load_logo_img()

        # 1) Metrics (movement + summary)
        out = combined_pitcher_report(neb_df, player, logo_img, coverage=0.8, season_label=season_label)
        if out:
            fig_m, _ = out
            show_and_close(fig_m)

        # 2) Release Points (+ checkbox grid)
        type_col_all = pick_col(neb_df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
        types_available = (
            neb_df.get(type_col_all, pd.Series(dtype=object))
                  .dropna().map(canonicalize_type)
                  .replace("Unknown", np.nan).dropna().unique().tolist()
        )
        types_available = sorted(types_available)
        st.markdown("### Release Points")
        rel_selected = pitchtype_checkbox_grid(
            "Pitch Types to Show",
            options=types_available,
            key_prefix="std_release_types",
            default_all=True,
            columns_per_row=6,
        )
        rel_fig = release_points_figure(neb_df, player, include_types=rel_selected)
        if rel_fig:
            show_and_close(rel_fig)

        # 3) Extensions (Top-3), ordered by mean extension (desc), RESPECT same pitch filter
        st.markdown("### Extensions (Top-3 by usage, sorted by mean extension ↓)")
        ext_fig = extensions_topN_figure(neb_df, player, include_types=rel_selected, top_n=3)
        if ext_fig:
            show_and_close(ext_fig)

# ── COMPARE TAB ────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("#### Compare Appearances")
    cmp_n = st.selectbox("Number of windows", [2,3], index=0, key="cmp_n")
    cmp_hand = st.radio("Batter Side (heatmaps)", ["Both","LHH","RHH"], index=0, horizontal=True, key="cmp_hand")

    # Available pitch types across full pitcher season (release plot)
    type_col_all = pick_col(df_pitcher_all, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
    types_avail_all = (
        df_pitcher_all.get(type_col_all, pd.Series(dtype=object))
                      .dropna().map(canonicalize_type)
                      .replace("Unknown", np.nan).dropna().unique().tolist()
    )
    types_avail_all = sorted(types_avail_all)
    st.markdown("**Pitch Types (Release Plot, all windows)**")
    cmp_types_selected = pitchtype_checkbox_grid(
        "Select pitch types",
        options=types_avail_all,
        key_prefix="cmp_rel_types",
        default_all=True,
        columns_per_row=6,
    )

    # Outcome heatmaps pitch-type filter (shared across windows)
    types_avail_out = (
        df_pitcher_all.get(type_col_all, pd.Series(dtype=object))
                      .dropna().astype(str).unique().tolist()
    )
    types_avail_out = sorted(types_avail_out)
    if not is_bullpen_segment:
        st.markdown("**Pitch Types (Heatmaps — Whiffs / Strikeouts / Damage)**")
        cmp_types_out_selected = pitchtype_checkbox_grid(
            "Select outcome pitch types",
            options=types_avail_out,
            key_prefix="cmp_types_outcomes",
            default_all=True,
            columns_per_row=6,
        )
    else:
        cmp_types_out_selected = []

    # Build per-window filters (months/days ONLY)
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
            dy_sel = st.multiselect(
                f"Days (Window {'ABC'[i]})",
                options=day_opts,
                key=f"cmp_days_{i}"
            )
            df_win = filter_by_month_day(df_pitcher_all, months=mo_sel, days=dy_sel)
            season_lab = build_pitcher_season_label(mo_sel, dy_sel, df_win)
            season_lab = f"{segment_choice} — {season_lab}" if season_lab else segment_choice
            windows.append((season_lab, df_win))

    st.markdown("---")
    cols_out = st.columns(cmp_n)
    logo_img = load_logo_img()
    for i, (season_lab, df_win) in enumerate(windows):
        with cols_out[i]:
            st.markdown(f"**Window {'ABC'[i]} — {season_lab}**")
            if df_win.empty:
                st.info("No data for this window."); 
                continue
            out_win = combined_pitcher_report(df_win, player, logo_img, coverage=0.8, season_label=season_lab)
            if out_win:
                fig_m, _ = out_win
                show_and_close(fig_m)
            # Heatmaps — only when not bullpen
            if not is_bullpen_segment:
                fig_h = combined_pitcher_heatmap_report(
                    df_win, player, hand_filter=cmp_hand, season_label=season_lab,
                    outcome_pitch_types=cmp_types_out_selected,  # list (possibly [])
                )
                if fig_h:
                    show_and_close(fig_h)
            fig_r = release_points_figure(df_win, player, include_types=cmp_types_selected)
            if fig_r:
                show_and_close(fig_r)

# ── PROFILES TAB ───────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("#### Pitcher Profiles")
    if is_bullpen_segment:
        st.info("Profiles are not available for **Bullpens** (no batting occurs).")
    else:
        # Independent filters for profiles
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
        prof_days = col_pd.multiselect(
            "Days (optional)",
            options=prof_days_all,
            default=[],
            key="prof_days"
        )
        last_n_games = int(col_ln.number_input("Last N games", min_value=0, max_value=50, value=0, step=1, format="%d", key="prof_lastn"))
        prof_hand = col_side.radio("Batter Side", ["Both","LHH","RHH"], index=0, horizontal=True, key="prof_hand")

        # Build filtered dataset for PROFILES
        df_prof = filter_by_month_day(df_pitcher_all, months=prof_months, days=prof_days).copy()

        # Batter-side filter
        side_col = find_batter_side_col(df_prof)
        if prof_hand in ("LHH","RHH") and side_col is not None and not df_prof.empty:
            sides = normalize_batter_side(df_prof[side_col])
            target = "L" if prof_hand == "LHH" else "R"
            df_prof = df_prof[sides == target].copy()

        # Last N games (applied after month/day & hand filters)
        if last_n_games and not df_prof.empty:
            uniq_dates = pd.to_datetime(df_prof['Date'], errors="coerce").dt.date.dropna().unique()
            uniq_dates = sorted(uniq_dates)
            last_dates = set(uniq_dates[-last_n_games:])
            df_prof = df_prof[pd.to_datetime(df_prof['Date'], errors="coerce").dt.date.isin(last_dates)].copy()

        if df_prof.empty:
            st.info("No rows for the selected profile filters.")
        else:
            # Batted Ball Profile — by Pitch Type (Total first, ≥20 only)
            bb_df_typed = make_pitcher_batted_ball_by_type(df_prof)
            st.markdown(f"### Batted Ball Profile (by Pitch Type) — {segment_choice}")
            st.table(themed_table(bb_df_typed))

            # Plate Discipline Profile — by Pitch Type (Total first, ≥20 only)
            pd_df_typed = make_pitcher_plate_discipline_by_type(df_prof)
            st.markdown(f"### Plate Discipline Profile (by Pitch Type) — {segment_choice}")
            st.table(themed_table(pd_df_typed))

            # 🔥 Heatmaps section MOVED HERE from Standard
            season_label_prof_base = build_pitcher_season_label(prof_months, prof_days, df_prof)
            season_label_prof = f"{segment_choice} — {season_label_prof_base}" if season_label_prof_base else segment_choice

            st.markdown("### Pitcher Heatmaps")
            type_col_for_hm = pick_col(df_prof, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
            types_available_hm = (
                df_prof.get(type_col_for_hm, pd.Series(dtype=object))
                      .dropna().astype(str).unique().tolist()
            )
            types_available_hm = sorted(types_available_hm)
            hm_selected = pitchtype_checkbox_grid(
                "Filter Whiffs / Strikeouts / Damage by Pitch Type",
                options=types_available_hm,
                key_prefix="prof_hm_types",
                default_all=True,
                columns_per_row=6,
            )
            fig_h_prof = combined_pitcher_heatmap_report(
                df_prof, player, hand_filter=prof_hand, season_label=season_label_prof,
                outcome_pitch_types=hm_selected,
            )
            if fig_h_prof:
                show_and_close(fig_h_prof)
