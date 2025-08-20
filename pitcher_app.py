# pitcher_app.py

import os
import math
import numpy as np
import pandas as pd
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
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Nebraska Pitcher Reports",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "/mnt/data/B10C25_streamlit_streamlit_columns.csv"  # update if needed
LOGO_PATH = "Nebraska-Cornhuskers-Logo.png"

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
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand.lower() in cols_lower:
            found = cols_lower[cand.lower()]
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
    if d is None or pd.isna(d): return ""
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

def build_pitcher_season_label(months_sel, days_sel, selected_df: pd.DataFrame, month_name_by_num: dict) -> str:
    if (not months_sel) and (not days_sel):
        return "Season"
    if months_sel and (not days_sel) and len(months_sel) == 1:
        return month_name_by_num.get(months_sel[0], "Season")
    if selected_df is None or selected_df.empty or "Date" not in selected_df.columns:
        return "Season"
    rng = summarize_dates_range(selected_df["Date"])
    return rng if rng else "Season"

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
        ax.vlines(sz_left + sz_width*f,  sz_bottom, sz_bottom+sz_height, colors="gray", ls="--", lw=1)
        ax.hlines(sz_bottom + sz_height*f, sz_left, sz_left+sz_width,     colors="gray", ls="--", lw=1)

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
# CONFERENCES (Big Ten filled; others placeholders)
# ──────────────────────────────────────────────────────────────────────────────
BIG_TEN_MAP = {
    'ILL_ILL': 'Illinois','MIC_SPA': 'Michigan State','UCLA': 'UCLA','IOW_HAW': 'Iowa',
    'IU': 'Indiana','MAR_TER': 'Maryland','MIC_WOL': 'Michigan','MIN_GOL': 'Minnesota',
    'NEB': 'Nebraska','NOR_CAT': 'Northwestern','ORE_DUC': 'Oregon','OSU_BUC': 'Ohio State',
    'PEN_NIT': 'Penn State','PUR_BOI': 'Purdue','RUT_SCA': 'Rutgers','SOU_TRO': 'USC','WAS_HUS': 'Washington'
}
BIG_12_MAP, SEC_MAP, ACC_MAP = {}, {}, {}
CONF_MAP = {"Big Ten": BIG_TEN_MAP, "Big 12": BIG_12_MAP, "SEC": SEC_MAP, "ACC": ACC_MAP}

MONTH_CHOICES = [
    (1,"January"), (2,"February"), (3,"March"), (4,"April"),
    (5,"May"), (6,"June"), (7,"July"), (8,"August"),
    (9,"September"), (10,"October"), (11,"November"), (12,"December")
]
MONTH_NAME_BY_NUM = {n: name for n, name in MONTH_CHOICES}

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_csv_norm(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df = ensure_date_column(df)
    return df

if not os.path.exists(DATA_PATH):
    st.error(f"Data not found at {DATA_PATH}")
    st.stop()
df_all = load_csv_norm(DATA_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# DENSITY & HAND
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

def pick_col(df: pd.DataFrame, *cands) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c and c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def find_batter_side_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, "BatterSide", "Batter Side", "Batter_Bats", "BatterBats", "Bats", "Stand", "BatSide")

def normalize_batter_side(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str[0].str.upper()
    return s.replace({"L":"L","R":"R","S":"S","B":"S"})

def parse_hand_filter_to_LR(hand_filter: str) -> str | None:
    s = str(hand_filter).strip().lower()
    if s in {"l", "lhh", "lhb", "left", "left-handed", "left handed"}:
        return "L"
    if s in {"r", "rhh", "rhb", "right", "right-handed", "right handed"}:
        return "R"
    return None

# ──────────────────────────────────────────────────────────────────────────────
# PITCHER: STANDARD REPORT (Movement + Summary)
# ──────────────────────────────────────────────────────────────────────────────
def combined_pitcher_report(df, pitcher_name, logo_img, coverage=0.8, season_label="Season", opponent_label=None):
    df_p = df[df['Pitcher'] == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' with the current filters.")
        return None

    total = len(df_p)
    is_strike = df_p['PitchCall'].isin(['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
    grp = df_p.groupby('AutoPitchType')

    summary = pd.DataFrame({
        'Pitch Type': grp.size().index,
        'Pitches': grp.size().values,
        'Usage %': grp.size().values / total * 100,
        'Strike %': grp.apply(lambda g: is_strike.loc[g.index].mean() * 100).values,
        'Rel Speed': grp['RelSpeed'].mean().values,
        'Spin Rate': grp['SpinRate'].mean().values,
        'IVB': grp['InducedVertBreak'].mean().values,
        'HB': grp['HorzBreak'].mean().values,
        'Rel Height': grp['RelHeight'].mean().values,
        'VAA': grp['VertApprAngle'].mean().values,
        'Extension': grp['Extension'].mean().values
    }).round({'Usage %':1,'Strike %':1,'Rel Speed':1,'Spin Rate':1,'IVB':1,'HB':1,'Rel Height':2,'VAA':1,'Extension':2}) \
     .sort_values('Pitches', ascending=False)

    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.7], hspace=0.3)

    # Movement
    axm = fig.add_subplot(gs[0, 0]); axm.set_title('Movement Plot', fontweight='bold')
    chi2v = chi2.ppf(coverage, df=2)
    axm.axhline(0, ls='--', color='grey'); axm.axvline(0, ls='--', color='grey')
    for ptype, g in grp:
        x, y = g['HorzBreak'], g['InducedVertBreak']; clr = get_pitch_color(ptype)
        axm.scatter(x, y, label=ptype, color=clr, alpha=0.7)
        if len(g) > 1:
            cov = np.cov(np.vstack((x, y))); vals, vecs = np.linalg.eigh(cov)
            ord_ = vals.argsort()[::-1]; vals, vecs = vals[ord_], vecs[:, ord_]
            ang = np.degrees(np.arctan2(*vecs[:,0][::-1])); w, h = 2*np.sqrt(vals*chi2v)
            axm.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang, edgecolor=clr, facecolor=clr, alpha=0.2, ls='--', lw=1.5))
    axm.set_xlim(-30,30); axm.set_ylim(-30,30); axm.set_aspect('equal','box')
    axm.set_xlabel('Horizontal Break'); axm.set_ylabel('Induced Vertical Break')
    axm.legend(title='Pitch Type', fontsize=8, title_fontsize=9, loc='upper right')

    # Summary table
    axt = fig.add_subplot(gs[1, 0]); axt.axis('off')
    tbl = axt.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Summary Metrics', fontweight='bold', y=0.87)

    # Logo
    if logo_img is not None:
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10); axl.imshow(logo_img); axl.axis('off')
    elif os.path.exists(LOGO_PATH):
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10); axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis('off')

    # Title
    extra = f" vs {opponent_label}" if opponent_label else ""
    fig.suptitle(f"{format_name(pitcher_name)} Metrics{extra}\n({season_label})", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, summary

# ──────────────────────────────────────────────────────────────────────────────
# PITCHER HEATMAPS — Top 3 pitches; Whiffs/Strikeouts/Damage; handedness filter
# ──────────────────────────────────────────────────────────────────────────────
def combined_pitcher_heatmap_report(df, pitcher_name, hand_filter="Both", grid_size=100, season_label="Season", opponent_label=None):
    df_p = df[df['Pitcher'] == pitcher_name].copy()
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' with the current filters.")
        return None

    side_col = find_batter_side_col(df_p)
    hand_label = "Both"
    if side_col is not None:
        sides = normalize_batter_side(df_p[side_col])
        want = parse_hand_filter_to_LR(hand_filter)
        if want == "L":
            df_p = df_p[sides == "L"]; hand_label = "LHH"
        elif want == "R":
            df_p = df_p[sides == "R"]; hand_label = "RHH"

    if df_p.empty:
        st.info("No pitches for the selected batter-side filter.")
        return None

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

    top3 = list(df_p['AutoPitchType'].value_counts().index[:3])
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        if i < len(top3):
            pitch = top3[i]
            sub = df_p[df_p['AutoPitchType'] == pitch]
            panel(ax, sub, f"{pitch} (n={len(sub)})")
        else:
            draw_strikezone(ax, z_left, z_bottom, z_w, z_h)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal','box')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("—", fontweight='bold')

    sub_wh = df_p[df_p['PitchCall'] == 'StrikeSwinging']
    sub_ks = df_p[df_p['KorBB'] == 'Strikeout']
    sub_dg = df_p[df_p['ExitSpeed'] >= 95]
    ax = fig.add_subplot(gs[1, 0]); panel(ax, sub_wh, f"Whiffs (n={len(sub_wh)})")
    ax = fig.add_subplot(gs[1, 1]); panel(ax, sub_ks, f"Strikeouts (n={len(sub_ks)})")
    ax = fig.add_subplot(gs[1, 2]); panel(ax, sub_dg, f"Damage (n={len(sub_dg)})", color='orange')

    axt = fig.add_subplot(gs[2, :]); axt.axis('off')
    fp  = (df_p[(df_p['Balls']==0) & (df_p['Strikes']==0)]['PitchCall'].isin(['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']).mean()*100) if len(df_p)>0 else np.nan
    mix = (df_p[((df_p['Balls']==1)&(df_p['Strikes']==0)) | ((df_p['Balls']==0)&(df_p['Strikes']==1)) | ((df_p['Balls']==1)&(df_p['Strikes']==1))]['PitchCall'].isin(['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']).mean()*100) if len(df_p)>0 else np.nan
    hp  = (df_p[((df_p['Balls']==2)&(df_p['Strikes']==0)) | ((df_p['Balls']==2)&(df_p['Strikes']==1)) | ((df_p['Balls']==3)&(df_p['Strikes']==1))]['PitchCall'].isin(['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']).mean()*100) if len(df_p)>0 else np.nan
    two = (df_p[(df_p['Strikes']==2) & (df_p['Balls']<3)]['PitchCall'].isin(['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']).mean()*100) if len(df_p)>0 else np.nan
    metrics = pd.DataFrame({'1st Pitch %':[fp],'Mix Count %':[mix],'Hitter+ %':[hp],'2-Strike %':[two]}).round(1)
    tbl = axt.table(cellText=metrics.values, colLabels=metrics.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Strike Percentage by Count', y=0.75, fontweight='bold')

    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88, 0.92, 0.10, 0.10], anchor='NE', zorder=10); axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis('off')

    extra = f" vs {opponent_label}" if opponent_label else ""
    fig.suptitle(f"{format_name(pitcher_name)} Heatmaps{extra}\n({season_label}) ({hand_label})", fontsize=18, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# RELEASE POINTS (with pitch-type filter)
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
    x_col       = pick_col(df, "Relside","RelSide","ReleaseSide","Release_Side","release_pos_x")
    y_col       = pick_col(df, "Relheight","RelHeight","ReleaseHeight","Release_Height","release_pos_z")
    type_col    = pick_col(df, "AutoPitchType","Auto Pitch Type","PitchType","TaggedPitchType") or "AutoPitchType"
    speed_col   = pick_col(df, "Relspeed","RelSpeed","ReleaseSpeed","RelSpeedMPH","release_speed")

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

    if include_types:
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
        means = means.sort_values("mean_speed", ascending=False, na_position="last").reset_index(drop=True)

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
        eLx, eLy = x1 + px*ARM_TIP_HALF_WIDTH,  y1 + py*ARM_TIP_HALF_WIDTH
        eRx, eRy = x1 - px*ARM_TIP_HALF_WIDTH,  y1 - py*ARM_TIP_HALF_WIDTH
        arm_poly = Polygon([(sLx, sLy), (eLx, eLy), (eRx, eRy), (sRx, sRy)],
                           closed=True, facecolor=ARM_FILL_COLOR, edgecolor=ARM_FILL_COLOR, zorder=1)
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
        handles.append(Line2D([0],[0], marker="o", linestyle="none", markersize=6,
                              label=label, color=row["color"]))
    if handles:
        ax2.legend(handles=handles, title="Pitch Type", loc="upper right")

    fig.suptitle(f"{format_name(pitcher_name)} Release Points", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR (Pitcher → Opponent → Months/Days)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.caption("Use the ◀ chevron at top-left to hide/show this drawer.")

    neb_df_all = df_all[df_all['PitcherTeam']=='NEB'].copy()
    pitchers_all = sorted(neb_df_all['Pitcher'].dropna().unique().tolist())
    st.selectbox("Pitcher", pitchers_all, key="neb_player")

    pitcher = st.session_state.get("neb_player")
    if pitcher:
        df_pitcher_all = neb_df_all[neb_df_all['Pitcher'] == pitcher].copy()
    else:
        df_pitcher_all = neb_df_all.iloc[0:0].copy()

    team_map_all = {**BIG_TEN_MAP, **BIG_12_MAP, **SEC_MAP, **ACC_MAP}
    opp_codes = sorted(df_pitcher_all['BatterTeam'].dropna().unique().tolist())
    opp_pairs = [(code, team_map_all.get(code, code)) for code in opp_codes]
    opp_display = ["(All Opponents)"] + [name for _, name in opp_pairs]
    opp_sel_name = st.selectbox("Opponent (optional)", opp_display, index=0, key="neb_pitch_opp_name")
    if opp_sel_name and opp_sel_name != "(All Opponents)":
        opp_code = next(code for code, name in opp_pairs if name == opp_sel_name)
    else:
        opp_code = None
    st.session_state['neb_pitch_opp'] = opp_code

    df_scope = df_pitcher_all if opp_code is None else df_pitcher_all[df_pitcher_all['BatterTeam'] == opp_code]
    present_months = sorted(df_scope['Date'].dropna().dt.month.unique().tolist())
    st.multiselect(
        "Months (optional)",
        options=present_months,
        format_func=lambda n: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][n-1],
        default=[],
        key="neb_pitch_months",
    )

    months_sel = st.session_state.get("neb_pitch_months", [])
    dates_series = df_scope['Date'].dropna()
    if months_sel:
        dates_series = dates_series[dates_series.dt.month.isin(months_sel)]
    present_days = sorted(dates_series.dt.day.unique().tolist())
    st.multiselect(
        "Days (optional)",
        options=present_days,
        default=[],
        key="neb_pitch_days",
    )

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
st.title("Nebraska Baseball")
st.subheader("Pitcher Report")

player   = st.session_state.get('neb_player')
logo_img = mpimg.imread(LOGO_PATH) if os.path.exists(LOGO_PATH) else None

if not player:
    st.info("Choose a pitcher in the Filters drawer.")
    st.stop()

df_pitcher_all = neb_df_all[neb_df_all['Pitcher'] == player].copy()
appearances = int(pd.to_datetime(df_pitcher_all['Date'], errors="coerce").dt.date.nunique())
st.subheader(f"{format_name(player)} ({appearances} Appearances)")

tabs = st.tabs(["Standard", "Compare"])

with tabs[0]:
    months_sel = st.session_state.get("neb_pitch_months", [])
    days_sel   = st.session_state.get("neb_pitch_days", [])
    opp_code   = st.session_state.get("neb_pitch_opp", None)
    team_map_all = {**BIG_TEN_MAP, **BIG_12_MAP, **SEC_MAP, **ACC_MAP}
    opp_label = team_map_all.get(opp_code, opp_code) if opp_code else None

    df_scope = df_pitcher_all if opp_code is None else df_pitcher_all[df_pitcher_all['BatterTeam'] == opp_code]
    neb_df   = filter_by_month_day(df_scope, months=months_sel, days=days_sel)
    season_label = build_pitcher_season_label(months_sel, days_sel, neb_df, MONTH_NAME_BY_NUM)

    if neb_df.empty:
        st.info("No rows with current filters.")
    else:
        # Movement + Summary
        out = combined_pitcher_report(neb_df, player, logo_img, coverage=0.8, season_label=season_label, opponent_label=opp_label)
        if out: fig, _ = out; st.pyplot(fig=fig)

        # Heatmaps
        st.markdown("### Pitcher Heatmaps")
        hand_choice = st.radio("Batter Side", ["Both","LHH","RHH"], index=0, horizontal=True, key="neb_heat_hand_main")
        heat_fig = combined_pitcher_heatmap_report(neb_df, player, hand_filter=hand_choice, season_label=season_label, opponent_label=opp_label)
        if heat_fig: st.pyplot(fig=heat_fig)

        # Release Points
        types_available = (
            neb_df.get('AutoPitchType', pd.Series(dtype=object))
                 .dropna().map(canonicalize_type)
                 .replace("Unknown", np.nan).dropna().unique().tolist()
        )
        types_available = sorted(types_available)
        st.markdown("### Release Points")
        if types_available:
            sel_types = st.multiselect("Pitch Types (Release Plot)", options=types_available, default=types_available, key="release_types")
            rel_fig = release_points_figure(neb_df, player, include_types=sel_types if sel_types else [])
            if rel_fig: st.pyplot(fig=rel_fig)
        else:
            st.info("No recognizable pitch types available to plot.")

with tabs[1]:
    st.markdown("#### Compare Appearances")
    cmp_n = st.selectbox("Number of windows", [2,3], index=0, key="neb_cmp_n_tab")
    cmp_hand = st.radio("Batter Side (heatmaps)", ["Both","LHH","RHH"], index=0, key="neb_cmp_hand_tab")

    types_avail_all = (
        df_pitcher_all.get('AutoPitchType', pd.Series(dtype=object))
            .dropna().map(canonicalize_type)
            .replace("Unknown", np.nan).dropna().unique().tolist()
    )
    types_avail_all = sorted(types_avail_all)
    cmp_types = st.multiselect("Pitch Types (release plot, all windows)", options=types_avail_all, default=types_avail_all, key="neb_cmp_types_tab")

    month_options = sorted(df_pitcher_all['Date'].dropna().dt.month.unique().tolist())
    cols_cmp = st.columns(cmp_n)
    windows = []
    for i in range(cmp_n):
        with cols_cmp[i]:
            st.markdown(f"**Window {'ABC'[i]} Filters**")
            mo_sel = st.multiselect(
                f"Months (Window {'ABC'[i]})",
                options=month_options,
                format_func=lambda n: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][n-1],
                key=f"cmp_months_tab_{i}"
            )
            dser = df_pitcher_all['Date'].dropna()
            if mo_sel: dser = dser[dser.dt.month.isin(mo_sel)]
            day_opts = sorted(dser.dt.day.unique().tolist())
            dy_sel = st.multiselect(f"Days (Window {'ABC'[i]})", options=day_opts, key=f"cmp_days_tab_{i}")
            df_win = filter_by_month_day(df_pitcher_all, months=mo_sel, days=dy_sel)
            season_lab = build_pitcher_season_label(mo_sel, dy_sel, df_win, MONTH_NAME_BY_NUM)
            windows.append((season_lab, df_win))

    st.markdown("---")
    cols_out = st.columns(cmp_n)
    for i, (season_lab, df_win) in enumerate(windows):
        with cols_out[i]:
            st.markdown(f"**Window {'ABC'[i]} — {season_lab}**")
            if df_win.empty:
                st.info("No data for this window.")
                continue
            out_win = combined_pitcher_report(df_win, player, logo_img, coverage=0.8, season_label=season_lab)
            if out_win:
                fig_m, _ = out_win; st.pyplot(fig=fig_m)
            fig_h = combined_pitcher_heatmap_report(df_win, player, hand_filter=cmp_hand, season_label=season_lab)
            if fig_h: st.pyplot(fig=fig_h)
            fig_r = release_points_figure(df_win, player, include_types=cmp_types if cmp_types else None)
            if fig_r: st.pyplot(fig=fig_r)
