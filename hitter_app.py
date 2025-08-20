# hitter_app.py
import os
import gc
import base64
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from matplotlib import colors

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nebraska Baseball — Hitter Reports",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.set_option("client.showErrorDetails", True)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "B10C25_streamlit_streamlit_columns.csv"
LOGO_PATH = "Nebraska-Cornhuskers-Logo.png"
BANNER_IMG = "NebraskaChampions.jpg"

# ──────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS & FIGURE DISPLAY
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
    try:
        st.pyplot(fig=fig, clear_figure=False)
    finally:
        plt.close(fig); gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────────────────────────────
def hero_banner(title: str, *, subtitle: str | None = None, height_px: int = 260):
    b64 = load_banner_b64()
    bg_url = f"data:image/jpeg;base64,{b64}" if b64 else ""
    sub_html = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <style>
        .hero-wrap {{
            position: relative; width: 100%; height: {height_px}px;
            border-radius: 10px; overflow: hidden; margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }}
        .hero-bg {{
            position: absolute; inset: 0;
            background:
              linear-gradient(to bottom, rgba(0,0,0,0.45), rgba(0,0,0,0.60)),
              url('{bg_url}');
            background-size: cover; background-position: center;
            filter: saturate(105%);
        }}
        .hero-text {{
            position: absolute; inset: 0; display: flex;
            align-items: center; justify-content: center; flex-direction: column;
            color: #fff; text-align: center;
        }}
        .hero-title {{
            font-size: 40px; font-weight: 800; letter-spacing: .5px;
            text-shadow: 0 2px 8px rgba(0,0,0,.45); margin: 0;
        }}
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

# Banner: just “Nebraska Baseball”
hero_banner("Nebraska Baseball", subtitle=None, height_px=260)

# ──────────────────────────────────────────────────────────────────────────────
# DATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
DATE_CANDIDATES = [
    "Date","date","GameDate","GAME_DATE","Game Date","date_game",
    "Datetime","DateTime","game_datetime","GameDateTime"
]
def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    found = None
    lower = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand.lower() in lower:
            found = lower[cand.lower()]; break
    if found is None:
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

# ──────────────────────────────────────────────────────────────────────────────
# STRIKE ZONE/DENSITY (same as pitcher app)
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
    l, b, w, h = get_zone_bounds()
    return l - w*0.8, l + w + w*0.8, b - h*0.6, b + h + h*0.6
def draw_strikezone(ax):
    l, b, w, h = get_zone_bounds()
    ax.add_patch(Rectangle((l, b), w, h, fill=False, lw=2, color="black"))
    for f in (1/3, 2/3):
        ax.vlines(l + w*f, b, b+h, colors="gray", ls="--", lw=1)
        ax.hlines(b + h*f, l, l+w, colors="gray", ls="--", lw=1)
def compute_density_hitter(x, y, xi_m, yi_m):
    coords = np.vstack([x, y])
    mask = np.isfinite(coords).all(axis=0)
    if mask.sum() <= 1: return np.zeros(xi_m.shape)
    try:
        kde = gaussian_kde(coords[:, mask])
        return kde(np.vstack([xi_m.ravel(), yi_m.ravel()])).reshape(xi_m.shape)
    except Exception:
        return np.zeros(xi_m.shape)
def format_name(name):
    if isinstance(name, str) and ',' in name:
        last, first = [s.strip() for s in name.split(',', 1)]
        return f"{first} {last}"
    return str(name)

# ──────────────────────────────────────────────────────────────────────────────
# HITTER FIGURES
# ──────────────────────────────────────────────────────────────────────────────
def compute_density_hitter_plot(ax, sub: pd.DataFrame, title: str):
    x_min, x_max, y_min, y_max = get_view_bounds()
    draw_strikezone(ax)
    x = sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy()
    y = sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y); x, y = x[valid], y[valid]

    if len(sub) < 10:
        for _, r in sub.iterrows():
            if np.isfinite(r.get('PlateLocSide', np.nan)) and np.isfinite(r.get('PlateLocHeight', np.nan)):
                ax.plot(r['PlateLocSide'], r['PlateLocHeight'], 'o', color='deepskyblue', alpha=0.8, ms=6)
    else:
        xi = np.linspace(x_min, x_max, 200)
        yi = np.linspace(y_min, y_max, 200)
        xi_m, yi_m = np.meshgrid(xi, yi)
        zi = compute_density_hitter(
            sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy(),
            sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy(),
            xi_m, yi_m
        )
        ax.imshow(zi, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap=custom_cmap)
        draw_strikezone(ax)

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal', 'box')
    ax.set_title(title, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])

def combined_hitter_heatmap_report(df, batter, logo_img=None):
    df_b = df[df['Batter'] == batter].copy()
    if df_b.empty:
        st.error(f"No data for batter '{batter}'."); return None

    df_b['iscontact'] = df_b['PitchCall'].isin(['InPlay','FoulBallFieldable','FoulBallNotFieldable'])
    df_b['iswhiff']   = df_b['PitchCall'].eq('StrikeSwinging')
    df_b['is95plus']  = df_b['ExitSpeed'] >= 95

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

    # LHP vs RHP panes for Contact / Whiffs / >=95
    sub_contact_l = df_b[df_b['iscontact'] & (df_b['PitcherThrows']=='Left')]
    sub_contact_r = df_b[df_b['iscontact'] & (df_b['PitcherThrows']=='Right')]
    ax1 = fig.add_subplot(gs[0, 0]); compute_density_hitter_plot(ax1, sub_contact_l, 'Contact vs LHP')
    ax2 = fig.add_subplot(gs[0, 2]); compute_density_hitter_plot(ax2, sub_contact_r, 'Contact vs RHP')

    sub_whiff_l = df_b[df_b['iswhiff'] & (df_b['PitcherThrows']=='Left')]
    sub_whiff_r = df_b[df_b['iswhiff'] & (df_b['PitcherThrows']=='Right')]
    ax3 = fig.add_subplot(gs[0, 3]); compute_density_hitter_plot(ax3, sub_whiff_l, 'Whiffs vs LHP')
    ax4 = fig.add_subplot(gs[0, 5]); compute_density_hitter_plot(ax4, sub_whiff_r, 'Whiffs vs RHP')

    sub_95_l = df_b[df_b['is95plus'] & (df_b['PitcherThrows']=='Left')]
    sub_95_r = df_b[df_b['is95plus'] & (df_b['PitcherThrows']=='Right')]
    ax5 = fig.add_subplot(gs[0, 6]); compute_density_hitter_plot(ax5, sub_95_l, 'Exit ≥95 vs LHP')
    ax6 = fig.add_subplot(gs[0, 8]); compute_density_hitter_plot(ax6, sub_95_r, 'Exit ≥95 vs RHP')

    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88,0.88,0.12,0.12], anchor='NE'); axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis('off')

    fig.suptitle(f"{format_name(batter)} Heatmaps", fontsize=18, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def create_hitter_report(df, batter, ncols=3):
    bdf = df[df['Batter'] == batter]
    pa_groups = list(bdf.groupby(['GameID','Inning','Top/Bottom','PAofInning']))
    n_pa = len(pa_groups); nrows = max(1, math.ceil(n_pa/ncols))

    fig = plt.figure(figsize=(3+4*ncols+1, 4*nrows))
    gs = GridSpec(nrows, ncols, wspace=0.1, hspace=0.3)
    for idx, ((_, inn, tb, _), padf) in enumerate(pa_groups):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col]); draw_strikezone(ax)
        for _, p in padf.iterrows():
            clr = {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan','InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.get(p.PitchCall,'black')
            ax.scatter(p.PlateLocSide, p.PlateLocHeight, c=clr, s=120, edgecolor='white', lw=1, zorder=2)
        ax.set_xlim(-3,3); ax.set_ylim(0,5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}", fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig

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
    st.error(f"Data not found at {DATA_PATH}"); st.stop()
df_all = load_csv_norm(DATA_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# UI – Nebraska Hitter Reports
# ──────────────────────────────────────────────────────────────────────────────
neb_b_df = df_all[df_all.get('BatterTeam','') == 'NEB'].copy()
st.markdown("### Hitter Report")

# Date then batter for chosen game
date_opts = sorted(neb_b_df['Date'].dropna().dt.date.unique().tolist())
sel_date = st.selectbox("Game Date", options=date_opts, format_func=lambda d: format_date_long(d)) if date_opts else None
if not sel_date:
    st.info("No Nebraska hitter dates available."); st.stop()

df_date = neb_b_df[neb_b_df['Date'].dt.date==sel_date]
batters = sorted(df_date['Batter'].dropna().unique().tolist())
batter  = st.selectbox("Batter", batters) if batters else None

if not batter:
    st.info("Choose a batter."); st.stop()

tabs = st.tabs(["Standard", "Heatmaps"])
with tabs[0]:
    fig = create_hitter_report(df_date, batter, ncols=3)
    if fig: show_and_close(fig)
with tabs[1]:
    fig = combined_hitter_heatmap_report(df_date, batter)
    if fig: show_and_close(fig)
