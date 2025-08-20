# hitter_app.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from matplotlib import colors

st.set_page_config(layout="wide", page_title="Nebraska Hitter Reports", initial_sidebar_state="expanded")

DATA_PATH = "/mnt/data/B10C25_streamlit_streamlit_columns.csv"
LOGO_PATH = "Nebraska-Cornhuskers-Logo.png"

DATE_CANDIDATES = ["Date","date","GameDate","GAME_DATE","Game Date","date_game","Datetime","DateTime","game_datetime","GameDateTime"]

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    found = None
    m = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand.lower() in m:
            found = m[cand.lower()]
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

custom_cmap = colors.LinearSegmentedColormap.from_list("custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")], N=256)

def get_zone_bounds():
    return -0.83, 1.17, 1.66, 2.75

def get_view_bounds():
    l, b, w, h = get_zone_bounds()
    return l - w*0.8, l + w*1.8, b - h*0.6, b + h*1.6

def draw_strikezone(ax):
    l, b, w, h = get_zone_bounds()
    ax.add_patch(Rectangle((l,b), w, h, fill=False, lw=2, color="black"))
    for f in (1/3, 2/3):
        ax.vlines(l + w*f,  b, b+h, colors="gray", ls="--", lw=1)
        ax.hlines(b + h*f, l, l+w, colors="gray", ls="--", lw=1)

def compute_density_hitter(x, y, xi_m, yi_m):
    coords = np.vstack([x, y])
    mask = np.isfinite(coords).all(axis=0)
    if mask.sum() <= 1:
        return np.zeros(xi_m.shape)
    kde = gaussian_kde(coords[:, mask])
    return kde(np.vstack([xi_m.ravel(), yi_m.ravel()])).reshape(xi_m.shape)

def get_pitch_color(ptype):
    if isinstance(ptype, str) and (ptype.lower().startswith("four-seam fastball") or ptype.lower() == "fastball"):
        return "#E60026"
    savant = {"sinker":"#FF9300","cutter":"#800080","changeup":"#008000","curveball":"#0033CC",
              "slider":"#CCCC00","splitter":"#00CCCC","knuckle curve":"#000000","screwball":"#CC0066","eephus":"#666666"}
    return savant.get(str(ptype).lower(), "#E60026")

def format_name(name):
    if isinstance(name, str) and ',' in name:
        last, first = [s.strip() for s in name.split(',', 1)]
        return f"{first} {last}"
    return str(name)

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

def compute_density_hitter_plot(ax, sub, title):
    x_min, x_max, y_min, y_max = get_view_bounds()
    draw_strikezone(ax)
    x = sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy()
    y = sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(sub) < 10:
        for _, r in sub.iterrows():
            if np.isfinite(r.get('PlateLocSide', np.nan)) and np.isfinite(r.get('PlateLocHeight', np.nan)):
                color = get_pitch_color(r.get('AutoPitchType', ''))
                ax.plot(r['PlateLocSide'], r['PlateLocHeight'], 'o', color=color, alpha=0.8, ms=6)
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
    ax.set_title(title, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

def combined_hitter_heatmap_report(df, batter, logo_img=None):
    df_b = df[df['Batter'] == batter].copy()
    if df_b.empty:
        st.error(f"No data for batter '{batter}' on that date.")
        return None

    df_b['iscontact'] = df_b['PitchCall'].isin(['InPlay','FoulBallFieldable','FoulBallNotFieldable'])
    df_b['iswhiff'] = df_b['PitchCall'].eq('StrikeSwinging')
    df_b['is95plus'] = df_b['ExitSpeed'] >= 95

    date_str = format_date_long(df_b["Date"].iloc[0]) if "Date" in df_b.columns and df_b["Date"].notna().any() else ""

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

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

    formatted = format_name(batter)
    fig.suptitle(f"{formatted}{(' — ' + date_str) if date_str else ''}", fontsize=22, x=0.5, y=0.87)
    plt.tight_layout(rect=[0, 0, 1, 0.78])
    return fig

def create_hitter_report(df, batter, ncols=3):
    bdf = df[df['Batter'] == batter]
    pa = list(bdf.groupby(['GameID','Inning','Top/Bottom','PAofInning']))
    n_pa = len(pa); nrows = max(1, math.ceil(n_pa/ncols))
    descs = []
    for _, padf in pa:
        lines = []
        for _, p in padf.iterrows():
            vel = getattr(p, 'EffectiveVelo', np.nan)
            vel_str = f"{vel:.1f}" if pd.notna(vel) else "—"
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType} {vel_str} MPH / {p.PitchCall}")
        ip = padf[padf['PitchCall']=='InPlay']
        if not ip.empty:
            last = ip.iloc[-1]; res = last.PlayResult or 'InPlay'
            if not pd.isna(last.ExitSpeed): res += f" ({last.ExitSpeed:.1f} MPH)"
            lines.append(f"▶ PA Result: {res}")
        else:
            balls = (padf['PitchCall']=='BallCalled').sum()
            strikes = padf['PitchCall'].isin(['StrikeCalled','StrikeSwinging']).sum()
            if balls>=4: lines.append('▶ PA Result: Walk')
            elif strikes>=3: lines.append('▶ PA Result: Strikeout')
        descs.append(lines)

    fig = plt.figure(figsize=(3+4*ncols+1, 4*nrows))
    gs = GridSpec(nrows, ncols+1, width_ratios=[0.8]+[1]*ncols, wspace=0.1)
    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88,0.88,0.12,0.12], anchor='NE'); axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis('off')

    date_title = format_date_long(bdf["Date"].iloc[0]) if "Date" in bdf.columns and bdf["Date"].notna().any() else ""
    fig.suptitle(f"{format_name(batter)} Hitter Report{(' — ' + date_title) if date_title else ''}",
                 fontsize=16, x=0.55, y=1.0, fontweight='bold')

    for idx, ((_, inn, tb, _), padf) in enumerate(pa):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col+1])
        draw_strikezone(ax)
        hand = 'LHP' if str(padf['PitcherThrows'].iloc[0]).startswith('L') else 'RHP'
        pitchr = padf['Pitcher'].iloc[0]
        for _, p in padf.iterrows():
            mk = {'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}.get(p.AutoPitchType,'o')
            clr = {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan','InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.get(p.PitchCall,'black')
            sz = 200 if p.AutoPitchType=='Slider' else 150
            ax.scatter(p.PlateLocSide, p.PlateLocHeight, marker=mk, c=clr, s=sz, edgecolor='white', lw=1, zorder=2)
            yoff = -0.05 if p.AutoPitchType=='Slider' else 0
            ax.text(p.PlateLocSide, p.PlateLocHeight+yoff, str(int(p.PitchofPA)), ha='center', va='center', fontsize=6, fontweight='bold', zorder=3)
        ax.set_xlim(-3,3); ax.set_ylim(0,5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}", fontsize=10, fontweight='bold')
        ax.text(0.5,0.1,f"vs {pitchr} ({hand})", transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic')

    axd = fig.add_subplot(gs[:,0]); axd.axis('off'); y0 = 1.0; dy = 1.0/(max(1,n_pa)*5.0)
    for i, lines in enumerate(descs, 1):
        axd.hlines(y0-dy*0.1,0,1, transform=axd.transAxes, color='black', lw=1)
        axd.text(0.02,y0,f"PA {i}", fontsize=6, fontweight='bold', transform=axd.transAxes)
        yln = y0-dy
        for ln in lines:
            axd.text(0.02,yln,ln, fontsize=6, transform=axd.transAxes); yln -= dy
        y0 = yln - dy*0.05

    res_handles = [Line2D([0],[0], marker='o', color='w', label=k, markerfacecolor=v, ms=10, markeredgecolor='k')
                   for k,v in {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan','InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.items()]
    fig.legend(res_handles, [h.get_label() for h in res_handles], title='Result', loc='lower right', bbox_to_anchor=(0.90,0.02))
    pitch_handles = [Line2D([0],[0], marker=m, color='w', label=k, markerfacecolor='gray', ms=10, markeredgecolor='k')
                     for k,m in {'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}.items()]
    fig.legend(pitch_handles, [h.get_label() for h in pitch_handles], title='Pitches', loc='lower right', bbox_to_anchor=(0.98,0.02))
    plt.tight_layout(rect=[0.12,0.05,1,0.88])
    return fig

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.caption("Use the ◀ chevron to hide/show.")
    neb_b_df = df_all[df_all['BatterTeam']=='NEB'].copy()
    date_opts = sorted(neb_b_df['Date'].dropna().dt.date.unique().tolist())
    sel_date = st.selectbox("Game Date", options=date_opts, format_func=format_date_long) if date_opts else None

st.title("Nebraska Baseball")
st.subheader("Hitter Report")

if not sel_date:
    st.info("No Nebraska hitter dates available.")
    st.stop()

df_date = neb_b_df[neb_b_df['Date'].dt.date==sel_date]
batters = sorted(df_date['Batter'].dropna().unique().tolist())
batter  = st.selectbox("Batter", batters) if batters else None

if not batter:
    st.info("Choose a batter.")
    st.stop()

tabs = st.tabs(["Standard", "Heatmaps"])
with tabs[0]:
    fig = create_hitter_report(df_date, batter, ncols=3)
    if fig: st.pyplot(fig=fig)
with tabs[1]:
    fig = combined_hitter_heatmap_report(df_date, batter)
    if fig: st.pyplot(fig=fig)
