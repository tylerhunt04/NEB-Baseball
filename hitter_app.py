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

# ───────────────────────────── Page setup ─────────────────────────────
st.set_page_config(layout="wide", page_title="Nebraska Baseball – Hitter Report", initial_sidebar_state="expanded")

# Change this to your file path if different
DATA_PATH = "B10C25_streamlit_streamlit_columns.csv"

# Optional assets
LOGO_PATH = "Nebraska-Cornhuskers-Logo.png"
BANNER_IMG = "NebraskaChampions.jpg"  # shown at the very top if present

# ───────────────────────────── Helpers: dates ─────────────────────────
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

# ─────────────────────────── Normalization ───────────────────────────
RENAME_MAP = {
    # PA & pitch indexing
    "PitchOfPA": "PitchofPA",
    "PitchNumberInPA": "PitchofPA",
    "Pitch#": "PitchofPA",
    "PAOfInning": "PAofInning",
    "PAIndex": "PAofInning",
    "TopBottom": "Top/Bottom",
    "HalfInning": "Top/Bottom",
    "TB": "Top/Bottom",

    # Types & calls
    "Auto Pitch Type": "AutoPitchType",
    "PitchType": "AutoPitchType",
    "TaggedPitchType": "AutoPitchType",
    "Pitch Call": "PitchCall",
    "Call": "PitchCall",

    # Velo
    "EffectiveVeloMPH": "EffectiveVelo",
    "EVelo": "EffectiveVelo",

    # Names & handedness
    "Batter Name": "Batter",
    "BatterName": "Batter",
    "Pitcher Name": "Pitcher",
    "PitcherName": "Pitcher",
    "PitcherLastFirst": "Pitcher",
    "Pitcher Throws": "PitcherThrows",
    "P_Throws": "PitcherThrows",
    "PitcherHand": "PitcherThrows",

    # Location
    "PlateLocSideX": "PlateLocSide",
    "PlateLocHeightZ": "PlateLocHeight",

    # Misc
    "Game Id": "GameID",
    "Game_ID": "GameID",
    "InningNumber": "Inning",
}

def normalize_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    to_apply = {k: v for k, v in RENAME_MAP.items() if k in df.columns and v not in df.columns}
    if to_apply:
        df = df.rename(columns=to_apply)
    for col in [
        "GameID","Inning","Top/Bottom","PAofInning","PitchofPA",
        "AutoPitchType","EffectiveVelo","PitchCall","PitcherThrows",
        "Batter","Pitcher","PlateLocSide","PlateLocHeight",
        "ExitSpeed","PlayResult"
    ]:
        if col not in df.columns:
            df[col] = np.nan
    return df

# ─────────────────────────── Load & cache ────────────────────────────
@st.cache_data(show_spinner=True)
def load_csv_norm(path: str, mtime_key: float | None):
    df = pd.read_csv(path, low_memory=False)
    df = ensure_date_column(df)
    df = normalize_core_columns(df)
    return df

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at {DATA_PATH}")
    st.stop()

try:
    _mtime = os.path.getmtime(DATA_PATH)
except Exception:
    _mtime = None

df_all = load_csv_norm(DATA_PATH, _mtime)

# ─────────────────────────── Visual helpers ──────────────────────────
def get_pitch_color(ptype):
    if isinstance(ptype, str) and (ptype.lower().startswith("four-seam fastball") or ptype.lower() == "fastball"):
        return "#E60026"
    palette = {
        "sinker": "#FF9300","cutter": "#800080","changeup": "#008000","curveball": "#0033CC",
        "slider": "#CCCC00","splitter": "#00CCCC","knuckle curve": "#000000","screwball": "#CC0066","eephus": "#666666",
    }
    return palette.get(str(ptype).lower(), "#E60026")

def draw_strikezone(ax):
    left, bottom, width, height = -0.83, 1.17, 1.66, 2.75
    ax.add_patch(Rectangle((left, bottom), width, height, fill=False, lw=2, color="black"))
    for f in (1/3, 2/3):
        ax.vlines(left + width*f,  bottom, bottom+height, colors="gray", ls="--", lw=1)
        ax.hlines(bottom + height*f, left, left+width,     colors="gray", ls="--", lw=1)

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

def compute_density_hitter_plot(ax, sub, title):
    left, bottom, width, height = -0.83, 1.17, 1.66, 2.75
    x_min, x_max = left - width*0.8, left + width + width*0.8
    y_min, y_max = bottom - height*0.6, bottom + height + height*0.6

    draw_strikezone(ax)
    x = pd.to_numeric(sub.get('PlateLocSide', pd.Series(dtype=float)), errors='coerce').to_numpy()
    y = pd.to_numeric(sub.get('PlateLocHeight', pd.Series(dtype=float)), errors='coerce').to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 10:
        for xi, yi, row in zip(x, y, sub.loc[valid].itertuples()):
            color = get_pitch_color(getattr(row, 'AutoPitchType', ''))
            ax.plot(xi, yi, 'o', color=color, alpha=0.8, ms=6)
    else:
        xi = np.linspace(x_min, x_max, 200)
        yi = np.linspace(y_min, y_max, 200)
        xi_m, yi_m = np.meshgrid(xi, yi)
        zi = compute_density_hitter(
            pd.to_numeric(sub.get('PlateLocSide', pd.Series(dtype=float)), errors='coerce').to_numpy(),
            pd.to_numeric(sub.get('PlateLocHeight', pd.Series(dtype=float)), errors='coerce').to_numpy(),
            xi_m, yi_m
        )
        custom_cmap = colors.LinearSegmentedColormap.from_list(
            "custom_cmap",
            [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
            N=256,
        )
        ax.imshow(zi, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap=custom_cmap)
        draw_strikezone(ax)

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal', 'box')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

# ───────────────────── Hitter heatmaps report ────────────────────────
def combined_hitter_heatmap_report(df, batter, logo_img=None):
    df_b = df[df['Batter'] == batter].copy()
    if df_b.empty:
        st.error(f"No data for batter '{batter}' on that date.")
        return None

    df_b['iscontact'] = df_b['PitchCall'].isin(['InPlay','FoulBallFieldable','FoulBallNotFieldable'])
    df_b['iswhiff'] = df_b['PitchCall'].eq('StrikeSwinging')
    df_b['is95plus'] = pd.to_numeric(df_b['ExitSpeed'], errors='coerce') >= 95

    date_str = summarize_dates_range(df_b.get("Date", pd.Series(dtype="datetime64[ns]")))

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

    title_str = f"{batter} — {date_str}" if date_str else f"{batter}"
    fig.suptitle(title_str, fontsize=22, x=0.5, y=0.87)
    plt.tight_layout(rect=[0, 0, 1, 0.78])
    return fig

# ───────────────── Hitter report (pitch-by-pitch) ────────────────────
def _safe_pitch_number(row) -> str:
    for key in ("PitchofPA","PitchOfPA","PitchNumberInPA","Pitch#"):
        if key in row and pd.notna(row[key]):
            val = pd.to_numeric(row[key], errors="coerce")
            if pd.notna(val):
                return str(int(val))
    return "?"

def create_hitter_report(df, batter, ncols=3):
    bdf = df[df['Batter'] == batter].copy()
    if bdf.empty:
        st.error("No data for that hitter with current filters.")
        return None

    # Choose grouping columns; fallback if missing
    group_cols = [c for c in ['GameID','Inning','Top/Bottom','PAofInning'] if c in bdf.columns]
    if len(group_cols) >= 2:
        grouped = list(bdf.groupby(group_cols, sort=False))
    else:
        bdf = bdf.sort_values(['Date','GameID','Inning','Top/Bottom','PAofInning','PitchofPA'], axis=0, na_position='last')
        starts = pd.Series(False, index=bdf.index)
        if 'PitchofPA' in bdf.columns:
            starts |= (pd.to_numeric(bdf['PitchofPA'], errors='coerce') == 1).fillna(False)
        if 'PAofInning' in bdf.columns:
            starts |= bdf['PAofInning'].ne(bdf['PAofInning'].shift(1)).fillna(True)
        bdf['__grp__'] = starts.cumsum()
        grouped = list(bdf.groupby('__grp__', sort=False))

    n_pa = len(grouped); nrows = max(1, math.ceil(n_pa/ncols))
    descs = []

    fig = plt.figure(figsize=(3+4*ncols+1, 4*nrows))
    gs = GridSpec(nrows, ncols+1, width_ratios=[0.8]+[1]*ncols, wspace=0.1)

    # Logo (optional)
    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88,0.88,0.12,0.12], anchor='NE'); axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis('off')

    # Title
    date_title = summarize_dates_range(bdf.get("Date", pd.Series(dtype="datetime64[ns]")))
    fig.suptitle(f"{batter} Hitter Report{(' — ' + date_title) if date_title else ''}",
                 fontsize=16, x=0.55, y=1.0, fontweight='bold')

    # Global stats line
    gd = pd.concat([grp for _, grp in grouped]) if grouped else bdf
    whiffs   = (gd['PitchCall']=='StrikeSwinging').sum()
    hardhits = (pd.to_numeric(gd['ExitSpeed'], errors='coerce')>95).sum()
    chases   = gd[(gd['PitchCall']=='StrikeSwinging') &
                  ((pd.to_numeric(gd['PlateLocSide'], errors='coerce')<-0.83)|
                   (pd.to_numeric(gd['PlateLocSide'], errors='coerce')>0.83)|
                   (pd.to_numeric(gd['PlateLocHeight'], errors='coerce')<1.5)|
                   (pd.to_numeric(gd['PlateLocHeight'], errors='coerce')>3.5))].shape[0]
    fig.text(0.55,0.96,f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",
             ha='center', va='top', fontsize=12)

    # Left description panel
    axd = fig.add_subplot(gs[:,0]); axd.axis('off')
    y0=1.0; dy=1.0/(max(1,n_pa)*5.0)

    # Panels per PA
    for idx, (gkey, padf) in enumerate(grouped):
        lines=[]
        for _, p in padf.iterrows():
            pitch_no = _safe_pitch_number(p)
            vel = pd.to_numeric(p.get('EffectiveVelo', np.nan), errors='coerce')
            vel_str = f"{vel:.1f}" if pd.notna(vel) else "—"
            ptype = str(p.get('AutoPitchType', '—'))
            call  = str(p.get('PitchCall', '—'))
            lines.append(f"{pitch_no} / {ptype} {vel_str} MPH / {call}")

        ip=padf[padf['PitchCall']=='InPlay']
        if not ip.empty:
            last=ip.iloc[-1]; res=last.PlayResult if pd.notna(last.PlayResult) else 'InPlay'
            es=pd.to_numeric(last.ExitSpeed, errors='coerce')
            if pd.notna(es): res+=f" ({es:.1f} MPH)"
            lines.append(f"▶ PA Result: {res}")
        else:
            balls=(padf['PitchCall']=='BallCalled').sum()
            strikes=padf['PitchCall'].isin(['StrikeCalled','StrikeSwinging']).sum()
            if balls>=4: lines.append('▶ PA Result: Walk')
            elif strikes>=3: lines.append('▶ PA Result: Strikeout')
        descs.append(lines)

        # Plot this PA
        row,col=divmod(idx,ncols)
        ax=fig.add_subplot(gs[row,col+1]); draw_strikezone(ax)
        for _,p in padf.iterrows():
            mk={'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}.get(p.get('AutoPitchType',''), 'o')
            clr={'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan','InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.get(p.get('PitchCall',''), 'black')
            sz=200 if p.get('AutoPitchType','')=='Slider' else 150
            x = pd.to_numeric(p.get('PlateLocSide', np.nan), errors='coerce')
            y = pd.to_numeric(p.get('PlateLocHeight', np.nan), errors='coerce')
            if pd.notna(x) and pd.notna(y):
                ax.scatter(x, y, marker=mk, c=clr, s=sz, edgecolor='white', lw=1, zorder=2)
                # Label with pitch number (robust)
                pitch_no = _safe_pitch_number(p)
                yoff=-0.05 if p.get('AutoPitchType','')=='Slider' else 0
                ax.text(x, y+yoff, pitch_no, ha='center', va='center', fontsize=6, fontweight='bold', zorder=3)

        ax.set_xlim(-3,3); ax.set_ylim(0,5); ax.set_xticks([]); ax.set_yticks([])
        inn = padf.get('Inning'); inn = int(pd.to_numeric(inn.iloc[0], errors='coerce')) if hasattr(inn, "iloc") and pd.notna(inn.iloc[0]) else "—"
        tb  = str(padf.get('Top/Bottom', pd.Series(['—'])).iloc[0])
        pitchr = str(padf.get('Pitcher', pd.Series(['—'])).iloc[0])
        throws = str(padf.get('PitcherThrows', pd.Series(['—'])).iloc[0])
        hand='LHP' if str(throws).startswith('L') else 'RHP'
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}", fontsize=10, fontweight='bold')
        ax.text(0.5,0.1,f"vs {pitchr} ({hand})", transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic')

    # Write the left column details
    for i,lines in enumerate(descs,1):
        axd.hlines(y0-dy*0.1,0,1, transform=axd.transAxes, color='black', lw=1)
        axd.text(0.02,y0,f"PA {i}", fontsize=6, fontweight='bold', transform=axd.transAxes)
        yln=y0-dy
        for ln in lines:
            axd.text(0.02,yln,ln, fontsize=6, transform=axd.transAxes); yln-=dy
        y0=yln-dy*0.05

    # Legends
    res_handles=[Line2D([0],[0], marker='o', color='w', label=k, markerfacecolor=v, ms=10, markeredgecolor='k')
                 for k,v in {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan','InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.items()]
    fig.legend(res_handles, [h.get_label() for h in res_handles], title='Result', loc='lower right', bbox_to_anchor=(0.90,0.02))
    pitch_handles=[Line2D([0],[0], marker=m, color='w', label=k, markerfacecolor='gray', ms=10, markeredgecolor='k')
                   for k,m in {'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}.items()]
    fig.legend(pitch_handles, [h.get_label() for h in pitch_handles], title='Pitches', loc='lower right', bbox_to_anchor=(0.98,0.02))
    plt.tight_layout(rect=[0.12,0.05,1,0.88])
    return fig

# ────────────────────────────── UI / App ─────────────────────────────
# Banner (optional)
if os.path.exists(BANNER_IMG):
    st.image(BANNER_IMG, use_container_width=True)

st.subheader("Nebraska Hitter Report")

# Filter to Nebraska hitters, then pick a date and a batter on that date
neb_b_df = df_all[df_all.get('BatterTeam', '') == 'NEB'].copy()
date_opts = sorted(neb_b_df['Date'].dropna().dt.date.unique().tolist())
sel_date = st.selectbox("Game Date", options=date_opts, format_func=format_date_long) if date_opts else None

if not sel_date:
    st.info("No Nebraska hitter dates available."); st.stop()

df_date = neb_b_df[neb_b_df['Date'].dt.date==sel_date].copy()
batters = sorted(df_date['Batter'].dropna().unique().tolist())
batter  = st.selectbox("Batter", batters) if batters else None

if not batter:
    st.info("Choose a batter."); st.stop()

tabs = st.tabs(["Standard", "Heatmaps"])

with tabs[0]:
    fig = create_hitter_report(df_date, batter, ncols=3)
    if fig: st.pyplot(fig=fig)

with tabs[1]:
    fig = combined_hitter_heatmap_report(df_date, batter)
    if fig: st.pyplot(fig=fig)
