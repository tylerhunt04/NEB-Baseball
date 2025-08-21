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

# ───────────────────────────── Page setup ─────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Nebraska Baseball – Hitter Report",
    initial_sidebar_state="expanded"
)

# Fixed data paths (Nebraska-only app)
DATA_PATH  = "B10C25_streamlit_streamlit_columns.csv"
LOGO_PATH  = "Nebraska-Cornhuskers-Logo.png"
BANNER_IMG = "NebraskaChampions.jpg"  # optional banner

# ─────────────────────────── Helpers: dates ───────────────────────────
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
    if d is None or pd.isna(d): return ""
    d = pd.to_datetime(d).date()
    return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"

# ───────────────────── Column normalization (safe) ────────────────────
RENAME_MAP = {
    "PitchOfPA":"PitchofPA","PitchNumberInPA":"PitchofPA","Pitch#":"PitchofPA",
    "PAOfInning":"PAofInning","PAIndex":"PAofInning",
    "TopBottom":"Top/Bottom","HalfInning":"Top/Bottom","TB":"Top/Bottom",
    "Auto Pitch Type":"AutoPitchType","PitchType":"AutoPitchType","TaggedPitchType":"AutoPitchType",
    "Pitch Call":"PitchCall","Call":"PitchCall",
    "EffectiveVeloMPH":"EffectiveVelo","EVelo":"EffectiveVelo",
    "Batter Name":"Batter","BatterName":"Batter",
    "Pitcher Name":"Pitcher","PitcherName":"Pitcher","PitcherLastFirst":"Pitcher",
    "Pitcher Throws":"PitcherThrows","P_Throws":"PitcherThrows","PitcherHand":"PitcherThrows",
    "PlateLocSideX":"PlateLocSide","PlateLocHeightZ":"PlateLocHeight",
    "Game Id":"GameID","Game_ID":"GameID","InningNumber":"Inning",
}
REQUIRED_COLS = [
    "Batter","GameID","Inning","Top/Bottom","PAofInning","PitchofPA",
    "AutoPitchType","EffectiveVelo","PitchCall","PitcherThrows","Pitcher",
    "PlateLocSide","PlateLocHeight","ExitSpeed","PlayResult","Date","BatterTeam"
]
def normalize_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    need = {k: v for k, v in RENAME_MAP.items() if k in df.columns and v not in df.columns}
    if need:
        df = df.rename(columns=need)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ─────────────────────────── Load & cache ────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at: {DATA_PATH}")
    st.stop()

@st.cache_data(show_spinner=True)
def load_csv_norm(path: str, _mtime: float):
    df = pd.read_csv(path, low_memory=False)
    df = ensure_date_column(df)
    df = normalize_core_columns(df)
    return df

# Include file mtime in the cache-key without altering the path string
mtime = os.path.getmtime(DATA_PATH)
try:
    df_all = load_csv_norm(DATA_PATH, mtime)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Enforce Nebraska-only hitters
if "BatterTeam" not in df_all.columns:
    st.error("The data file is missing the 'BatterTeam' column. Cannot restrict to Nebraska hitters.")
    st.stop()

df_all = df_all[df_all["BatterTeam"] == "NEB"].copy()
if df_all.empty:
    st.info("No Nebraska hitter rows found in the data.")
    st.stop()

# ───────────────────────── Visual helpers ────────────────────────────
shape_map = {
    'Fastball':  'o',
    'Curveball': 's',
    'Slider':    '^',
    'Changeup':  'D'
}
color_map = {
    'StrikeCalled':         '#CCCC00',
    'BallCalled':           'green',
    'FoulBallNotFieldable': 'tan',
    'InPlay':               '#6699CC',
    'StrikeSwinging':       'red',
    'HitByPitch':           'lime'
}
def draw_strikezone(ax, left=-0.83, right=0.83, bottom=1.5, top=3.5):
    ax.add_patch(Rectangle((left, bottom), right-left, top-bottom,
                           fill=False, linewidth=2, color='black'))
    dx, dy = (right-left)/3, (top-bottom)/3
    for i in (1, 2):
        ax.add_line(Line2D([left+i*dx]*2, [bottom, top], linestyle='--', color='gray'))
        ax.add_line(Line2D([left, right], [bottom+i*dy]*2, linestyle='--', color='gray'))

# ───────────────────── Hitter report plotting (yours) ─────────────────
def create_hitter_report(fig_df: pd.DataFrame, batter: str, ncols: int = 3):
    bdf = fig_df[fig_df['Batter'] == batter].copy()
    if bdf.empty:
        st.info("No data for that batter on the chosen date.")
        return None

    pa_groups = list(bdf.groupby(['GameID','Inning','Top/Bottom','PAofInning'], sort=False))
    n_pa = len(pa_groups)
    nrows = max(1, math.ceil(n_pa / ncols))

    # textual descriptions per PA
    descriptions = []
    for _, pa_df in pa_groups:
        lines = []
        for _, p in pa_df.iterrows():
            try:
                pitch_no = int(pd.to_numeric(p.get("PitchofPA", np.nan), errors="coerce"))
            except Exception:
                pitch_no = "?"
            ptype = str(p.get("AutoPitchType", ""))
            velo  = pd.to_numeric(p.get("EffectiveVelo", np.nan), errors="coerce")
            velo_s= f"{velo:.1f}" if pd.notna(velo) else "—"
            call  = str(p.get("PitchCall", ""))
            lines.append(f"{pitch_no} / {ptype}  {velo_s} MPH / {call}")

        inplay = pa_df[pa_df['PitchCall']=='InPlay']
        if not inplay.empty:
            last = inplay.iloc[-1]
            res = last.PlayResult if pd.notna(last.PlayResult) else "InPlay"
            es  = pd.to_numeric(last.get("ExitSpeed", np.nan), errors="coerce")
            if pd.notna(es): res += f" ({es:.1f} MPH)"
            lines.append(f"  ▶ PA Result: {res}")
        else:
            balls = (pa_df['PitchCall']=='BallCalled').sum()
            strikes = pa_df['PitchCall'].isin(['StrikeCalled','StrikeSwinging']).sum()
            if balls >= 4:
                lines.append("  ▶ PA Result: Walk 🚶")
            elif strikes >= 3:
                lines.append("  ▶ PA Result: Strikeout 💥")

        descriptions.append(lines)

    # figure size & grid
    fig_h = (3.0 + 1.0) if nrows == 1 else (4*nrows)
    fig = plt.figure(figsize=(3 + 4*ncols + 1, fig_h))
    gs = GridSpec(nrows, ncols+1, width_ratios=[0.8]+[1]*ncols, wspace=0.1, hspace=0.35)

    # logo
    if os.path.exists(LOGO_PATH):
        logo = mpimg.imread(LOGO_PATH)
        ax_logo = fig.add_axes([0.88, 0.88, 0.12, 0.12], anchor='NE')
        ax_logo.imshow(logo); ax_logo.axis('off')

    # title & summary metrics
    report_date = pa_groups[0][1]['Date'].iloc[0]
    date_label = format_date_long(report_date) if pd.notna(report_date) else "—"
    fig.suptitle(f"{batter} Hitter Report for {date_label}",
                 fontsize=16, x=0.55, y=0.98, fontweight='bold')

    game_df = pd.concat([grp for _, grp in pa_groups], ignore_index=True)
    whiffs   = (game_df['PitchCall']=='StrikeSwinging').sum()
    hardhits = (pd.to_numeric(game_df['ExitSpeed'], errors='coerce') > 95).sum()
    chases   = game_df[
        (game_df['PitchCall']=='StrikeSwinging') &
        ((pd.to_numeric(game_df['PlateLocSide'], errors='coerce') < -0.83) |
         (pd.to_numeric(game_df['PlateLocSide'], errors='coerce') > 0.83) |
         (pd.to_numeric(game_df['PlateLocHeight'], errors='coerce') < 1.5) |
         (pd.to_numeric(game_df['PlateLocHeight'], errors='coerce') > 3.5))
    ].shape[0]
    fig.text(0.55, 0.94,
             f"Whiffs: {whiffs}    Hard Hits: {hardhits}    Chases: {chases}",
             ha='center', va='top', fontsize=12)

    # plot panels
    for idx, ((_, inn, tb, _), pa_df) in enumerate(pa_groups):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col+1])
        draw_strikezone(ax)

        throws = str(pa_df['PitcherThrows'].iloc[0]) if not pa_df.empty else ""
        hand_label = 'LHP' if throws.upper().startswith('L') else 'RHP'
        pitcher = pa_df['Pitcher'].iloc[0] if 'Pitcher' in pa_df and not pa_df.empty else "—"

        for _, p in pa_df.iterrows():
            mk = shape_map.get(p.get("AutoPitchType",""), 'o')
            clr= color_map.get(p.get("PitchCall",""), 'black')
            sz = 200 if p.get("AutoPitchType","") == 'Slider' else 150
            x  = pd.to_numeric(p.get("PlateLocSide", np.nan), errors='coerce')
            y  = pd.to_numeric(p.get("PlateLocHeight", np.nan), errors='coerce')
            if pd.notna(x) and pd.notna(y):
                ax.scatter(x, y, marker=mk, c=clr, s=sz, edgecolor='white', linewidth=1, zorder=2)
                pno_val = pd.to_numeric(p.get("PitchofPA", np.nan), errors="coerce")
                pitch_no = int(pno_val) if pd.notna(pno_val) else "?"
                y_off = -0.05 if p.get("AutoPitchType","")=='Slider' else 0
                ax.text(x, y+y_off, str(pitch_no), ha='center', va='center',
                        color='black', fontsize=6, fontweight='bold', zorder=3)

        ax.set_xlim(-3,3); ax.set_ylim(0,5)
        ax.set_xticks([]); ax.set_yticks([])
        inn_i = int(pd.to_numeric(inn, errors='coerce')) if pd.notna(pd.to_numeric(inn, errors='coerce')) else inn
        ax.set_title(f"PA {idx+1} | Inning {inn_i} {tb}",
                     fontsize=10, fontweight='bold')
        ax.text(0.5,0.1, f"vs {pitcher} ({hand_label})",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, style='italic')

    # descriptions column (with extra spacing if only one row of zones)
    ax_desc = fig.add_subplot(gs[:,0]); ax_desc.axis('off')
    lines_per_pa = 5.0
    denom = max(3.0, min(8.0, len(descriptions) * lines_per_pa))
    y0 = 1.0; dy = 1.0 / denom
    for i, lines in enumerate(descriptions, start=1):
        ax_desc.hlines(y0 - dy*0.1, 0, 1, transform=ax_desc.transAxes, color='black', linewidth=1)
        ax_desc.text(0.02, y0, f"PA {i}", fontsize=6, fontweight='bold', transform=ax_desc.transAxes)
        y_line = y0 - dy
        for ln in lines:
            ax_desc.text(0.02, y_line, ln, fontsize=6, transform=ax_desc.transAxes)
            y_line -= dy
        y0 = y_line - dy*0.05

    # legends
    res_handles = [Line2D([0],[0], marker='o', color='w',
                          label=name, markerfacecolor=colr,
                          markersize=10, markeredgecolor='k')
                   for name,colr in color_map.items()]
    fig.legend(res_handles, color_map.keys(),
               title='Result', loc='lower right',
               bbox_to_anchor=(0.90,0.02), frameon=True)

    pitch_handles = [Line2D([0],[0], marker=m, color='w',
                            label=name, markerfacecolor='gray',
                            markersize=10, markeredgecolor='k')
                     for name,m in shape_map.items()]
    fig.legend(pitch_handles, shape_map.keys(),
               title='Pitches', loc='lower right',
               bbox_to_anchor=(0.98,0.02), frameon=True)

    # add a larger bottom gap only if single row (keeps zone size unchanged)
    rect_bottom = 0.14 if nrows == 1 else 0.08
    plt.tight_layout(rect=[0.12, rect_bottom, 1, 0.88])
    return fig

# ────────────────────────────── UI / App ─────────────────────────────
# Banner (optional)
if os.path.exists(BANNER_IMG):
    st.image(BANNER_IMG, use_container_width=True)

st.subheader("Nebraska Hitter Report")

# Date → Batter (Nebraska-only)
available_dates = sorted(df_all['Date'].dropna().dt.date.unique().tolist())
if not available_dates:
    st.info("No valid dates found for Nebraska hitters.")
    st.stop()

col1, col2 = st.columns([1,2])
with col1:
    sel_date = st.selectbox("Game Date", options=available_dates, format_func=format_date_long)

df_date = df_all[df_all['Date'].dt.date == sel_date]
batters = sorted(df_date['Batter'].dropna().unique().tolist())

with col2:
    batter = st.selectbox("Batter (NEB)", batters) if batters else None

if not batter:
    st.info("Choose a Nebraska batter.")
    st.stop()

fig = create_hitter_report(df_date, batter, ncols=3)
if fig:
    st.pyplot(fig=fig)
