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
from numpy.linalg import LinAlgError
from matplotlib import colors

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG / PATHS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Nebraska Hitter Reports")

DATA_PATH = "B10C25_hitter_app_columns.csv"  # update if needed
BANNER_CANDIDATES = [
    "NebraskaChampions.jpg",
  
]

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS — DATES, FORMATTING
# ──────────────────────────────────────────────────────────────────────────────
def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Try common date columns; if none, create empty
    cand = None
    lower = {c.lower(): c for c in df.columns}
    for name in ["date", "gamedate", "game date", "datetime", "game_datetime", "gamedatetime"]:
        if name in lower:
            cand = lower[name]
            break
    if cand is None:
        df["Date"] = pd.NaT
        return df
    dt = pd.to_datetime(df[cand], errors="coerce")
    df["Date"] = pd.to_datetime(dt.dt.date, errors="coerce")
    return df

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

# Keep the same axes across all panels so visual size never changes
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
# FORMATTING HELPERS
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

def fmt_one_decimal(x):
    try:
        if pd.isna(x): return "—"
        return f"{round(float(x), 1)}"
    except Exception:
        return "—"

# ──────────────────────────────────────────────────────────────────────────────
# BATTED BALL / DISCIPLINE / STATS METRIC CALC
# (robust to missing optional columns)
# ──────────────────────────────────────────────────────────────────────────────
def _safe(series, default=np.nan):
    return series if series is not None else pd.Series(dtype=type(default))

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
    # Requires: PitchCall, TaggedHitType (optional), Bearing (optional), BatterSide (optional)
    inplay = df[df.get('PitchCall', pd.Series(dtype=object)) == 'InPlay'].copy()
    if 'TaggedHitType' not in inplay.columns:
        # Graceful fallback if column was dropped in your slim CSV
        inplay['TaggedHitType'] = pd.NA
    if 'Bearing' not in inplay.columns:
        inplay['Bearing'] = np.nan
    if 'BatterSide' not in inplay.columns:
        inplay['BatterSide'] = ""

    inplay['spray_cat'] = inplay.apply(assign_spray_category, axis=1)

    def pct(mask):
        try:
            return round(100 * float(np.nanmean(mask.astype(float))), 1) if len(mask) else 0.0
        except Exception:
            return 0.0

    bb = pd.DataFrame([{
        "Ground ball %": pct(inplay["TaggedHitType"].astype(str).str.contains("GroundBall", case=False, na=False)),
        "Fly ball %":    pct(inplay["TaggedHitType"].astype(str).str.contains("FlyBall",   case=False, na=False)),
        "Line drive %":  pct(inplay["TaggedHitType"].astype(str).str.contains("LineDrive", case=False, na=False)),
        "Popup %":       pct(inplay["TaggedHitType"].astype(str).str.contains("Popup",     case=False, na=False)),
        "Pull %":        pct(inplay["spray_cat"].astype(str).eq("Pull")),
        "Straight %":    pct(inplay["spray_cat"].astype(str).eq("Straight")),
        "Opposite %":    pct(inplay["spray_cat"].astype(str).eq("Opposite")),
    }])
    return bb

def create_plate_discipline_profile(df: pd.DataFrame):
    # Uses PlateLocSide/PlateLocHeight; robust if missing
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
    s_call = df.get('PitchCall', pd.Series(dtype=object))
    play   = df.get('PlayResult', pd.Series(dtype=object))
    korbb  = df.get('KorBB', pd.Series(dtype=object))
    exitv  = pd.to_numeric(df.get('ExitSpeed', pd.Series(dtype=float)), errors="coerce")
    angle  = pd.to_numeric(df.get('Angle', pd.Series(dtype=float)), errors="coerce")
    pitchofpa = pd.to_numeric(df.get('PitchofPA', pd.Series(dtype=float)), errors="coerce")

    pa_mask   = pitchofpa.eq(1)
    hit_mask  = (s_call.eq('InPlay') & play.isin(['Single','Double','Triple','HomeRun']))
    so_mask   = korbb.eq('Strikeout')
    bbout     = s_call.eq('InPlay') & play.eq('Out')
    fc_mask   = play.eq('FieldersChoice')
    err_mask  = play.eq('Error')
    walk_mask = korbb.eq('Walk')
    hbp_mask  = s_call.eq('HitByPitch')

    hits = int(hit_mask.sum())
    so   = int(so_mask.sum())
    bbouts = int(bbout.sum())
    fc   = int(fc_mask.sum())
    err  = int(err_mask.sum())
    ab   = hits + so + bbouts + fc + err

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
        "Avg Exit Vel": round(avg_exit, 1) if pd.notna(avg_exit) else np.nan,
        "Max Exit Vel": round(max_exit, 1) if pd.notna(max_exit) else np.nan,
        "Avg Angle":    round(avg_angle, 1) if pd.notna(avg_angle) else np.nan,
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
# STANDARD HITTER REPORT (single game)
# ──────────────────────────────────────────────────────────────────────────────
def create_hitter_report(df, batter, ncols=3):
    bdf = df[df.get('Batter') == batter]
    pa_groups = list(bdf.groupby(['GameID','Inning','Top/Bottom','PAofInning']))
    n_pa = len(pa_groups)
    nrows = max(1, math.ceil(n_pa / ncols))

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
            if pd.notna(es):
                res += f" ({float(es):.1f} MPH)"
            lines.append(f"  ▶ PA Result: {res}")
        else:
            balls = (pa_df.get('PitchCall')=='BallCalled').sum()
            strikes = pa_df.get('PitchCall').isin(['StrikeCalled','StrikeSwinging']).sum()
            if balls >= 4:
                lines.append("  ▶ PA Result: Walk 🚶")
            elif strikes >= 3:
                lines.append("  ▶ PA Result: Strikeout 💥")
        descriptions.append(lines)

    fig = plt.figure(figsize=(3 + 4*ncols + 1, 4*nrows))
    gs = GridSpec(nrows, ncols+1, width_ratios=[0.8] + [1]*ncols, wspace=0.15, hspace=0.55)

    # Optional date title
    if pa_groups:
        date = pa_groups[0][1].get('Date').iloc[0]
        date_str = format_date_long(date)
        fig.suptitle(f"{batter} Hitter Report for {date_str}",
                     fontsize=16, x=0.55, y=1.0, fontweight='bold')

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
        draw_strikezone(ax)  # fixed size
        # scatter of pitches with numbers
        hand_lbl = "RHP"
        thr = str(pa_df.get('PitcherThrows').iloc[0]) if not pa_df.empty else ""
        if thr.upper().startswith('L'): hand_lbl = "LHP"
        pitcher = str(pa_df.get('Pitcher').iloc[0]) if not pa_df.empty else "—"

        for _, p in pa_df.iterrows():
            mk = {'Fastball':'o', 'Curveball':'s', 'Slider':'^', 'Changeup':'D'}.get(str(p.get('AutoPitchType')), 'o')
            clr = {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan',
                   'InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.get(str(p.get('PitchCall')), 'black')
            sz = 200 if str(p.get('AutoPitchType'))=='Slider' else 150
            x = p.get('PlateLocSide')
            y = p.get('PlateLocHeight')
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

    # legends (nudged lower)
    res_handles = [Line2D([0],[0], marker='o', color='w', label=k,
                          markerfacecolor=v, markersize=10, markeredgecolor='k')
                   for k,v in {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan',
                               'InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.items()]
    fig.legend(res_handles, [h.get_label() for h in res_handles],
               title='Result', loc='lower right', bbox_to_anchor=(0.90, 0.015))

    pitch_handles = [Line2D([0],[0], marker=m, color='w', label=k,
                             markerfacecolor='gray', markersize=10, markeredgecolor='k')
                     for k,m in {'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}.items()]
    fig.legend(pitch_handles, [h.get_label() for h in pitch_handles],
               title='Pitches', loc='lower right', bbox_to_anchor=(0.98, 0.015))

    plt.tight_layout(rect=[0.12, 0.06, 1, 0.92])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# HITTER HEATMAPS (works for any filtered subset)
# ──────────────────────────────────────────────────────────────────────────────
def hitter_heatmaps(df_b: pd.DataFrame, batter: str):
    sub = df_b[df_b.get('Batter') == batter].copy()
    if sub.empty:
        return None

    sub['iscontact'] = sub.get('PitchCall').isin(['InPlay','FoulBallFieldable','FoulBallNotFieldable'])
    sub['iswhiff']   = sub.get('PitchCall').eq('StrikeSwinging')
    sub['is95plus']  = pd.to_numeric(sub.get('ExitSpeed'), errors="coerce") >= 95

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

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

    # LHP / RHP splits + contact/whiff/damage
    sub_l  = sub[sub.get('PitcherThrows').astype(str).str.upper().str.startswith('L')]
    sub_r  = sub[sub.get('PitcherThrows').astype(str).str.upper().str.startswith('R')]

    ax1 = fig.add_subplot(gs[0, 0]); _panel(ax1, "Contact vs LHP", sub_l[sub_l['iscontact']])
    ax2 = fig.add_subplot(gs[0, 2]); _panel(ax2, "Contact vs RHP", sub_r[sub_r['iscontact']])
    ax3 = fig.add_subplot(gs[0, 3]); _panel(ax3, "Whiffs vs LHP",  sub_l[sub_l['iswhiff']])
    ax4 = fig.add_subplot(gs[0, 5]); _panel(ax4, "Whiffs vs RHP",  sub_r[sub_r['iswhiff']])
    ax5 = fig.add_subplot(gs[0, 6]); _panel(ax5, "Exit ≥95 vs LHP", sub_l[sub_l['is95plus']])
    ax6 = fig.add_subplot(gs[0, 8]); _panel(ax6, "Exit ≥95 vs RHP", sub_r[sub_r['is95plus']])

    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    return ensure_date_column(df)

df_all = load_csv(DATA_PATH)
if df_all.empty:
    st.error("No data loaded.")
    st.stop()

# Nebraska hitters only
df_neb_bat = df_all[df_all.get('BatterTeam') == 'NEB'].copy()
if df_neb_bat.empty:
    st.error("No Nebraska hitter rows found in the dataset.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# BANNER
# ──────────────────────────────────────────────────────────────────────────────
def show_banner():
    for p in BANNER_CANDIDATES:
        if os.path.exists(p):
            st.image(p, use_container_width=True)
            return
    # If not found, do nothing silently

show_banner()

st.markdown("### Nebraska Hitter Reports")

# ──────────────────────────────────────────────────────────────────────────────
# TOP FILTERS — Standard Hitter Report (single game only)
# ──────────────────────────────────────────────────────────────────────────────
date_opts = sorted(pd.to_datetime(df_neb_bat['Date'], errors="coerce").dropna().dt.date.unique().tolist())
colA, colB = st.columns([1, 1])

selected_date = colA.selectbox("Game Date", options=date_opts, format_func=format_date_long, index=len(date_opts)-1 if date_opts else 0)
df_date = df_neb_bat[pd.to_datetime(df_neb_bat['Date'], errors="coerce").dt.date == selected_date]

batters_list = sorted(df_date.get('Batter').dropna().unique().tolist())
batter = colB.selectbox("Player", options=batters_list, index=0 if batters_list else None)

if not batter:
    st.info("Select a player to see the Standard Hitter Report.")
else:
    st.markdown("### Standard Hitter Report")
    fig_std = create_hitter_report(df_date, batter, ncols=3)
    if fig_std:
        st.pyplot(fig_std)

# ──────────────────────────────────────────────────────────────────────────────
# LOWER FILTERS — apply to Profiles & Heatmaps (not the Standard single-game)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Profiles & Heatmaps Filters")

colM, colD, colN, colH = st.columns([1.2, 1.2, 0.8, 1.0])

# Months / Days
present_dates_all = pd.to_datetime(df_neb_bat[df_neb_bat['Batter'] == batter]['Date'], errors="coerce").dropna().dt.date if batter else pd.Series([], dtype="datetime64[ns]")
present_months = sorted(pd.Series(present_dates_all).map(lambda d: d.month).unique().tolist()) if not present_dates_all.empty else []
sel_months = colM.multiselect(
    "Months (optional)",
    options=present_months,
    format_func=lambda n: MONTH_NAME_BY_NUM.get(n, str(n)),
    default=[],
    key="prof_months",
)

# Days derive from selected months (for this batter)
if batter:
    dser = pd.Series(present_dates_all)
    if sel_months:
        dser = dser[dser.map(lambda d: d.month).isin(sel_months)]
    present_days = sorted(pd.Series(dser).map(lambda d: d.day).unique().tolist())
else:
    present_days = []
sel_days = colD.multiselect("Days (optional)", options=present_days, default=[], key="prof_days")

# Last N games
lastN = int(colN.number_input("Last N games (optional)", min_value=0, max_value=50, step=1, value=0, format="%d", key="prof_lastn"))

# Pitcher Hand
hand_choice = colH.radio("Pitcher Hand", ["Both","LHP","RHP"], index=0, horizontal=True, key="prof_hand")

# Build filtered dataset for profiles/heatmaps
if batter:
    df_player_all = df_neb_bat[df_neb_bat['Batter'] == batter].copy()
else:
    df_player_all = df_neb_bat.iloc[0:0].copy()

# Month/Day filter
if sel_months:
    mask_m = pd.to_datetime(df_player_all['Date'], errors="coerce").dt.month.isin(sel_months)
else:
    mask_m = pd.Series(True, index=df_player_all.index)
if sel_days:
    mask_d = pd.to_datetime(df_player_all['Date'], errors="coerce").dt.day.isin(sel_days)
else:
    mask_d = pd.Series(True, index=df_player_all.index)
df_profiles = df_player_all[mask_m & mask_d].copy()

# Last N games filter (applies after month/day filter, if >0)
if lastN and not df_profiles.empty:
    uniq_dates = pd.to_datetime(df_profiles['Date'], errors="coerce").dt.date.dropna().unique()
    uniq_dates = sorted(uniq_dates)
    last_dates = set(uniq_dates[-lastN:])
    df_profiles = df_profiles[pd.to_datetime(df_profiles['Date'], errors="coerce").dt.date.isin(last_dates)].copy()

# Pitcher hand filter
if hand_choice == "LHP":
    df_profiles = df_profiles[df_profiles.get('PitcherThrows').astype(str).str.upper().str.startswith('L')].copy()
elif hand_choice == "RHP":
    df_profiles = df_profiles[df_profiles.get('PitcherThrows').astype(str).str.upper().str.startswith('R')].copy()

# ──────────────────────────────────────────────────────────────────────────────
# PROFILES (stacked) + HEATMAPS on filtered data
# ──────────────────────────────────────────────────────────────────────────────
if batter and df_profiles.empty:
    st.info("No rows for the selected filters.")
elif batter:
    # Season year for first column — if mixed years, show "Multiple"
    year_vals = pd.to_datetime(df_profiles['Date'], errors="coerce").dt.year.dropna().unique()
    if len(year_vals) == 1:
        season_year = int(year_vals[0])
    elif len(year_vals) > 1:
        season_year = "Multiple"
    else:
        # fallback to selected_date year if available
        season_year = int(pd.to_datetime(selected_date).year) if selected_date else "—"

    # Batted Ball Profile
    bb_df = create_batted_ball_profile(df_profiles).copy()
    for c in bb_df.columns:
        bb_df[c] = bb_df[c].apply(lambda v: fmt_pct(v, decimals=1))
    bb_df.insert(0, "Season", season_year)
    st.markdown("### Batted Ball Profile")
    st.table(bb_df)

    # Plate Discipline Profile
    pd_df = create_plate_discipline_profile(df_profiles).copy()
    # One decimal for Zone % & Zone Contact %
    for c in ["Zone %", "Zone Contact %"]:
        if c in pd_df.columns:
            pd_df[c] = pd_df[c].apply(lambda v: fmt_pct(v, decimals=1))
    # Two decimals for swing/whiff style rates
    for c in ["Zone Swing %", "Chase %", "Swing %", "Whiff %"]:
        if c in pd_df.columns:
            pd_df[c] = pd_df[c].apply(fmt_pct2)
    pd_df.insert(0, "Season", season_year)
    st.markdown("### Plate Discipline Profile")
    st.table(pd_df)

    # Batting Statistics
    st_df, pa_cnt, ab_cnt = create_batting_stats_profile(df_profiles)
    st_df = st_df.copy()
    # Exit velo: 1 decimal already; AVG/OBP/SLG/OPS exact formatting
    for c in ["AVG", "OBP", "SLG", "OPS"]:
        if c in st_df.columns:
            st_df[c] = st_df[c].apply(fmt_avg3)
    if "HardHit %" in st_df.columns:
        st_df["HardHit %"] = st_df["HardHit %"].apply(lambda v: fmt_pct(v, decimals=1))
    if "K %" in st_df.columns:
        st_df["K %"] = st_df["K %"].apply(fmt_pct2)
    if "BB %" in st_df.columns:
        st_df["BB %"] = st_df["BB %"].apply(fmt_pct2)
    st_df.insert(0, "Season", season_year)
    st.markdown("### Batting Statistics")
    st.table(st_df)

    # Heatmaps
    st.markdown("### Hitter Heatmaps")
    fig_hm = hitter_heatmaps(df_profiles, batter)
    if fig_hm:
        st.pyplot(fig_hm)
