# hitter_app.py

import os
import math
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from matplotlib import colors

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nebraska Hitter Reports", layout="centered")  # wide OFF

DATA_PATH = "B10C25_hitter_app_columns.csv"  # update if needed
BANNER_CANDIDATES = [
    "NebraskaChampions.jpg",
  
]

HUSKER_RED = "#E60026"

# Big Ten / opponents pretty names
TEAM_NAME_MAP = {
    "ILL_ILL": "Illinois",
    "MIC_SPA": "Michigan State",
    "UCLA": "UCLA",
    "IOW_HAW": "Iowa",
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
# DATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
# BANNER (darker background like pitcher app)
# ──────────────────────────────────────────────────────────────────────────────
def _img_to_b64(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None

def render_nb_banner(
    image_candidates=BANNER_CANDIDATES,
    title="Nebraska Baseball",
    height_px=180
):
    b64 = None
    for p in image_candidates:
        b64 = _img_to_b64(p)
        if b64:
            break
    if not b64:
        return  # silently skip if not found

    # Add dark overlay + slight image darkening so title stands out
    st.markdown(
        f"""
        <div style="
            position: relative;
            width: 100%;
            height: {height_px}px;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 10px;">
          <img src="data:image/jpeg;base64,{b64}"
               style="width:100%; height:100%; object-fit:cover; filter: brightness(0.6);" />
          <div style="position:absolute; inset:0; background: rgba(0,0,0,0.35);"></div>
          <div style="
              position:absolute; inset:0;
              display:flex; align-items:center; justify-content:center;">
            <div style="
                font-size:40px; font-weight:800; color:white;
                text-shadow: 0 2px 12px rgba(0,0,0,.9);">
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
# HITTER HEATMAPS — 3 panels (Contact, Whiffs, Damage) using filtered data
# ──────────────────────────────────────────────────────────────────────────────
def hitter_heatmaps(df_filtered_for_profiles: pd.DataFrame, batter: str):
    sub = df_filtered_for_profiles[df_filtered_for_profiles.get('Batter') == batter].copy()
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
# LOAD DATA
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
render_nb_banner(title="Nebraska Baseball")

# Top section selector: Standard vs Profiles & Heatmaps
view_mode = st.radio("View", ["Standard Hitter Report", "Profiles & Heatmaps"], horizontal=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODE: STANDARD HITTER REPORT (batter → date with opponent names)
# ──────────────────────────────────────────────────────────────────────────────
if view_mode == "Standard Hitter Report":
    st.markdown("### Nebraska Hitter Reports")

    colB, colD = st.columns([1, 1])

    # Batter select (from all NEB hitters)
    batters_global = sorted(df_neb_bat.get('Batter').dropna().unique().tolist())
    batter_std = colB.selectbox("Player", options=batters_global, index=0 if batters_global else None)

    # Date options for the selected batter, with opponent label (mapped to names)
    if batter_std:
        df_b_all = df_neb_bat[df_neb_bat['Batter'] == batter_std].copy()
        df_b_all['DateOnly'] = pd.to_datetime(df_b_all['Date'], errors="coerce").dt.date
        # Opponents by date
        date_groups = df_b_all.groupby('DateOnly')['PitcherTeam'].agg(
            lambda s: sorted(set([TEAM_NAME_MAP.get(str(x), str(x)) for x in s if pd.notna(x)]))
        )
        date_opts = []
        date_labels = {}
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

    # Data for that single game
    if batter_std and selected_date:
        df_date = df_b_all[df_b_all['DateOnly'] == selected_date].copy()
    else:
        df_date = df_b_all.iloc[0:0].copy()

    # Standard Hitter Report
    if not batter_std or df_date.empty:
        st.info("Select a player and game date to see the Standard Hitter Report.")
    else:
        st.markdown("### Standard Hitter Report")
        fig_std = create_hitter_report(df_date, batter_std, ncols=3)
        if fig_std:
            st.pyplot(fig_std)

# ──────────────────────────────────────────────────────────────────────────────
# MODE: PROFILES & HEATMAPS (separate section with its own filters)
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("### Profiles & Heatmaps")

    # Batter select for profiles section
    batters_global = sorted(df_neb_bat.get('Batter').dropna().unique().tolist())
    batter = st.selectbox("Player", options=batters_global, index=0 if batters_global else None)

    st.markdown("#### Filters")

    # Give the hand radio more room so options fit in one line
    colM, colD2, colN, colH = st.columns([1.2, 1.2, 0.9, 1.9])

    # Months / Days (built from this batter's season)
    if batter:
        df_b_all = df_neb_bat[df_neb_bat['Batter'] == batter].copy()
        dates_all = pd.to_datetime(df_b_all['Date'], errors="coerce").dropna().dt.date
        present_months = sorted(pd.Series(dates_all).map(lambda d: d.month).unique().tolist())
    else:
        df_b_all = df_neb_bat.iloc[0:0].copy()
        dates_all = pd.Series([], dtype="datetime64[ns]")
        present_months = []

    sel_months = colM.multiselect(
        "Months",
        options=present_months,
        format_func=lambda n: MONTH_NAME_BY_NUM.get(n, str(n)),
        default=[],
        key="prof_months",
    )

    # Days derive from selected months
    if batter:
        dser = pd.Series(dates_all)
        if sel_months:
            dser = dser[dser.map(lambda d: d.month).isin(sel_months)]
        present_days = sorted(pd.Series(dser).map(lambda d: d.day).unique().tolist())
    else:
        present_days = []
    sel_days = colD2.multiselect("Days", options=present_days, default=[], key="prof_days")

    # Last N games
    lastN = int(colN.number_input("Last N games", min_value=0, max_value=50, step=1, value=0, format="%d", key="prof_lastn"))

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

    # Last N games (after month/day filter)
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

    # Render profiles & heatmaps
    if batter and df_profiles.empty:
        st.info("No rows for the selected filters.")
    elif batter:
        # Season column (year). If mixed, "Multiple".
        year_vals = pd.to_datetime(df_profiles['Date'], errors="coerce").dt.year.dropna().unique()
        if len(year_vals) == 1:
            season_year = int(year_vals[0])
        elif len(year_vals) > 1:
            season_year = "Multiple"
        else:
            # fallback if no dates but batter chosen
            season_year = "—"

        # Month column (if any months chosen)
        month_label = ", ".join(MONTH_NAME_BY_NUM.get(m, str(m)) for m in sorted(sel_months)) if sel_months else None

        # Opponents (mapped to names) within the filtered set
        opp_codes = df_profiles.get('PitcherTeam', pd.Series(dtype=object)).dropna().astype(str).unique().tolist()
        opp_fullnames = [TEAM_NAME_MAP.get(code, code) for code in sorted(opp_codes)]
        opp_label = "/".join(opp_fullnames) if opp_fullnames else None

        # Batted Ball Profile
        bb_df = create_batted_ball_profile(df_profiles).copy()
        for c in bb_df.columns:
            bb_df[c] = bb_df[c].apply(lambda v: fmt_pct(v, decimals=1))
        bb_df.insert(0, "Season", season_year)
        if month_label:
            bb_df.insert(1, "Month", month_label)
            if opp_label:
                bb_df.insert(2, "Opponent(s)", opp_label)
        st.markdown("#### Batted Ball Profile")
        st.table(themed_styler(bb_df))

        # Plate Discipline Profile
        pd_df = create_plate_discipline_profile(df_profiles).copy()
        if "Zone %" in pd_df.columns:
            pd_df["Zone %"] = pd_df["Zone %"].apply(lambda v: fmt_pct(v, decimals=1))
        if "Zone Contact %" in pd_df.columns:
            pd_df["Zone Contact %"] = pd_df["Zone Contact %"].apply(lambda v: fmt_pct(v, decimals=1))
        for c in ["Zone Swing %","Chase %","Swing %","Whiff %"]:
            if c in pd_df.columns:
                pd_df[c] = pd_df[c].apply(fmt_pct2)
        pd_df.insert(0, "Season", season_year)
        if month_label:
            pd_df.insert(1, "Month", month_label)
            if opp_label:
                pd_df.insert(2, "Opponent(s)", opp_label)
        st.markdown("#### Plate Discipline Profile")
        st.table(themed_styler(pd_df))

        # Batting Statistics
        st_df, pa_cnt, ab_cnt = create_batting_stats_profile(df_profiles)
        st_df = st_df.copy()
        # Format: AVG/OBP/SLG/OPS as .xxx
        for c in ["AVG", "OBP", "SLG", "OPS"]:
            if c in st_df.columns:
                st_df[c] = st_df[c].apply(fmt_avg3)
        # EV & LA fixed to two decimals
        if "Avg Exit Vel" in st_df.columns:
            st_df["Avg Exit Vel"] = st_df["Avg Exit Vel"].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "—")
        if "Max Exit Vel" in st_df.columns:
            st_df["Max Exit Vel"] = st_df["Max Exit Vel"].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "—")
        if "Avg Angle" in st_df.columns:
            st_df["Avg Angle"] = st_df["Avg Angle"].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "—")
        # Percentages
        if "HardHit %" in st_df.columns:
            st_df["HardHit %"] = st_df["HardHit %"].apply(lambda v: fmt_pct(v, decimals=1))
        if "K %" in st_df.columns:
            st_df["K %"] = st_df["K %"].apply(fmt_pct2)
        if "BB %" in st_df.columns:
            st_df["BB %"] = st_df["BB %"].apply(fmt_pct2)
        # Short headers to keep on one line
        rename_map = {
            "Avg Exit Vel": "Avg EV",
            "Max Exit Vel": "Max EV",
            "Avg Angle":    "Avg LA",
            "HardHit %":    "HardHit%",
            "K %":          "K%",
            "BB %":         "BB%",
        }
        st_df.rename(columns={k:v for k,v in rename_map.items() if k in st_df.columns}, inplace=True)
        st_df.insert(0, "Season", season_year)
        if month_label:
            st_df.insert(1, "Month", month_label)
            if opp_label:
                st_df.insert(2, "Opponent(s)", opp_label)

        st.markdown("#### Batting Statistics")
        st.table(themed_styler(st_df, nowrap=True))

        # Heatmaps (use the same filtered dataset as profiles)
        st.markdown("#### Hitter Heatmaps")
        fig_hm = hitter_heatmaps(df_profiles, batter)
        if fig_hm:
            st.pyplot(fig_hm)
