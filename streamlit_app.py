# streamlit_app.py
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
from matplotlib import colors
from scipy.stats import chi2, gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Baseball Analytics", layout="wide")
APP_BUILD = "NB-full-restore-2025-08-18"
st.caption(f"Build: {APP_BUILD}")

# Point to your current file:
CSV_PATH = "B10C25_streamlit_streamlit_columns.csv"
LOGO_PATH = "Nebraska-Cornhuskers-Logo.png"

# Fixed strike-zone & view extents
SZ_LEFT, SZ_RIGHT = -0.83, 0.83
SZ_BOTTOM, SZ_TOP = 1.17, 3.92
SZ_W, SZ_H = SZ_RIGHT - SZ_LEFT, SZ_TOP - SZ_BOTTOM
VIEW_X_MARGIN = SZ_W * 0.8
VIEW_Y_MARGIN = SZ_H * 0.4
X_MIN, X_MAX = SZ_LEFT - VIEW_X_MARGIN, SZ_RIGHT + VIEW_X_MARGIN
Y_MIN, Y_MAX = SZ_BOTTOM - VIEW_Y_MARGIN, SZ_TOP + VIEW_Y_MARGIN

HEATMAP_GRID = 200
custom_cmap = colors.LinearSegmentedColormap.from_list(
    'custom_cmap',
    [(0.0, 'white'), (0.2, 'deepskyblue'), (0.3, 'white'), (0.7, 'red'), (1.0, 'red')],
    N=256
)

# Power 4 conference → team-code mapping (extend as needed)
POWER4 = {
    "Big Ten": {"ILL_ILL","MIC_SPA","UCLA","IOW_HAW","IU","MAR_TER","MIC_WOL","MIN_GOL",
                "NEB","NOR_CAT","ORE_DUC","OSU_BUC","PEN_NIT","PUR_BOI","RUT_SCA","SOU_TRO","WAS_HUS"},
    "Big 12": set(),  # fill with your codes if present in CSV
    "SEC":    set(),
    "ACC":    set(),
}

TEAM_MAP = {
    'ILL_ILL': 'Illinois','MIC_SPA':'Michigan State','UCLA':'UCLA','IOW_HAW':'Iowa','IU':'Indiana',
    'MAR_TER':'Maryland','MIC_WOL':'Michigan','MIN_GOL':'Minnesota','NEB':'Nebraska','NOR_CAT':'Northwestern',
    'ORE_DUC':'Oregon','OSU_BUC':'Ohio State','PEN_NIT':'Penn State','PUR_BOI':'Purdue','RUT_SCA':'Rutgers',
    'SOU_TRO':'USC','WAS_HUS':'Washington'
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_pitch_color(ptype: str) -> str:
    if not isinstance(ptype, str):
        return "#7F7F7F"
    s = ptype.strip().lower()
    if s.startswith("four-seam") or s == "fastball":
        return "#E60026"
    savant = {
        'sinker': '#FF9300', 'cutter': '#800080', 'changeup': '#008000',
        'curveball': '#0033CC', 'slider': '#CCCC00', 'splitter': '#00CCCC',
        'knuckle curve': '#000000', 'screwball': '#CC0066', 'eephus': '#666666',
        'sweeper': '#B5651D'
    }
    return savant.get(s, "#E60026")

def canonicalize_type(raw: str) -> str:
    if not isinstance(raw, str):
        return "Unknown"
    s = raw.strip().lower()
    if "sinker" in s:
        return "Fastball"
    mapping = {
        "four-seam": "Fastball", "four-seam fastball": "Fastball",
        "4-seam": "Fastball", "4-seam fastball": "Fastball",
        "fastball": "Fastball",
        "two-seam": "Two-Seam Fastball", "2-seam": "Two-Seam Fastball", "two-seam fastball": "Two-Seam Fastball",
        "cutter": "Cutter", "changeup": "Changeup", "splitter": "Splitter",
        "curveball": "Curveball", "knuckle curve": "Knuckle Curve", "slider": "Slider",
        "sweeper": "Sweeper", "screwball": "Screwball", "eephus": "Eephus"
    }
    return mapping.get(s, "Unknown")

def color_for_canon(label: str) -> str:
    l = (label or "").strip().lower()
    return {
        "fastball":"#E60026","two-seam fastball":"#FF9300","cutter":"#800080",
        "changeup":"#008000","splitter":"#00CCCC","curveball":"#0033CC",
        "knuckle curve":"#000000","slider":"#CCCC00","sweeper":"#B5651D",
        "screwball":"#CC0066","eephus":"#666666"
    }.get(l, "#7F7F7F")

def last_first_to_first_last(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    if "," in name:
        last, first = [t.strip() for t in name.split(",", 1)]
        return f"{first} {last}"
    return name.strip()

def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10,'th')}"

def _fmt_mdY(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"

def summarize_dates_range(series_like) -> str:
    if series_like is None:
        return "Season"
    ser = pd.Series(series_like)
    ser = pd.to_datetime(ser, errors="coerce").dropna()
    if ser.empty:
        return "Season"
    uniq = ser.dt.normalize().unique()
    if len(uniq) == 1:
        return _fmt_mdY(uniq[0])
    # If all are same month: show single month label
    months = pd.to_datetime(uniq).month
    if len(set(months)) == 1:
        m = pd.to_datetime(uniq[0]).month
        return pd.Timestamp(month= m, year=ser.dt.year.min(), day=1).strftime("(%B)")
    return f"{_fmt_mdY(ser.min())} – {_fmt_mdY(ser.max())}"

def month_name_map():
    return {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
            7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

def draw_strikezone(ax):
    ax.add_patch(Rectangle((SZ_LEFT, SZ_BOTTOM), SZ_W, SZ_H, fill=False, lw=2, color='black'))
    for frac in (1/3, 2/3):
        ax.vlines(SZ_LEFT + SZ_W * frac, SZ_BOTTOM, SZ_BOTTOM + SZ_H, colors='gray', ls='--', lw=1)
        ax.hlines(SZ_BOTTOM + SZ_H * frac, SZ_LEFT, SZ_LEFT + SZ_W, colors='gray', ls='--', lw=1)

def compute_density(x, y, xi_m, yi_m):
    coords = np.vstack([x, y])
    mask = np.isfinite(coords).all(axis=0)
    if mask.sum() < 2:
        return np.zeros_like(xi_m)
    try:
        kde = gaussian_kde(coords[:, mask])
        zi = kde(np.vstack([xi_m.ravel(), yi_m.ravel()])).reshape(xi_m.shape)
        return zi
    except Exception:
        return np.zeros_like(xi_m)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors="coerce")
        df['DateOnly'] = df['Date'].dt.date
        df['Month'] = df['Date'].dt.month
        df['Year']  = df['Date'].dt.year
    return df

df_all = load_data(CSV_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# REPORTS (Pitcher Full, Heatmaps, Release Points)
# ─────────────────────────────────────────────────────────────────────────────
def combined_pitcher_report(df_p: pd.DataFrame, pitcher_name: str, logo_img=None, coverage=0.8):
    if df_p.empty:
        return None
    total = len(df_p)
    is_strike = df_p['PitchCall'].isin(['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay'])
    grp = df_p.groupby('AutoPitchType', dropna=False)
    summary = pd.DataFrame({
        'Pitch Type': grp.size().index,
        'Pitches':    grp.size().values,
        'Usage %':    (grp.size().values / total * 100) if total > 0 else 0,
        'Strike %':   grp.apply(lambda g: is_strike.loc[g.index].mean() * 100 if len(g) else 0).values,
        'Rel Speed':  grp['RelSpeed'].mean().values,
        'Spin Rate':  grp['SpinRate'].mean().values,
        'IVB':        grp['InducedVertBreak'].mean().values,
        'HB':         grp['HorzBreak'].mean().values,
        'Rel Height': grp['RelHeight'].mean().values,
        'VAA':        grp['VertApprAngle'].mean().values,
        'Extension':  grp['Extension'].mean().values
    }).round({'Usage %':1,'Strike %':1,'Rel Speed':1,'Spin Rate':1,'IVB':1,'HB':1,'Rel Height':2,'VAA':1,'Extension':2})
    summary = summary.sort_values('Pitches', ascending=False)
    cols = ['Pitch Type','Pitches','Usage %','Strike %','Rel Speed','Spin Rate','IVB','HB','Rel Height','VAA','Extension']
    summary = summary[cols]

    fig = plt.figure(figsize=(8, 12))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.7], hspace=0.3)

    axm = fig.add_subplot(gs[0, 0])
    axm.set_title('Movement Plot', fontweight='bold')
    chi2v = chi2.ppf(0.8 if coverage is None else coverage, df=2)
    axm.axhline(0, ls='--', color='grey'); axm.axvline(0, ls='--', color='grey')
    for ptype, g in grp:
        x, y = g['HorzBreak'], g['InducedVertBreak']
        clr = get_pitch_color(ptype if isinstance(ptype, str) else "")
        axm.scatter(x, y, color=clr, alpha=0.7, label=str(ptype))
        if len(g) > 1 and np.isfinite(x).sum() > 1 and np.isfinite(y).sum() > 1:
            cov = np.cov(np.vstack((x, y)))
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            ang = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            w, h = 2 * np.sqrt(np.maximum(vals, 0) * chi2v)
            ell = Ellipse((np.nanmean(x), np.nanmean(y)), w, h, angle=ang,
                          edgecolor=clr, facecolor=clr, alpha=0.2, ls='--', lw=1.5)
            axm.add_patch(ell)
    axm.set_xlim(-30, 30); axm.set_ylim(-30, 30); axm.set_aspect('equal', 'box')
    axm.set_xlabel('Horizontal Break'); axm.set_ylabel('Induced Vertical Break')
    axm.legend(title='Pitch Type', fontsize=8, title_fontsize=9, loc='upper right')

    axt = fig.add_subplot(gs[1, 0]); axt.axis('off')
    tbl = axt.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Summary Metrics', fontweight='bold', y=0.87)

    if logo_img is not None:
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10); axl.imshow(logo_img); axl.axis('off')
    elif os.path.exists(LOGO_PATH):
        logo = mpimg.imread(LOGO_PATH)
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10); axl.imshow(logo); axl.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def pitcher_heatmaps(df_p: pd.DataFrame, handed: str = "Both"):
    if df_p.empty:
        return None
    side_series = df_p['BatterSide'].astype(str).str.strip().str.capitalize()
    if handed == "LHH":
        dfv = df_p[side_series.eq('Left')]
    elif handed == "RHH":
        dfv = df_p[side_series.eq('Right')]
    else:
        dfv = df_p.copy()

    top3 = list(dfv['AutoPitchType'].value_counts().index[:3])
    xi = np.linspace(X_MIN, X_MAX, HEATMAP_GRID)
    yi = np.linspace(Y_MIN, Y_MAX, HEATMAP_GRID)
    xi_m, yi_m = np.meshgrid(xi, yi)

    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)

    # Top row
    for i, pitch in enumerate(top3):
        ax = fig.add_subplot(gs[0, i])
        sub = dfv[dfv['AutoPitchType'] == pitch]
        x = pd.to_numeric(sub['PlateLocSide'], errors='coerce').to_numpy()
        y = pd.to_numeric(sub['PlateLocHeight'], errors='coerce').to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 12:
            ax.scatter(x, y, s=30, alpha=0.8, color=get_pitch_color(pitch), edgecolors='black', linewidths=0.3)
            ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
        else:
            zi = compute_density(x, y, xi_m, yi_m)
            ax.imshow(zi, origin='lower', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], aspect='equal', cmap=custom_cmap)
        draw_strikezone(ax)
        ax.set_title(f"{pitch} (n={len(sub)})", fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom row: Whiffs, Strikeouts, Damage
    # Whiffs
    ax_wh = fig.add_subplot(gs[1, 0])
    wh = dfv[dfv['PitchCall'] == 'StrikeSwinging']
    x = pd.to_numeric(wh['PlateLocSide'], errors='coerce').to_numpy()
    y = pd.to_numeric(wh['PlateLocHeight'], errors='coerce').to_numpy()
    mask = np.isfinite(x) & np.isfinite(y); x, y = x[mask], y[mask]
    if len(x) < 12: ax_wh.scatter(x,y,s=30,alpha=0.8,color='deepskyblue',edgecolors='black',linewidths=0.3); ax_wh.set_xlim(X_MIN,X_MAX); ax_wh.set_ylim(Y_MIN,Y_MAX)
    else:           ax_wh.imshow(compute_density(x,y,xi_m,yi_m),origin='lower',extent=[X_MIN,X_MAX,Y_MIN,Y_MAX],aspect='equal',cmap=custom_cmap)
    draw_strikezone(ax_wh); ax_wh.set_title(f"Whiffs (n={len(wh)})", fontweight='bold'); ax_wh.set_xticks([]); ax_wh.set_yticks([])

    # Strikeouts
    ax_ks = fig.add_subplot(gs[1, 1])
    ks = dfv[dfv['KorBB'].astype(str).str.contains('Strikeout', case=False, na=False)]
    x = pd.to_numeric(ks['PlateLocSide'], errors='coerce').to_numpy()
    y = pd.to_numeric(ks['PlateLocHeight'], errors='coerce').to_numpy()
    mask = np.isfinite(x) & np.isfinite(y); x, y = x[mask], y[mask]
    if len(x) < 12: ax_ks.scatter(x,y,s=30,alpha=0.8,color='orange',edgecolors='black',linewidths=0.3); ax_ks.set_xlim(X_MIN,X_MAX); ax_ks.set_ylim(Y_MIN,Y_MAX)
    else:           ax_ks.imshow(compute_density(x,y,xi_m,yi_m),origin='lower',extent=[X_MIN,X_MAX,Y_MIN,Y_MAX],aspect='equal',cmap=custom_cmap)
    draw_strikezone(ax_ks); ax_ks.set_title(f"Strikeouts (n={len(ks)})", fontweight='bold'); ax_ks.set_xticks([]); ax_ks.set_yticks([])

    # Damage ≥95
    ax_dm = fig.add_subplot(gs[1, 2])
    dmg = dfv[pd.to_numeric(dfv['ExitSpeed'], errors='coerce') >= 95]
    x = pd.to_numeric(dmg['PlateLocSide'], errors='coerce').to_numpy()
    y = pd.to_numeric(dmg['PlateLocHeight'], errors='coerce').to_numpy()
    mask = np.isfinite(x) & np.isfinite(y); x, y = x[mask], y[mask]
    if len(x) < 12: ax_dm.scatter(x,y,s=30,alpha=0.8,color='crimson',edgecolors='black',linewidths=0.3); ax_dm.set_xlim(X_MIN,X_MAX); ax_dm.set_ylim(Y_MIN,Y_MAX)
    else:           ax_dm.imshow(compute_density(x,y,xi_m,yi_m),origin='lower',extent=[X_MIN,X_MAX,Y_MIN,Y_MAX],aspect='equal',cmap=custom_cmap)
    draw_strikezone(ax_dm); ax_dm.set_title(f"Damage (n={len(dmg)})", fontweight='bold'); ax_dm.set_xticks([]); ax_dm.set_yticks([])

    # Strike % by count
    def strike_rate(sub_df):
        if sub_df.empty: return np.nan
        strike_calls = ['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']
        return sub_df['PitchCall'].isin(strike_calls).mean() * 100
    fp = strike_rate(dfv[(dfv['Balls'] == 0) & (dfv['Strikes'] == 0)])
    mix= strike_rate(dfv[((dfv['Balls'] == 1) & (dfv['Strikes'] == 0)) | ((dfv['Balls'] == 0) & (dfv['Strikes'] == 1)) | ((dfv['Balls'] == 1) & (dfv['Strikes'] == 1))])
    hp = strike_rate(dfv[((dfv['Balls'] == 2) & (dfv['Strikes'] == 0)) | ((dfv['Balls'] == 2) & (dfv['Strikes'] == 1)) | ((dfv['Balls'] == 3) & (dfv['Strikes'] == 1))])
    two= strike_rate(dfv[(dfv['Strikes'] == 2) & (dfv['Balls'] < 3)])

    ax_tbl = fig.add_axes([0.10, 0.02, 0.80, 0.12]); ax_tbl.axis('off')
    data = pd.DataFrame({'1st Pitch %':[np.round(fp,1)], 'Mix Count %':[np.round(mix,1)], 'Hitter+ %':[np.round(hp,1)], '2-Strike %':[np.round(two,1)]})
    tbl = ax_tbl.table(cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.4,1.3)
    ax_tbl.set_title('Strike Percentage by Count', y=0.8, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0.12, 0.97, 0.97])
    return fig

def release_points_figure(df_p: pd.DataFrame, pitcher_name: str, include_types=None):
    if include_types is None: include_types = []
    if df_p.empty: return None
    plot_df = df_p[['AutoPitchType']].copy()
    for cand in ['Relside','RelSide','ReleaseSide','Release_Side']:
        if cand in df_p.columns: plot_df['Relside'] = pd.to_numeric(df_p[cand], errors='coerce'); break
    for cand in ['Relheight','RelHeight','ReleaseHeight','Release_Height']:
        if cand in df_p.columns: plot_df['RelHeight'] = pd.to_numeric(df_p[cand], errors='coerce'); break
    plot_df['RelSpeed'] = pd.to_numeric(df_p['RelSpeed'], errors='coerce') if 'RelSpeed' in df_p.columns else np.nan
    plot_df['_type_canon'] = plot_df['AutoPitchType'].apply(canonicalize_type)
    plot_df = plot_df[(plot_df['_type_canon']!="Unknown") & np.isfinite(plot_df['Relside']) & np.isfinite(plot_df['RelHeight'])]
    if include_types: plot_df = plot_df[plot_df['_type_canon'].isin(include_types)]
    if plot_df.empty: return None

    plot_df['_color'] = plot_df['_type_canon'].apply(color_for_canon)
    means = plot_df.groupby('_type_canon', as_index=False).agg(mean_x=('Relside','mean'), mean_y=('RelHeight','mean'), mean_speed=('RelSpeed','mean'))
    if 'mean_speed' in means.columns: means = means.sort_values('mean_speed', ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.0), sharey=True)
    ax1.scatter(plot_df['Relside'], plot_df['RelHeight'], s=12, alpha=0.75, c=plot_df['_color'], edgecolors='none')
    ax1.set_xlim(-5,5); ax1.set_ylim(0,8); ax1.set_aspect("equal"); ax1.axhline(0, color="black", lw=1); ax1.axvline(0, color="black", lw=1)
    ax1.set_xlabel("Relside"); ax1.set_ylabel("RelHeight"); ax1.set_title(f"All Releases (n={len(plot_df)})", fontweight="bold")

    def draw_arm(ax, start_xy, end_xy, ring_color):
        x0,y0 = start_xy; x1,y1 = end_xy; dx,dy = x1-x0, y1-y0; L = float(np.hypot(dx,dy)); 
        if L<=1e-6: return
        ux,uy = dx/L, dy/L; px,py = -uy, ux
        ARM_BASE_HALF_WIDTH = 0.24; ARM_TIP_HALF_WIDTH = 0.08; SHOULDER_RADIUS_OUT = 0.20; HAND_RING_OUTER_R = 0.26; HAND_RING_INNER_R = 0.15
        ARM_FILL_COLOR="#111111"; SHOULDER_COLOR="#0d0d0d"
        sLx,sLy = x0+px*ARM_BASE_HALF_WIDTH, y0+py*ARM_BASE_HALF_WIDTH
        sRx,sRy = x0-px*ARM_BASE_HALF_WIDTH, y0-py*ARM_BASE_HALF_WIDTH
        eLx,eLy = x1+px*ARM_TIP_HALF_WIDTH,  y1+py*ARM_TIP_HALF_WIDTH
        eRx,eRy = x1-px*ARM_TIP_HALF_WIDTH,  y1-py*ARM_TIP_HALF_WIDTH
        ax.add_patch(Polygon([(sLx,sLy),(eLx,eLy),(eRx,eRy),(sRx,sRy)],closed=True,facecolor=ARM_FILL_COLOR,edgecolor=ARM_FILL_COLOR,zorder=1))
        ax.add_patch(Circle((x0,y0),radius=SHOULDER_RADIUS_OUT,facecolor=SHOULDER_COLOR,edgecolor=SHOULDER_COLOR,zorder=2))
        ax.add_patch(Circle((x1,y1),radius=HAND_RING_OUTER_R,facecolor=ring_color,edgecolor=ring_color,zorder=4))
        inner_face=ax.get_facecolor()
        ax.add_patch(Circle((x1,y1),radius=HAND_RING_INNER_R,facecolor=inner_face,edgecolor=inner_face,zorder=5))

    for _, r in means.iterrows():
        draw_arm(ax2, (0.0,0.0), (float(r['mean_x']), float(r['mean_y'])), color_for_canon(r['_type_canon']))
    ax2.set_xlim(-5,5); ax2.set_ylim(0,8); ax2.set_aspect("equal"); ax2.axhline(0, color="black", lw=1); ax2.axvline(0, color="black", lw=1)
    ax2.set_xlabel("Relside"); ax2.set_title("Average Releases", fontweight="bold")

    handles = [Line2D([0],[0], marker='o', linestyle='none', markersize=6,
                      label=(f"{r['_type_canon']} ({r['mean_speed']:.1f})" if pd.notna(r['mean_speed']) else r['_type_canon']),
                      color=color_for_canon(r['_type_canon'])) for _, r in means.iterrows()]
    if handles: ax2.legend(handles=handles, title="Pitch Type", loc="upper right")
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# HITTER REPORT (Nebraska)
# ─────────────────────────────────────────────────────────────────────────────
shape_map = {'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}
color_map = {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan','InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}

def create_hitter_report(df,batter,ncols=3):
    bdf=df[df['Batter']==batter]
    if bdf.empty: return None
    pa=list(bdf.groupby(['GameID','Inning','Top/Bottom','PAofInning']))
    n_pa=len(pa); nrows=math.ceil(n_pa/ncols); descs=[]
    for _,padf in pa:
        lines=[]
        for _,p in padf.iterrows():
            velo = p.EffectiveVelo if 'EffectiveVelo' in p and pd.notna(p.EffectiveVelo) else p.RelSpeed if 'RelSpeed' in p and pd.notna(p.RelSpeed) else np.nan
            velo_s = f"{float(velo):.1f}" if pd.notna(velo) else "—"
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType} {velo_s} MPH / {p.PitchCall}")
        ip=padf[padf['PitchCall']=='InPlay']
        if not ip.empty:
            last=ip.iloc[-1]; res=last.PlayResult or 'InPlay'
            if not pd.isna(last.ExitSpeed): res+=f" ({last.ExitSpeed:.1f} MPH)"
            lines.append(f"▶ PA Result: {res}")
        else:
            balls=(padf['PitchCall']=='BallCalled').sum()
            strikes=padf['PitchCall'].isin(['StrikeCalled','StrikeSwinging']).sum()
            if balls>=4: lines.append('▶ PA Result: Walk')
            elif strikes>=3: lines.append('▶ PA Result: Strikeout')
        descs.append(lines)
    fig=plt.figure(figsize=(3+4*ncols+1,4*nrows))
    gs=GridSpec(nrows,ncols+1,width_ratios=[0.8]+[1]*ncols,wspace=0.1)
    logo=mpimg.imread(LOGO_PATH) if os.path.exists(LOGO_PATH) else None
    if logo is not None:
        axl=fig.add_axes([0.88,0.88,0.12,0.12],anchor='NE'); axl.imshow(logo); axl.axis('off')
    date=pa[0][1]['Date'].iloc[0]
    fig.suptitle(f"{batter} Hitter Report for {pd.Timestamp(date).strftime('%B %d, %Y')}",fontsize=16,x=0.55,y=1.0,fontweight='bold')
    gd=pd.concat([grp for _,grp in pa])
    whiffs=(gd['PitchCall']=='StrikeSwinging').sum()
    hardhits=(pd.to_numeric(gd['ExitSpeed'], errors='coerce')>95).sum()
    chases=gd[(gd['PitchCall']=='StrikeSwinging')&((gd['PlateLocSide']<SZ_LEFT)|(gd['PlateLocSide']>SZ_RIGHT)|(gd['PlateLocHeight']<SZ_BOTTOM)|(gd['PlateLocHeight']>SZ_TOP))].shape[0]
    fig.text(0.55,0.96,f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",ha='center',va='top',fontsize=12)
    for idx,((_,inn,tb,_),padf) in enumerate(pa):
        row,col=divmod(idx,ncols)
        ax=fig.add_subplot(gs[row,col+1]); draw_strikezone(ax)
        hand='LHP' if str(padf['PitcherThrows'].iloc[0]).startswith('L') else 'RHP'
        pitchr=padf['Pitcher'].iloc[0]
        for _,p in padf.iterrows():
            mk=shape_map.get(p.AutoPitchType,'o'); clr=color_map.get(p.PitchCall,'black')
            sz=200 if p.AutoPitchType=='Slider' else 150
            ax.scatter(p.PlateLocSide,p.PlateLocHeight,marker=mk,c=clr,s=sz,edgecolor='white',linewidth=1,zorder=2)
            yoff=-0.05 if p.AutoPitchType=='Slider' else 0
            ax.text(p.PlateLocSide,p.PlateLocHeight+yoff,str(int(p.PitchofPA)),ha='center',va='center',fontsize=6,fontweight='bold',zorder=3)
        ax.set_xlim(-3,3);ax.set_ylim(0,5);ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}",fontsize=10,fontweight='bold')
        ax.text(0.5,0.1,f"vs {pitchr} ({hand})",transform=ax.transAxes,ha='center',va='top',fontsize=9,style='italic')
    axd=fig.add_subplot(gs[:,0]); axd.axis('off'); y0=1.0; dy=1.0/(n_pa*5.0)
    for i,lines in enumerate(descs,1):
        axd.hlines(y0-dy*0.1,0,1,transform=axd.transAxes,color='black',linewidth=1)
        axd.text(0.02,y0,f"PA {i}",fontsize=6,fontweight='bold',transform=axd.transAxes)
        yln=y0-dy
        for ln in lines: axd.text(0.02,yln,ln,fontsize=6,transform=axd.transAxes); yln-=dy
        y0=yln-dy*0.05
    res_handles=[Line2D([0],[0],marker='o',color='w',label=k,markerfacecolor=v,markersize=10,markeredgecolor='k') for k,v in color_map.items()]
    fig.legend(res_handles,[h.get_label() for h in res_handles],title='Result',loc='lower right',bbox_to_anchor=(0.90,0.02))
    pitch_handles=[Line2D([0],[0],marker=m,color='w',label=k,markerfacecolor='gray',markersize=10,markeredgecolor='k') for k,m in shape_map.items()]
    fig.legend(pitch_handles,[h.get_label() for h in pitch_handles],title='Pitches',loc='lower right',bbox_to_anchor=(0.98,0.02))
    plt.tight_layout(rect=[0.12,0.05,1,0.88])
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# D1 STATS (Hitters & Pitchers)
# ─────────────────────────────────────────────────────────────────────────────
def compute_hitter_rates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    pr = d['PlayResult'].astype(str).str.lower()
    d['is_ab']  = pr.isin({'single','double','triple','homerun','out','error','fielderschoice'}).astype(int)
    d['is_hit'] = pr.isin({'single','double','triple','homerun'}).astype(int)
    d['is_bb']  = d['KorBB'].astype(str).str.lower().eq('walk').astype(int)
    d['is_k']   = d['KorBB'].astype(str).str.contains('strikeout', case=False, na=False).astype(int)
    d['is_hbp'] = d['PitchCall'].astype(str).eq('HitByPitch').astype(int)
    d['is_sf']  = d['PlayResult'].astype(str).str.contains('Sacrifice', case=False, na=False).astype(int)
    d['is_1b']  = pr.eq('single').astype(int)
    d['is_2b']  = pr.eq('double').astype(int)
    d['is_3b']  = pr.eq('triple').astype(int)
    d['is_hr']  = pr.eq('homerun').astype(int)
    d['is_pa']  = (d['is_ab'] | d['is_bb'] | d['is_hbp'] | d['is_sf']).astype(int)
    g = (d.groupby(['BatterTeam','Batter'], as_index=False)
           .agg(PA=('is_pa','sum'), AB=('is_ab','sum'), Hits=('is_hit','sum'),
                Doubles=('is_2b','sum'), Triples=('is_3b','sum'), Homeruns=('is_hr','sum'),
                HBP=('is_hbp','sum'), BB=('is_bb','sum'), K=('is_k','sum'), Singles=('is_1b','sum'), SF=('is_sf','sum')))
    g['TB'] = g['Singles'] + 2*g['Doubles'] + 3*g['Triples'] + 4*g['Homeruns']
    g = g.rename(columns={'Doubles':'2B','Triples':'3B','Homeruns':'HR'})
    with np.errstate(divide='ignore', invalid='ignore'):
        ba = np.where(g['AB']>0, g['Hits']/g['AB'], 0.0)
        slg= np.where(g['AB']>0, g['TB']/g['AB'], 0.0)
        den = g['AB'] + g['BB'] + g['HBP'] + g['SF']
        obp = np.divide(g['Hits'] + g['BB'] + g['HBP'], den, out=np.zeros_like(den,dtype=float), where=den>0)
        ops = obp + slg
    g['BA']  = [f"{x:.3f}" for x in ba]
    g['OBP'] = [f"{x:.3f}" for x in obp]
    g['SLG'] = [f"{x:.3f}" for x in slg]
    g['OPS'] = [f"{x:.3f}" for x in ops]
    g['Team'] = g['BatterTeam'].map(TEAM_MAP).fillna(g['BatterTeam'])
    g['Batter'] = g['Batter'].apply(last_first_to_first_last)
    out = g[['Team','Batter','PA','AB','Hits','2B','3B','HR','HBP','BB','K','BA','OBP','SLG','OPS']].copy()
    return out.sort_values('OPS', ascending=False)

def compute_pitcher_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    pr_low  = d['PlayResult'].astype(str).str.lower()
    korbb_s = d['KorBB'].astype(str)
    pitch_s = d['PitchCall'].astype(str)
    hit_values = {'single','double','triple','homerun'}
    d['is_hit']  = pr_low.isin(hit_values).astype(int)
    d['is_hr']   = pr_low.eq('homerun').astype(int)
    d['is_bb']   = korbb_s.str.lower().eq('walk').astype(int)
    d['is_so']   = korbb_s.str.contains('strikeout', case=False, na=False).astype(int)
    d['is_hbp']  = pitch_s.eq('HitByPitch').astype(int)
    if 'OutsOnPlay' in d.columns:
        outs_on_play = pd.to_numeric(d['OutsOnPlay'], errors='coerce').fillna(0).astype(int)
        outs_from_k  = d['is_so'] * (outs_on_play == 0)  # avoid double counting
        d['outs']    = outs_on_play + outs_from_k
    else:
        is_out = pr_low.eq('out').astype(int)
        d['outs'] = is_out + d['is_so']
    g = (d.groupby(['PitcherTeam','Pitcher'], as_index=False)
           .agg(Hits=('is_hit','sum'), HR=('is_hr','sum'), BB=('is_bb','sum'),
                HBP=('is_hbp','sum'), SO=('is_so','sum'), Outs=('outs','sum')))
    g['IP_num'] = g['Outs'] / 3.0
    def ip_display(ip_num: float) -> str:
        outs = int(round(ip_num * 3))
        return f"{outs//3}.{outs%3}"
    g['IP'] = g['IP_num'].apply(ip_display)
    ip = g['IP_num'].replace(0, np.nan)
    g['WHIP_num'] = ((g['BB'] + g['Hits']) / ip).fillna(0.0)
    g['BB/9_num'] = (g['BB']  / ip * 9).fillna(0.0)
    g['H/9_num']  = (g['Hits']/ ip * 9).fillna(0.0)
    g['HR/9_num'] = (g['HR']  / ip * 9).fillna(0.0)
    g['SO/9_num'] = (g['SO']  / ip * 9).fillna(0.0)
    g['WHIP'] = g['WHIP_num'].map(lambda x: f"{x:.3f}")
    g['BB/9'] = g['BB/9_num'].map(lambda x: f"{x:.2f}")
    g['H/9']  = g['H/9_num'].map(lambda x: f"{x:.2f}")
    g['HR/9'] = g['HR/9_num'].map(lambda x: f"{x:.2f}")
    g['SO/9'] = g['SO/9_num'].map(lambda x: f"{x:.2f}")
    g = g.rename(columns={'PitcherTeam':'Team', 'Pitcher':'Name'})
    g['Team'] = g['Team'].map(TEAM_MAP).fillna(g['Team'])
    g['Name'] = g['Name'].apply(last_first_to_first_last)
    out = g[['Team','Name','IP','Hits','HR','BB','HBP','SO','WHIP','BB/9','H/9','HR/9','SO/9']].copy()
    out = out.join(g[['IP_num','WHIP_num','BB/9_num','H/9_num','HR/9_num','SO/9_num']])
    return out.sort_values('WHIP_num', ascending=True)

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
section = st.radio("Choose Section", ["Nebraska Baseball", "D1 Baseball"], horizontal=True)
st.title(section)

# Sidebar = collapsible filter drawer
with st.sidebar:
    st.header("Filters")

if section == "Nebraska Baseball":
    st.subheader("Pitcher or Hitter Report")

    tab_pitch, tab_hit = st.tabs(["Pitcher Report", "Hitter Report"])

    # Nebraska filters (common)
    neb_df = df_all.copy()
    # Pitcher tab
    with tab_pitch:
        st.markdown("**Pitcher Report**")

        tab_std, tab_cmp = st.tabs(["Standard", "Compare"])

        # Restrict to NEB pitchers
        neb_pitch = neb_df[neb_df.get('PitcherTeam','') == 'NEB'].copy().sort_values('Date')
        if neb_pitch.empty:
            st.warning("No Nebraska pitcher rows found.")
        else:
            pitchers = sorted(neb_pitch['Pitcher'].dropna().unique().tolist())
            with st.sidebar:
                player = st.selectbox("Pitcher", pitchers, key="neb_pitcher_select")
                first_last = last_first_to_first_last(player)

            # Pitcher context
            p_df_all = neb_pitch[neb_pitch['Pitcher'] == player].copy()
            season_apps = p_df_all['DateOnly'].dropna().nunique()
            p_dates_all = sorted(p_df_all['DateOnly'].dropna().unique().tolist())
            p_months_all = sorted(pd.to_datetime(p_dates_all).month)
            MONTH_MAP = month_name_map()

            # STANDARD
            with tab_std:
                with st.sidebar:
                    st.markdown("**Standard Filters**")
                    month_opts = sorted(set(p_months_all))
                    sel_months_names = st.multiselect("Months", [MONTH_MAP[m] for m in month_opts], key="std_months")
                    sel_months = [m for m in month_opts if MONTH_MAP[m] in sel_months_names]
                    if sel_months:
                        avail_dates = [d for d in p_dates_all if pd.Timestamp(d).month in sel_months]
                    else:
                        avail_dates = p_dates_all
                    sel_dates = st.multiselect("Dates", avail_dates,
                                               format_func=lambda d: pd.Timestamp(d).strftime("%B %d, %Y"),
                                               key="std_dates")

                    handed = st.radio("Split (Heatmaps)", ["Both", "LHH", "RHH"], horizontal=True, key="std_handed")

                    # Release points pitch filter
                    canon_types_present = sorted(
                        pd.Series(p_df_all['AutoPitchType'].dropna().astype(str).map(canonicalize_type))
                        .replace("Unknown", np.nan).dropna().unique()
                    )
                    sel_types = st.multiselect("Release Points: Pitch Types",
                                               canon_types_present,
                                               default=canon_types_present,
                                               key="std_rel_types")

                # Build filtered df (union months + dates)
                p_df = p_df_all.copy()
                mask = pd.Series(False, index=p_df.index)
                if sel_months:
                    mask = mask | p_df['Date'].dt.month.isin(sel_months)
                if sel_dates:
                    mask = mask | p_df['DateOnly'].isin(set(sel_dates))
                if sel_months or sel_dates:
                    p_df = p_df[mask]

                # Title block
                label_dates = sorted(p_df['DateOnly'].dropna().unique().tolist())
                season_lab = summarize_dates_range(label_dates)
                st.markdown(f"### {first_last} Metrics")
                st.caption(f"({season_lab if season_lab else 'Season'})")
                st.markdown(f"**{first_last} ({season_apps} Appearances)**")

                # Full metrics
                fig_full = combined_pitcher_report(p_df, player, logo_img=None, coverage=0.8)
                if fig_full: st.pyplot(fig_full)

                # Heatmaps
                st.markdown(f"### {first_last} Heatmaps")
                st.caption(f"({season_lab if season_lab else 'Season'}) ({handed})")
                fig_hm = pitcher_heatmaps(p_df, handed=handed)
                if fig_hm: st.pyplot(fig_hm)

                # Release points
                st.markdown("### Release Points")
                rel_fig = release_points_figure(p_df, player, include_types=sel_types if sel_types else [])
                if rel_fig: st.pyplot(rel_fig)

            # COMPARE
            with tab_cmp:
                st.markdown("**Compare Appearances**")
                with st.sidebar:
                    st.markdown("**Compare Filters**")
                    month_opts = sorted(set(pd.to_datetime(p_dates_all).month))
                    sel_months_A_names = st.multiselect("Months (A)", [MONTH_MAP[m] for m in month_opts], key="cmp_mA")
                    sel_months_A = [m for m in month_opts if MONTH_MAP[m] in sel_months_A_names]
                    if sel_months_A:
                        avail_A = [d for d in p_dates_all if pd.Timestamp(d).month in sel_months_A]
                    else:
                        avail_A = p_dates_all
                    sel_dates_A = st.multiselect("Dates (A)", avail_A,
                                                 format_func=lambda d: pd.Timestamp(d).strftime("%B %d, %Y"),
                                                 key="cmp_dA")

                    sel_months_B_names = st.multiselect("Months (B)", [MONTH_MAP[m] for m in month_opts], key="cmp_mB")
                    sel_months_B = [m for m in month_opts if MONTH_MAP[m] in sel_months_B_names]
                    if sel_months_B:
                        avail_B = [d for d in p_dates_all if pd.Timestamp(d).month in sel_months_B]
                    else:
                        avail_B = p_dates_all
                    sel_dates_B = st.multiselect("Dates (B)", avail_B,
                                                 format_func=lambda d: pd.Timestamp(d).strftime("%B %d, %Y"),
                                                 key="cmp_dB")

                def apply_union(df_base, months, dates):
                    m = pd.Series(False, index=df_base.index)
                    if months: m = m | df_base['Date'].dt.month.isin(months)
                    if dates:  m = m | df_base['DateOnly'].isin(set(dates))
                    return df_base[m] if m.any() else df_base.iloc[0:0]

                pA = apply_union(p_df_all, sel_months_A, sel_dates_A)
                pB = apply_union(p_df_all, sel_months_B, sel_dates_B)
                labA = summarize_dates_range(sorted(pA['DateOnly'].dropna().unique().tolist()))
                labB = summarize_dates_range(sorted(pB['DateOnly'].dropna().unique().tolist()))

                colA, colB = st.columns(2)
                with colA:
                    st.markdown(f"#### {first_last} — {labA if labA else 'Season'}")
                    figA = combined_pitcher_report(pA, player);  st.pyplot(figA) if figA else None
                    figAh = pitcher_heatmaps(pA, handed="Both"); st.pyplot(figAh) if figAh else None
                with colB:
                    st.markdown(f"#### {first_last} — {labB if labB else 'Season'}")
                    figB = combined_pitcher_report(pB, player);  st.pyplot(figB) if figB else None
                    figBh = pitcher_heatmaps(pB, handed="Both"); st.pyplot(figBh) if figBh else None

    # HITTER tab
    with tab_hit:
        st.markdown("**Hitter Report**")
        neb_bat = neb_df[neb_df.get('BatterTeam','') == 'NEB'].copy()
        if neb_bat.empty:
            st.info("No Nebraska hitter rows found.")
        else:
            with st.sidebar:
                batter = st.selectbox("Batter", sorted(neb_bat['Batter'].dropna().unique().tolist()), key="neb_batter")
            fig = create_hitter_report(neb_bat, batter, ncols=3)
            if fig: st.pyplot(fig)

# D1 Baseball
else:
    tab_h, tab_p = st.tabs(["Hitter Statistics", "Pitcher Statistics"])
    # HITTER STATS
    with tab_h:
        st.markdown("**Hitter Statistics**")
        d1 = df_all.copy()
        with st.sidebar:
            conf = st.selectbox("Conference", ["All Conferences"] + list(POWER4.keys()), key="d1_conf_h")
            if conf != "All Conferences":
                possible = POWER4.get(conf, set())
                present_codes = set(d1['BatterTeam'].dropna().unique().tolist())
                team_codes = sorted(list(present_codes & possible))
            else:
                team_codes = sorted(d1['BatterTeam'].dropna().unique().tolist())
            pretty_team = {k: TEAM_MAP.get(k, k) for k in team_codes}
            team_choice = st.selectbox("Team", ["All Teams"] + [pretty_team[t] for t in team_codes], key="d1_team_h")
            months_all = sorted(d1['Date'].dt.month.dropna().unique().tolist()) if 'Date' in d1.columns else []
            MONTH_MAP = month_name_map()
            sel_months_names = st.multiselect("Month(s)", [MONTH_MAP[m] for m in months_all], key="d1_mo_h")
            sel_months = [m for m in months_all if MONTH_MAP[m] in sel_months_names]
            if sel_months:
                avail_dates = sorted(d1[d1['Date'].dt.month.isin(sel_months)]['DateOnly'].dropna().unique().tolist())
            else:
                avail_dates = sorted(d1['DateOnly'].dropna().unique().tolist()) if 'DateOnly' in d1.columns else []
            sel_dates = st.multiselect("Day(s)", avail_dates, format_func=lambda d: pd.Timestamp(d).strftime("%B %d, %Y"), key="d1_days_h")

        if team_choice != "All Teams":
            team_key = [k for k,v in pretty_team.items() if v == team_choice][0]
            d1 = d1[d1['BatterTeam'] == team_key]

        if 'Date' in d1.columns:
            mask = pd.Series(False, index=d1.index)
            if sel_months: mask = mask | d1['Date'].dt.month.isin(sel_months)
            if sel_dates:  mask = mask | d1['DateOnly'].isin(set(sel_dates))
            if sel_months or sel_dates: d1 = d1[mask]

        if d1.empty:
            st.info("No rows match your filters.")
        else:
            stats = compute_hitter_rates(d1)
            st.dataframe(stats, use_container_width=True)

    # PITCHER STATS
    with tab_p:
        st.markdown("**Pitcher Statistics**")
        d1p = df_all.copy()
        with st.sidebar:
            conf = st.selectbox("Conference", ["All Conferences"] + list(POWER4.keys()), key="d1_conf_p")
            if conf != "All Conferences":
                possible = POWER4.get(conf, set())
                present_codes = set(d1p['PitcherTeam'].dropna().unique().tolist())
                team_codes = sorted(list(present_codes & possible))
            else:
                team_codes = sorted(d1p['PitcherTeam'].dropna().unique().tolist())
            pretty_team = {k: TEAM_MAP.get(k, k) for k in team_codes}
            team_choice = st.selectbox("Team", ["All Teams"] + [pretty_team[t] for t in team_codes], key="d1_team_p")
            months_all = sorted(d1p['Date'].dt.month.dropna().unique().tolist()) if 'Date' in d1p.columns else []
            MONTH_MAP = month_name_map()
            sel_months_names = st.multiselect("Month(s)", [MONTH_MAP[m] for m in months_all], key="d1_mo_p")
            sel_months = [m for m in months_all if MONTH_MAP[m] in sel_months_names]
            if sel_months:
                avail_dates = sorted(d1p[d1p['Date'].dt.month.isin(sel_months)]['DateOnly'].dropna().unique().tolist())
            else:
                avail_dates = sorted(d1p['DateOnly'].dropna().unique().tolist()) if 'DateOnly' in d1p.columns else []
            sel_dates = st.multiselect("Day(s)", avail_dates, format_func=lambda d: pd.Timestamp(d).strftime("%B %d, %Y"), key="d1_days_p")

        if team_choice != "All Teams":
            team_key = [k for k,v in pretty_team.items() if v == team_choice][0]
            d1p = d1p[d1p['PitcherTeam'] == team_key]

        if 'Date' in d1p.columns:
            mask = pd.Series(False, index=d1p.index)
            if sel_months: mask = mask | d1p['Date'].dt.month.isin(sel_months)
            if sel_dates:  mask = mask | d1p['DateOnly'].isin(set(sel_dates))
            if sel_months or sel_dates: d1p = d1p[mask]

        if d1p.empty:
            st.info("No rows match your filters.")
        else:
            pstats = compute_pitcher_table(d1p)
            st.dataframe(pstats[['Team','Name','IP','Hits','HR','BB','HBP','SO','WHIP','BB/9','H/9','HR/9','SO/9']], use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Run: streamlit run streamlit_app.py
# Data path: update CSV_PATH to point at your uploaded file if needed
