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
from scipy.stats import chi2, gaussian_kde
from numpy.linalg import LinAlgError
from matplotlib import colors

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Baseball Analytics",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "B10C25_streamlit_streamlit_columns.csv"
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

def summarize_dates_range(series: pd.Series) -> str:
    if series is None or series.empty:
        return ""
    uniq = pd.to_datetime(series.dropna()).dt.date.unique()
    if len(uniq) == 0:
        return ""
    if len(uniq) == 1:
        return format_date_long(uniq[0])
    dmin, dmax = min(uniq), max(uniq)
    return f"{format_date_long(dmin)} – {format_date_long(dmax)}"

def filter_by_month_day(df, date_col="Date", months=None, days=None):
    """
    Filter df by selected months and/or days.
    - months only  -> rows whose month ∈ months  (combine all days in those months)
    - days only    -> rows whose day   ∈ days    (across all months)
    - both         -> month ∈ months AND day ∈ days
    - neither      -> unchanged
    """
    if date_col not in df.columns or df.empty:
        return df
    s = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if months:
        mask &= s.dt.month.isin(months)
    if days:
        mask &= s.dt.day.isin(days)
    return df[mask]

# ──────────────────────────────────────────────────────────────────────────────
# STRIKE ZONE & COLORS
# ──────────────────────────────────────────────────────────────────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
    N=256,
)

def get_zone_bounds():
    # Fixed zone so size stays identical across heatmap/scatter
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
# DENSITY HELPERS
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

def strike_rate(df):
    if len(df) == 0: return np.nan
    strike_calls = ['StrikeCalled','StrikeSwinging','FoulBallNotFieldable','FoulBallFieldable','InPlay']
    return df['PitchCall'].isin(strike_calls).mean() * 100

# ──────────────────────────────────────────────────────────────────────────────
# PITCHER: STANDARD REPORT (Movement + Summary)
# ──────────────────────────────────────────────────────────────────────────────
def combined_pitcher_report(df, pitcher_name, logo_img, coverage=0.8):
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

    date_str = summarize_dates_range(df_p.get("Date", pd.Series(dtype="datetime64[ns]")))
    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.7], hspace=0.3)

    # Movement
    axm = fig.add_subplot(gs[0, 0]); axm.set_title('Movement Plot')
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

    # Summary (single table)
    axt = fig.add_subplot(gs[1, 0]); axt.axis('off')
    tbl = axt.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Summary Metrics', fontweight='bold', y=0.87)

    # Logo
    if logo_img is not None:
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10); axl.imshow(logo_img); axl.axis('off')
    elif os.path.exists(LOGO_PATH):
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10); axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis('off')

    fig.suptitle(f"{pitcher_name} – Full Report{(' — ' + date_str) if date_str else ''}", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, summary

# ──────────────────────────────────────────────────────────────────────────────
# PITCHER HEATMAPS (auto-switch to scatter when n < 12; fixed strike zone)
# ──────────────────────────────────────────────────────────────────────────────
def combined_pitcher_heatmap_report(df, pitcher_name, grid_size=100):
    df_p = df[df['Pitcher'] == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' with the current filters.")
        return None

    x_min, x_max, y_min, y_max = get_view_bounds()
    xi = np.linspace(x_min, x_max, grid_size); yi = np.linspace(y_min, y_max, grid_size)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi); grid_coords = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()])
    z_left, z_bottom, z_w, z_h = get_zone_bounds()
    threshold = 12

    def panel(ax, sub, title, orange=False):
        n = len(sub)
        x = sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy()
        y = sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy()
        if n < threshold:
            ax.scatter(x, y, s=30, alpha=0.7, color=('orange' if orange else 'deepskyblue'), edgecolors='black')
        else:
            zi = compute_density(x, y, grid_coords, xi_mesh.shape)
            ax.imshow(zi, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap=custom_cmap)
        draw_strikezone(ax, z_left, z_bottom, z_w, z_h)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal','box')
        ax.set_title(title, fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])

    date_str = summarize_dates_range(df_p.get("Date", pd.Series(dtype="datetime64[ns]")))

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 0.6], hspace=0.4, wspace=0.3)

    # Top 4 pitch types
    top4 = list(df_p['AutoPitchType'].value_counts().index[:4])
    for i, pitch in enumerate(top4):
        ax = fig.add_subplot(gs[0, i]); sub = df_p[df_p['AutoPitchType'] == pitch]
        panel(ax, sub, f"{pitch} (n={len(sub)})")
    fig.add_subplot(gs[0, 4]).axis('off')

    # L/R, whiffs, K, damage
    sub_vl = df_p[df_p['BatterSide'] == 'Left'];  ax = fig.add_subplot(gs[1, 0]); panel(ax, sub_vl, f"vs Left-Handed (n={len(sub_vl)})")
    sub_vr = df_p[df_p['BatterSide'] == 'Right']; ax = fig.add_subplot(gs[1, 1]); panel(ax, sub_vr, f"vs Right-Handed (n={len(sub_vr)})")
    sub_wh = df_p[df_p['PitchCall'] == 'StrikeSwinging']; ax = fig.add_subplot(gs[1, 2]); panel(ax, sub_wh, f"Whiffs (n={len(sub_wh)})")
    sub_ks = df_p[df_p['KorBB'] == 'Strikeout'];          ax = fig.add_subplot(gs[1, 3]); panel(ax, sub_ks, f"Strikeouts (n={len(sub_ks)})")
    sub_dg = df_p[df_p['ExitSpeed'] >= 95];               ax = fig.add_subplot(gs[1, 4]); panel(ax, sub_dg, f"Damage (n={len(sub_dg)})", orange=True)

    axt = fig.add_subplot(gs[2, :]); axt.axis('off')
    fp  = strike_rate(df_p[(df_p['Balls']==0) & (df_p['Strikes']==0)])
    mix = strike_rate(df_p[((df_p['Balls']==1)&(df_p['Strikes']==0)) | ((df_p['Balls']==0)&(df_p['Strikes']==1)) | ((df_p['Balls']==1)&(df_p['Strikes']==1))])
    hp  = strike_rate(df_p[((df_p['Balls']==2)&(df_p['Strikes']==0)) | ((df_p['Balls']==2)&(df_p['Strikes']==1)) | ((df_p['Balls']==3)&(df_p['Strikes']==1))])
    two = strike_rate(df_p[(df_p['Strikes']==2) & (df_p['Balls']<3)])
    metrics = pd.DataFrame({'1st Pitch %':[fp],'Mix Count %':[mix],'Hitter+ %':[hp],'2-Strike %':[two]}).round(1)
    tbl = axt.table(cellText=metrics.values, colLabels=metrics.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Strike Percentage by Count', y=0.75, fontweight='bold')

    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88, 0.92, 0.10, 0.10], anchor='NE', zorder=10); axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); fig.suptitle(f"{pitcher_name} – Heatmap Report{(' — ' + date_str) if date_str else ''}", fontsize=18, y=0.98, fontweight='bold')
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# HITTER HEATMAPS & STANDARD REPORT
# ──────────────────────────────────────────────────────────────────────────────
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

    date_title = summarize_dates_range(bdf.get("Date", pd.Series(dtype="datetime64[ns]")))
    fig.suptitle(f"{batter} Hitter Report{(' — ' + date_title) if date_title else ''}",
                 fontsize=16, x=0.55, y=1.0, fontweight='bold')

    if pa:
        gd = pd.concat([grp for _, grp in pa])
        whiffs   = (gd['PitchCall']=='StrikeSwinging').sum()
        hardhits = (gd['ExitSpeed']>95).sum()
        chases   = gd[(gd['PitchCall']=='StrikeSwinging') &
                      ((gd['PlateLocSide']<-0.83)|(gd['PlateLocSide']>0.83)|
                       (gd['PlateLocHeight']<1.5)|(gd['PlateLocHeight']>3.5))].shape[0]
        fig.text(0.55,0.96,f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",
                 ha='center', va='top', fontsize=12)

    for idx, ((_, inn, tb, _), padf) in enumerate(pa):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col+1]); draw_strikezone(ax)
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

# ──────────────────────────────────────────────────────────────────────────────
# RELEASE POINTS (now with pitch-type filter)
# ──────────────────────────────────────────────────────────────────────────────
ARM_BASE_HALF_WIDTH = 0.24
ARM_TIP_HALF_WIDTH  = 0.08
SHOULDER_RADIUS_OUT = 0.20
HAND_RING_OUTER_R   = 0.26
HAND_RING_INNER_R   = 0.15
ARM_FILL_COLOR      = "#111111"
SHOULDER_COLOR      = "#0d0d0d"

def pick_col(df: pd.DataFrame, *cands) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c and c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def norm_text(x: str) -> str:
    return str(x).strip().lower()

def norm_type(x: str) -> str:
    s = norm_text(x)
    replacements = {"four seam": "four-seam", "4 seam":"4-seam", "two seam":"two-seam", "2 seam":"2-seam"}
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
        "fastball": "#E60026",
        "two-seam fastball": "#FF9300",
        "cutter": "#800080",
        "changeup": "#008000",
        "splitter": "#00CCCC",
        "curveball": "#0033CC",
        "knuckle curve": "#000000",
        "slider": "#CCCC00",
        "sweeper": "#B5651D",
        "screwball": "#CC0066",
        "eephus": "#666666",
    }
    return palette.get(key, "#7F7F7F")

def draw_stylized_arm(ax, start_xy, end_xy, ring_color):
    x0, y0 = start_xy; x1, y1 = end_xy
    dx, dy = x1 - x0, y1 - y0
    L = float(np.hypot(dx, dy))
    if L <= 1e-6: return
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    sLx, sLy = x0 + px*ARM_BASE_HALF_WIDTH, y0 + py*ARM_BASE_HALF_WIDTH
    sRx, sRy = x0 - px*ARM_BASE_HALF_WIDTH, y0 - py*ARM_BASE_HALF_WIDTH
    eLx, eLy = x1 + px*ARM_TIP_HALF_WIDTH,  y1 + py*ARM_TIP_HALF_WIDTH
    eRx, eRy = x1 - px*ARM_TIP_HALF_WIDTH,  y1 - py*ARM_TIP_HALF_WIDTH
    arm_poly = Polygon([(sLx, sLy), (eLx, eLy), (eRx, eRy), (sRx, sRy)],
                       closed=True, facecolor=ARM_FILL_COLOR, edgecolor=ARM_FILL_COLOR, zorder=1)
    ax.add_patch(arm_poly)
    ax.add_patch(Circle((x0, y0), radius=SHOULDER_RADIUS_OUT, facecolor=SHOULDER_COLOR, edgecolor=SHOULDER_COLOR, zorder=2))
    outer = Circle((x1, y1), radius=HAND_RING_OUTER_R, facecolor=ring_color, edgecolor=ring_color, zorder=4)
    ax.add_patch(outer)
    inner_face = ax.get_facecolor()
    inner = Circle((x1, y1), radius=HAND_RING_INNER_R, facecolor=inner_face, edgecolor=inner_face, zorder=5)
    ax.add_patch(inner)

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
    if speed_col: sub[speed_col] = pd.to_numeric(sub[speed_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col])

    sub["_type_canon"] = sub[type_col].apply(canonicalize_type)
    sub = sub[sub["_type_canon"] != "Unknown"].copy()

    # Apply pitch-type filter (canonical labels) if provided
    if include_types:
        sub = sub[sub["_type_canon"].isin(include_types)]
    if sub.empty:
        st.warning("No pitches after applying the selected pitch-type filter.")
        return None

    sub["_color"] = sub["_type_canon"].apply(color_for_release)

    agg = {"mean_x": (x_col, "mean"), "mean_y": (y_col, "mean")}
    if speed_col: agg["mean_speed"] = (speed_col, "mean")
    means = sub.groupby("_type_canon", as_index=False).agg(**agg)
    means["color"] = means["_type_canon"].apply(color_for_release)
    if "mean_speed" in means.columns:
        means = means.sort_values("mean_speed", ascending=False).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.0), sharey=True)
    ax1.scatter(sub[x_col], sub[y_col], s=12, alpha=0.75, c=sub["_color"], edgecolors="none")
    ax1.set_xlim(-5, 5); ax1.set_ylim(0, 8); ax1.set_aspect("equal")
    ax1.axhline(0, color="black", linewidth=1); ax1.axvline(0, color="black", linewidth=1)
    ax1.set_xlabel(x_col); ax1.set_ylabel(y_col)
    ax1.set_title(f"All Releases (n={len(sub)})", fontweight="bold")

    for _, row in means.iterrows():
        draw_stylized_arm(ax2, (0.0, 0.0), (float(row["mean_x"]), float(row["mean_y"])), ring_color=row["color"])
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
# D1 STATS
# ──────────────────────────────────────────────────────────────────────────────
DISPLAY_COLS_H = ['Team','Batter','PA','AB','Hits','2B','3B','HR','HBP','BB','K','BA','OBP','SLG','OPS']
RATE_COLS_H    = ['BA','OBP','SLG','OPS']

def compute_hitter_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pr = df['PlayResult'].astype(str).str.lower()

    ab_values  = {'single','double','triple','homerun','out','error','fielderschoice'}
    hit_values = {'single','double','triple','homerun'}

    df['is_ab']   = pr.isin(ab_values).astype(int)
    df['is_hit']  = pr.isin(hit_values).astype(int)
    df['is_bb']   = df['KorBB'].astype(str).str.lower().eq('walk').astype(int)
    df['is_k']    = df['KorBB'].astype(str).str.contains('strikeout', case=False, na=False).astype(int)
    df['is_hbp']  = df['PitchCall'].astype(str).eq('HitByPitch').astype(int)
    df['is_sf']   = df['PlayResult'].astype(str).str.contains('Sacrifice', case=False, na=False).astype(int)

    df['is_1b']   = pr.eq('single').astype(int)
    df['is_2b']   = pr.eq('double').astype(int)
    df['is_3b']   = pr.eq('triple').astype(int)
    df['is_hr']   = pr.eq('homerun').astype(int)

    df['is_pa'] = (df['is_ab'] | df['is_bb'] | df['is_hbp'] | df['is_sf']).astype(int)

    agg = (df.groupby(['BatterTeam','Batter'], as_index=False)
             .agg(PA=('is_pa','sum'), AB=('is_ab','sum'), Hits=('is_hit','sum'),
                  Doubles=('is_2b','sum'), Triples=('is_3b','sum'), Homeruns=('is_hr','sum'),
                  HBP=('is_hbp','sum'), BB=('is_bb','sum'), K=('is_k','sum'),
                  Singles=('is_1b','sum'), SF=('is_sf','sum')))

    agg['TB'] = agg['Singles'] + 2*agg['Doubles'] + 3*agg['Triples'] + 4*agg['Homeruns']
    agg = agg.rename(columns={'Doubles':'2B', 'Triples':'3B', 'Homeruns':'HR'})

    with np.errstate(divide='ignore', invalid='ignore'):
        ba  = np.where(agg['AB'] > 0, agg['Hits'] / agg['AB'], 0.0)
        slg = np.where(agg['AB'] > 0, agg['TB'] / agg['AB'], 0.0)
        obp_den = agg['AB'] + agg['BB'] + agg['HBP'] + agg['SF']
        obp_num = agg['Hits'] + agg['BB'] + agg['HBP']
        obp = np.divide(obp_num, obp_den, out=np.zeros_like(obp_den, dtype=float), where=obp_den > 0)
        ops = obp + slg

    agg['BA_num'], agg['OBP_num'], agg['SLG_num'], agg['OPS_num'] = ba, obp, slg, ops
    agg['BA']  = [f"{x:.3f}" for x in ba]
    agg['OBP'] = [f"{x:.3f}" for x in obp]
    agg['SLG'] = [f"{x:.3f}" for x in slg]
    agg['OPS'] = [f"{x:.3f}" for x in ops]

    agg = agg.rename(columns={'BatterTeam':'Team'})
    team_map_all = {**BIG_TEN_MAP, **BIG_12_MAP, **SEC_MAP, **ACC_MAP}
    agg['Team'] = agg['Team'].replace(team_map_all)

    keep = DISPLAY_COLS_H + [c+'_num' for c in RATE_COLS_H]
    return agg[keep].sort_values('BA_num', ascending=False)

DISPLAY_COLS_P = ['Team','Name','IP','Hits','HR','BB','HBP','SO','WHIP','BB/9','H/9','HR/9','SO/9']
RATE_NUMS_P    = ['IP_num','WHIP_num','BB/9_num','H/9_num','HR/9_num','SO/9_num']

def compute_pitcher_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pr_low  = df['PlayResult'].astype(str).str.lower()
    korbb_s = df['KorBB'].astype(str)
    pitch_s = df['PitchCall'].astype(str)

    hit_values    = {'single','double','triple','homerun'}
    df['is_hit']  = pr_low.isin(hit_values).astype(int)
    df['is_hr']   = pr_low.eq('homerun').astype(int)
    df['is_bb']   = korbb_s.str.lower().eq('walk').astype(int)
    df['is_so']   = korbb_s.str.contains('strikeout', case=False, na=False).astype(int)
    df['is_hbp']  = pitch_s.eq('HitByPitch').astype(int)
    df['is_sf']   = df['PlayResult'].astype(str).str.contains('Sacrifice', case=False, na=False).astype(int)

    if 'OutsOnPlay' in df.columns:
        outs_on_play = pd.to_numeric(df['OutsOnPlay'], errors='coerce').fillna(0).astype(int)
        outs_from_k  = df['is_so'] * (outs_on_play == 0)
        df['outs']   = outs_on_play + outs_from_k
    else:
        is_out = pr_low.eq('out').astype(int)
        df['outs'] = is_out + df['is_sf'] + df['is_so']

    grouped = (df.groupby(['PitcherTeam','Pitcher'], as_index=False)
                 .agg(Hits=('is_hit','sum'),
                      HR=('is_hr','sum'),
                      BB=('is_bb','sum'),
                      HBP=('is_hbp','sum'),
                      SO=('is_so','sum'),
                      Outs=('outs','sum')))

    grouped['IP_num'] = grouped['Outs'] / 3.0
    def ip_display(ip_num: float) -> str:
        outs = int(round(ip_num * 3)); return f"{outs//3}.{outs%3}"
    grouped['IP'] = grouped['IP_num'].apply(ip_display)

    ip = grouped['IP_num'].replace(0, np.nan)
    grouped['WHIP_num'] = ((grouped['BB'] + grouped['Hits']) / ip).fillna(0.0)
    grouped['BB/9_num'] = (grouped['BB']  / ip * 9).fillna(0.0)
    grouped['H/9_num']  = (grouped['Hits']/ ip * 9).fillna(0.0)
    grouped['HR/9_num'] = (grouped['HR']  / ip * 9).fillna(0.0)
    grouped['SO/9_num'] = (grouped['SO']  / ip * 9).fillna(0.0)

    grouped['WHIP'] = grouped['WHIP_num'].map(lambda x: f"{x:.3f}")
    grouped['BB/9'] = grouped['BB/9_num'].map(lambda x: f"{x:.2f}")
    grouped['H/9']  = grouped['H/9_num'].map(lambda x: f"{x:.2f}")
    grouped['HR/9'] = grouped['HR/9_num'].map(lambda x: f"{x:.2f}")
    grouped['SO/9'] = grouped['SO/9_num'].map(lambda x: f"{x:.2f}")

    grouped = grouped.rename(columns={'PitcherTeam':'Team', 'Pitcher':'Name'})
    team_map_all = {**BIG_TEN_MAP, **BIG_12_MAP, **SEC_MAP, **ACC_MAP}
    grouped['Team'] = grouped['Team'].map(team_map_all).fillna(grouped['Team'])

    def fmt_name(s: str) -> str:
        s = str(s).replace('\n', ' ')
        if ',' in s:
            last, first = [t.strip() for t in s.split(',', 1)]
            return f"{first} {last}"
        return s
    grouped['Name'] = grouped['Name'].apply(fmt_name)

    grouped = grouped.sort_values('WHIP_num', ascending=True)
    out = grouped[DISPLAY_COLS_P].copy()
    out = out.join(grouped[RATE_NUMS_P])
    return out

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS — pitcher first, then month/day based on that pitcher
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.caption("Use the ◀ chevron at top-left to hide/show this drawer.")
    st.radio("Mode", ["Nebraska Baseball", "D1 Baseball"], index=0, key="mode")

    if st.session_state["mode"] == "Nebraska Baseball":
        with st.expander("Report Controls", expanded=True):
            st.selectbox("Report Type", ["Pitcher Report","Hitter Report"], key="neb_report")

            if st.session_state["neb_report"] == "Pitcher Report":
                neb_df_all = df_all[df_all.get('PitcherTeam','') == 'NEB'].copy()
                pitchers_all = sorted(neb_df_all.get('Pitcher', pd.Series(dtype=object)).dropna().unique().tolist())
                st.selectbox("Pitcher", pitchers_all, key="neb_player")

                pitcher = st.session_state.get("neb_player")
                if pitcher:
                    df_pitcher_all = neb_df_all[neb_df_all['Pitcher'] == pitcher].copy()
                else:
                    df_pitcher_all = neb_df_all.iloc[0:0].copy()

                present_months = sorted(df_pitcher_all['Date'].dropna().dt.month.unique().tolist())
                st.multiselect(
                    "Months (optional)",
                    options=present_months,
                    format_func=lambda n: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][n-1],
                    default=[],
                    key="neb_pitch_months",
                )

                months_sel = st.session_state.get("neb_pitch_months", [])
                dates_series = df_pitcher_all['Date'].dropna()
                if months_sel:
                    dates_series = dates_series[dates_series.dt.month.isin(months_sel)]
                present_days = sorted(dates_series.dt.day.unique().tolist())
                st.multiselect(
                    "Days (optional)",
                    options=present_days,
                    default=[],
                    key="neb_pitch_days",
                )

            else:
                neb_mask = (df_all.get('BatterTeam','')=='NEB')
                neb_dates = sorted(pd.Series(df_all.loc[neb_mask, 'Date']).dropna().dt.date.unique())
                st.selectbox("Game Date", neb_dates, format_func=lambda d: format_date_long(d), key="neb_date")

                if "neb_date" in st.session_state and st.session_state["neb_date"] is not None:
                    df_date_tmp = df_all[df_all['Date'].dt.date==st.session_state["neb_date"]]
                    df_b_tmp = df_date_tmp[df_date_tmp['BatterTeam']=='NEB']
                    batters = sorted(df_b_tmp.get('Batter', pd.Series(dtype=object)).dropna().unique().tolist())
                else:
                    batters = []
                st.selectbox("Batter", batters, key="neb_player")

    else:
        with st.expander("D1 Statistics Controls", expanded=True):
            present_codes = pd.unique(
                pd.concat(
                    [df_all.get('BatterTeam', pd.Series(dtype=object)),
                     df_all.get('PitcherTeam', pd.Series(dtype=object))],
                    ignore_index=True
                ).dropna()
            )

            st.selectbox("Conference", ["Big Ten","Big 12","SEC","ACC"], index=0, key="d1_conference")
            team_map = CONF_MAP.get(st.session_state["d1_conference"], {})
            codes_in_conf = [code for code in team_map.keys() if code in present_codes]
            if not codes_in_conf:
                codes_in_conf = sorted(present_codes.tolist())
                team_map = {code: code for code in codes_in_conf}

            team_options = [(code, team_map.get(code, code)) for code in codes_in_conf]
            team_display = [name for _, name in team_options]
            team_sel_name = st.selectbox("Team", team_display, key="d1_team_name") if team_display else None

            team_code = None
            if team_sel_name:
                for code, name in team_options:
                    if name == team_sel_name:
                        team_code = code
                        break
            st.session_state['d1_team_code'] = team_code

            st.multiselect(
                "Months (optional)",
                options=[n for n,_ in MONTH_CHOICES],
                format_func=lambda n: MONTH_NAME_BY_NUM[n],
                default=[],
                key="d1_months",
            )
            st.multiselect(
                "Days (optional)",
                options=list(range(1,32)),
                default=[],
                key="d1_days",
            )
            st.radio("Stats Type", ["Hitter Statistics","Pitcher Statistics"], index=0, key="d1_stats_type")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────
mode = st.session_state.get("mode", "Nebraska Baseball")
# Dynamic page title per your request
st.title("Nebraska Baseball" if mode == "Nebraska Baseball" else "D1 Baseball")

# NEBRASKA
if mode == "Nebraska Baseball":
    report   = st.session_state.get('neb_report')
    player   = st.session_state.get('neb_player')
    logo_img = mpimg.imread(LOGO_PATH) if os.path.exists(LOGO_PATH) else None

    if report == "Pitcher Report":
        if not player:
            st.warning("Choose a pitcher in the Filters drawer.")
            st.stop()

        neb_df_all = df_all[df_all.get('PitcherTeam','')=='NEB'].copy()
        df_pitcher_all = neb_df_all[neb_df_all['Pitcher'] == player].copy()

        # Season "outings" = unique dates for this pitcher (across whole file/season)
        outings = int(pd.to_datetime(df_pitcher_all['Date'], errors="coerce").dt.date.nunique())
        st.subheader(f"{format_name(player)} — {outings} Outings")

        # Apply selected months/days (restricted to this pitcher)
        months_sel = st.session_state.get("neb_pitch_months", [])
        days_sel   = st.session_state.get("neb_pitch_days", [])
        neb_df = filter_by_month_day(df_pitcher_all, months=months_sel, days=days_sel)

        # Filter summary caption
        if months_sel or days_sel:
            if months_sel and not days_sel:
                mnames = ", ".join(MONTH_NAME_BY_NUM[m] for m in sorted(months_sel))
                st.caption(f"Filtered to months: {mnames}")
            elif months_sel and days_sel:
                mnames = ", ".join(MONTH_NAME_BY_NUM[m] for m in sorted(months_sel))
                dnames = ", ".join(str(d) for d in sorted(days_sel))
                st.caption(f"Filtered to months: {mnames} and days: {dnames}")
            else:
                dnames = ", ".join(str(d) for d in sorted(days_sel))
                st.caption(f"Filtered to days: {dnames} (across all months)")
        else:
            st.caption("Season totals for this pitcher (no month/day filter).")

        if neb_df.empty:
            st.info("No rows for the selected pitcher with current month/day filters.")
            st.stop()

        # 1) Post-game style (aggregated over selected dates)
        out = combined_pitcher_report(neb_df, player, logo_img, coverage=0.8)
        if out:
            fig, _summary = out
            st.pyplot(fig=fig)

        # 2) Heatmaps
        st.markdown("### Pitcher Heatmaps")
        heat_fig = combined_pitcher_heatmap_report(neb_df, player)
        if heat_fig:
            st.pyplot(fig=heat_fig)

        # 3) Release Points + new pitch-type filter
        # Build canonical pitch list from the currently filtered dataset
        types_available = (
            neb_df.get('AutoPitchType', pd.Series(dtype=object))
                 .dropna().map(canonicalize_type)
                 .replace("Unknown", np.nan).dropna().unique().tolist()
        )
        types_available = sorted(types_available)
        if types_available:
            st.markdown("### Release Points")
            sel_types = st.multiselect(
                "Pitch Types (Release Plot)",
                options=types_available,
                default=types_available,
                key="release_types"
            )
            rel_fig = release_points_figure(neb_df, player, include_types=sel_types if sel_types else [])
            if rel_fig:
                st.pyplot(fig=rel_fig)
        else:
            st.markdown("### Release Points")
            st.info("No recognizable pitch types available to plot.")

    else:  # Hitter Report (single date)
        sel_date = st.session_state.get('neb_date')
        if not sel_date:
            st.warning("Select a game date in the Filters drawer.")
            st.stop()

        st.subheader("Nebraska Hitter Report")
        df_date = df_all[df_all['Date'].dt.date==sel_date]
        df_b = df_date[df_date.get('BatterTeam','')=='NEB']
        if not player:
            st.warning("Choose a batter in the Filters drawer.")
            st.stop()

        st.markdown(f"**{player} — {format_date_long(sel_date)}**")
        tabs = st.tabs(["Standard", "Heatmaps"])
        with tabs[0]:
            fig = create_hitter_report(df_b, player, ncols=3)
            if fig: st.pyplot(fig=fig)
        with tabs[1]:
            fig = combined_hitter_heatmap_report(df_b, player, logo_img=logo_img)
            if fig: st.pyplot(fig=fig)

# D1
else:
    team_code  = st.session_state.get('d1_team_code')
    months_sel = st.session_state.get('d1_months', [])
    days_sel   = st.session_state.get('d1_days', [])
    stats_type = st.session_state.get('d1_stats_type', "Hitter Statistics")

    if not team_code:
        st.warning("Choose a conference and team in the Filters drawer.")
        st.stop()

    if stats_type == "Hitter Statistics":
        team_df = df_all[df_all.get('BatterTeam','') == team_code].copy()
    else:
        team_df = df_all[df_all.get('PitcherTeam','') == team_code].copy()

    team_df = filter_by_month_day(team_df, months=months_sel, days=days_sel)

    if team_df.empty:
        st.info("No rows after applying the selected filters.")
        st.stop()

    if len(months_sel) == 0 and len(days_sel) == 0:
        filt_text = "Season totals"
    elif len(months_sel) > 0 and len(days_sel) == 0:
        mnames = ", ".join(MONTH_NAME_BY_NUM[m] for m in sorted(months_sel))
        filt_text = f"Filtered to months: {mnames}"
    elif len(months_sel) > 0 and len(days_sel) > 0:
        mnames = ", ".join(MONTH_NAME_BY_NUM[m] for m in sorted(months_sel))
        dnames = ", ".join(str(d) for d in sorted(days_sel))
        filt_text = f"Filtered to months: {mnames} and days: {dnames}"
    else:
        dnames = ", ".join(str(d) for d in sorted(days_sel))
        filt_text = f"Filtered to days: {dnames} (across all months)"
    st.caption(filt_text)

    if stats_type == "Hitter Statistics":
        ranked_h = compute_hitter_rates(team_df)
        display_cols = DISPLAY_COLS_H + [c+'_num' for c in RATE_COLS_H]
        st.dataframe(ranked_h[display_cols], use_container_width=True)

        player_options = ranked_h['Batter'].unique().tolist()
        player_sel = st.selectbox("Highlight Player (optional)", ["(none)"] + player_options, index=0)
        if player_sel != "(none)":
            row = ranked_h[ranked_h['Batter'] == player_sel].head(1)
            if not row.empty:
                st.markdown("**Selected Player Stats**")
                st.table(row[['Team','Batter','PA','AB','Hits','2B','3B','HR','HBP','BB','K','BA','OBP','SLG','OPS']])

    else:
        table_p = compute_pitcher_table(team_df)
        display_cols_p = DISPLAY_COLS_P + RATE_NUMS_P
        st.dataframe(table_p[display_cols_p], use_container_width=True)

        p_options = table_p['Name'].unique().tolist()
        p_sel = st.selectbox("Highlight Pitcher (optional)", ["(none)"] + p_options, index=0)
        if p_sel != "(none)":
            row = table_p[table_p['Name'] == p_sel].head(1)
            if not row.empty:
                st.markdown("**Selected Pitcher Stats**")
                st.table(row[['Team','Name','IP','Hits','HR','BB','HBP','SO','WHIP','BB/9','H/9','HR/9','SO/9']])
