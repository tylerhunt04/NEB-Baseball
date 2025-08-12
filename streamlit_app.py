# unified_baseball_app.py
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2, gaussian_kde
from numpy.linalg import LinAlgError
from matplotlib import colors

# ─── CONFIG ───────────────────────────────────────────────────────────────────
NEB_DATA_PATH = "B10C25_small.parquet"         # Nebraska-only (trimmed)
LOGO_PATH     = "Nebraska-Cornhuskers-Logo.png"

st.set_page_config(layout="wide", page_title="Baseball Reports")
st.title("Baseball Analytics")

# ─── COLORMAP / STRIKE ZONE CONSTANTS ─────────────────────────────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
    N=256,
)

def get_zone_bounds():
    # fixed MLB zone (kept constant for heatmap/scatter parity)
    left, bottom = -0.83, 1.17
    width, height = 1.66, 2.75
    return left, bottom, width, height

def get_view_bounds():
    left, bottom, width, height = get_zone_bounds()
    mx, my = width * 0.8, height * 0.6
    return left - mx, left + width + mx, bottom - my, bottom + height + my

# ─── PITCH COLORS ─────────────────────────────────────────────────────────────
def get_pitch_color(ptype):
    if isinstance(ptype, str) and (ptype.lower().startswith("four-seam fastball") or ptype.lower() == "fastball"):
        return "#E60026"
    savant = {
        "sinker": "#FF9300","cutter": "#800080","changeup": "#008000","curveball": "#0033CC",
        "slider": "#CCCC00","splitter": "#00CCCC","knuckle curve": "#000000","screwball": "#CC0066","eephus": "#666666",
    }
    return savant.get(str(ptype).lower(), "#E60026")

PITCH_COLORS = {
    "Four-Seam": "#E60026","Sinker": "#FF9300","Cutter": "#800080","Changeup": "#008000",
    "Curveball": "#0033CC","Slider": "#CCCC00","Splitter": "#00CCCC","Knuckle Curve": "#000000",
    "Screwball": "#CC0066","Eephus": "#666666",
}

# ─── POWER-4 TEAM MAPS (extend with your codes) ───────────────────────────────
BIG_TEN_MAP = {
    'ILL_ILL': 'Illinois','MIC_SPA': 'Michigan State','UCLA': 'UCLA','IOW_HAW': 'Iowa',
    'IU': 'Indiana','MAR_TER': 'Maryland','MIC_WOL': 'Michigan','MIN_GOL': 'Minnesota',
    'NEB': 'Nebraska','NOR_CAT': 'Northwestern','ORE_DUC': 'Oregon','OSU_BUC': 'Ohio State',
    'PEN_NIT': 'Penn State','PUR_BOI': 'Purdue','RUT_SCA': 'Rutgers','SOU_TRO': 'USC','WAS_HUS': 'Washington'
}
BIG_12_MAP = {
    # 'TEX_CODE': 'Texas', ...
}
SEC_MAP = {
    # 'LSU_CODE': 'LSU', ...
}
ACC_MAP = {
    # 'FSU_CODE': 'Florida State', ...
}
CONF_MAP = {"Big Ten": BIG_TEN_MAP, "Big 12": BIG_12_MAP, "SEC": SEC_MAP, "ACC": ACC_MAP}

# ─── HELPERS ─────────────────────────────────────────────────────────────────
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

def format_name(name):
    if isinstance(name, str) and ',' in name:
        last, first = [s.strip() for s in name.split(',', 1)]
        return f"{first} {last}"
    return str(name)

# ─── HITTER HEATMAP HELPERS ───────────────────────────────────────────────────
def plot_conditional(ax, sub, title):
    x_min, x_max, y_min, y_max = get_view_bounds()
    draw_strikezone(ax)
    x = sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy()
    y = sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(sub) < 10:
        for _, r in sub.iterrows():
            if np.isfinite(r.get('PlateLocSide', np.nan)) and np.isfinite(r.get('PlateLocHeight', np.nan)):
                color = PITCH_COLORS.get(r.get('AutoPitchType', ''), 'gray')
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

# ─── STANDARD PITCHER REPORT ───────────────────────────────────────────────────
def combined_pitcher_report(df, pitcher_name, logo_img, coverage=0.8):
    df_p = df[df['Pitcher'] == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' on that date.")
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
    axm = fig.add_subplot(gs[0, 0]); axm.set_title('Movement Plot')
    chi2v = chi2.ppf(0.8, df=2)
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

    fig.suptitle(f"{pitcher_name} – Full Report", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, summary

# ─── PITCHER HEATMAPS (auto-switch to scatter if n<12) ────────────────────────
def combined_pitcher_heatmap_report(df, pitcher_name, logo_path, grid_size=100):
    df_p = df[df['Pitcher'] == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' on that date.")
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

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 0.6], hspace=0.4, wspace=0.3)

    # top 4 pitch types
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

    # summary metrics
    fp = strike_rate(df_p[(df_p['Balls']==0) & (df_p['Strikes']==0)])
    mix = strike_rate(df_p[((df_p['Balls']==1)&(df_p['Strikes']==0)) | ((df_p['Balls']==0)&(df_p['Strikes']==1)) | ((df_p['Balls']==1)&(df_p['Strikes']==1))])
    hp = strike_rate(df_p[((df_p['Balls']==2)&(df_p['Strikes']==0)) | ((df_p['Balls']==2)&(df_p['Strikes']==1)) | ((df_p['Balls']==3)&(df_p['Strikes']==1))])
    two = strike_rate(df_p[(df_p['Strikes']==2) & (df_p['Balls']<3)])
    metrics = pd.DataFrame({'1st Pitch %':[fp],'Mix Count %':[mix],'Hitter+ %':[hp],'2-Strike %':[two]}).round(1)

    axt = fig.add_subplot(gs[2, :]); axt.axis('off')
    tbl = axt.table(cellText=metrics.values, colLabels=metrics.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Strike Percentage by Count', y=0.75, fontweight='bold')

    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88, 0.92, 0.10, 0.10], anchor='NE', zorder=10); axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); fig.suptitle(f"{pitcher_name} – Heatmap Report", fontsize=18, y=0.98, fontweight='bold')
    return fig

# ─── HITTER HEATMAPS ──────────────────────────────────────────────────────────
def combined_hitter_heatmap_report(df, batter, logo_img=None):
    df_b = df[df['Batter'] == batter].copy()
    if df_b.empty:
        st.error(f"No data for batter '{batter}' on that date.")
        return None

    df_b['iscontact'] = df_b['PitchCall'].isin(['InPlay','FoulBallFieldable','FoulBallNotFieldable'])
    df_b['iswhiff'] = df_b['PitchCall'].eq('StrikeSwinging')
    df_b['is95plus'] = df_b['ExitSpeed'] >= 95

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

    sub_contact_l = df_b[df_b['iscontact'] & (df_b['PitcherThrows']=='Left')]
    sub_contact_r = df_b[df_b['iscontact'] & (df_b['PitcherThrows']=='Right')]
    ax1 = fig.add_subplot(gs[0, 0]); plot_conditional(ax1, sub_contact_l, 'Contact vs LHP')
    ax2 = fig.add_subplot(gs[0, 2]); plot_conditional(ax2, sub_contact_r, 'Contact vs RHP')

    sub_whiff_l = df_b[df_b['iswhiff'] & (df_b['PitcherThrows']=='Left')]
    sub_whiff_r = df_b[df_b['iswhiff'] & (df_b['PitcherThrows']=='Right')]
    ax3 = fig.add_subplot(gs[0, 3]); plot_conditional(ax3, sub_whiff_l, 'Whiffs vs LHP')
    ax4 = fig.add_subplot(gs[0, 5]); plot_condional = plot_conditional(ax4, sub_whiff_r, 'Whiffs vs RHP')  # keep same function

    sub_95_l = df_b[df_b['is95plus'] & (df_b['PitcherThrows']=='Left')]
    sub_95_r = df_b[df_b['is95plus'] & (df_b['PitcherThrows']=='Right')]
    ax5 = fig.add_subplot(gs[0, 6]); plot_conditional(ax5, sub_95_l, 'Exit ≥95 vs LHP')
    ax6 = fig.add_subplot(gs[0, 8]); plot_conditional(ax6, sub_95_r, 'Exit ≥95 vs RHP')

    formatted = format_name(batter)
    fig.suptitle(formatted, fontsize=22, x=0.5, y=0.87)
    plt.tight_layout(rect=[0, 0, 1, 0.78])
    return fig

# ─── STANDARD HITTER REPORT ───────────────────────────────────────────────────
def create_hitter_report(df, batter, ncols=3):
    bdf = df[df['Batter'] == batter]
    pa = list(bdf.groupby(['GameID','Inning','Top/Bottom','PAofInning']))
    n_pa = len(pa); nrows = max(1, math.ceil(n_pa/ncols))
    descs = []
    for _, padf in pa:
        lines = []
        for _, p in padf.iterrows():
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType} {p.EffectiveVelo:.1f} MPH / {p.PitchCall}")
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
    if pa:
        date = pa[0][1]['Date'].iloc[0]
        fig.suptitle(f"{batter} Hitter Report for {date}", fontsize=16, x=0.55, y=1.0, fontweight='bold')

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

# ─── HITTER STATISTICS (D1-wide) ──────────────────────────────────────────────
DISPLAY_COLS = ['Team','Batter','PA','AB','Hits','2B','3B','HR','HBP','BB','K','BA','OBP','SLG','OPS']
RATE_COLS    = ['BA','OBP','SLG','OPS']

def compute_rates(df: pd.DataFrame) -> pd.DataFrame:
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
    agg['Team'] = agg['Team'].replace({**BIG_TEN_MAP, **BIG_12_MAP, **SEC_MAP, **ACC_MAP})

    keep = DISPLAY_COLS + [c+'_num' for c in RATE_COLS]
    return agg[keep].sort_values('BA_num', ascending=False)

# ─── DATA LOADERS ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    return df

# ─── MAIN MODE SELECT ─────────────────────────────────────────────────────────
mode = st.radio("Choose mode:", ["Nebraska Baseball", "D1 Baseball"], horizontal=True)

# ─── NEBRASKA REPORTS ─────────────────────────────────────────────────────────
if mode == "Nebraska Baseball":
    if not os.path.exists(NEB_DATA_PATH):
        st.error(f"Data not found at {NEB_DATA_PATH}")
        st.stop()
    df_all = load_parquet(NEB_DATA_PATH)

    needed = ['Date','PitcherTeam','BatterTeam','Pitcher','Batter','PlayResult','KorBB','PitchCall']
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    all_dates = sorted(df_all['Date'].dropna().unique())

    c1,c2,c3,c4 = st.columns(4)
    report  = c1.selectbox("Report Type", ["Pitcher Report","Hitter Report"])
    variant = c2.selectbox("Variant", ["Standard","Heatmap"])
    sel_date = c3.selectbox("Game Date", all_dates)

    df_date = df_all[df_all['Date']==sel_date]
    logo_img = mpimg.imread(LOGO_PATH) if os.path.exists(LOGO_PATH) else None

    if report == "Pitcher Report":
        df_p = df_date[df_date['PitcherTeam']=='NEB']
        pitchers = sorted(df_p['Pitcher'].dropna().unique().tolist())
        player = c4.selectbox("Pitcher", pitchers) if pitchers else None
        if not player:
            st.warning("No NEB pitchers for that date."); st.stop()
        st.subheader(f"{player} — {sel_date}")
        if variant == "Heatmap":
            fig = combined_pitcher_heatmap_report(df_p, player, LOGO_PATH)
            if fig: st.pyplot(fig=fig)
        else:
            out = combined_pitcher_report(df_p, player, logo_img, coverage=0.8)
            if out:
                fig, summary = out
                st.pyplot(fig=fig); st.table(summary)

    else:  # Hitter Report
        df_b = df_date[df_date['BatterTeam']=='NEB']
        batters = sorted(df_b['Batter'].dropna().unique().tolist())
        player = c4.selectbox("Batter", batters) if batters else None
        if not player:
            st.warning("No NEB batters for that date."); st.stop()
        st.subheader(f"{player} — {sel_date}")
        if variant == "Heatmap":
            fig = combined_hitter_heatmap_report(df_b, player, logo_img=logo_img)
            if fig: st.pyplot(fig=fig)
        else:
            fig = create_hitter_report(df_b, player, ncols=3)
            if fig: st.pyplot(fig=fig)

# ─── D1 HITTER STATISTICS ─────────────────────────────────────────────────────
else:
    if not os.path.exists(D1_DATA_PATH):
        st.error(f"Data not found at {D1_DATA_PATH}")
        st.stop()
    df_all = load_parquet(D1_DATA_PATH)

    needed = ['BatterTeam','Batter','PlayResult','KorBB','PitchCall']
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    conference = c1.selectbox("Conference", ["Big Ten","Big 12","SEC","ACC"], index=0)
    team_map = CONF_MAP.get(conference, {})

    present_codes = set(df_all['BatterTeam'].dropna().unique())
    codes_in_conf = [code for code in team_map.keys() if code in present_codes]

    if not team_map:
        st.warning(f"No team code map found for {conference}. Fill the *_MAP dict for cleaner names.")
    if not codes_in_conf:
        st.info(f"No teams from {conference} found in the dataset.")
        st.stop()

    options = sorted([(code, team_map.get(code, code)) for code in codes_in_conf], key=lambda t: t[1])
    team_display = [name for _, name in options]
    team_sel_name = c2.selectbox("Team", team_display)
    team_code = [code for code, name in options if name == team_sel_name][0]

    team_df = df_all[df_all['BatterTeam'] == team_code]
    if team_df.empty:
        st.info("No rows for that team.")
        st.stop()

    ranked = compute_rates(team_df)
    player_options = ranked['Batter'].unique().tolist()
    player_sel = c3.selectbox("Player", player_options)

    st.caption("Tip: sort by *_num columns (numeric) for accurate ordering of rates.")
    display_cols = DISPLAY_COLS + [c+'_num' for c in RATE_COLS]
    st.dataframe(ranked[display_cols], use_container_width=True)

    row = ranked[ranked['Batter'] == player_sel].head(1)
    if not row.empty:
        st.markdown("**Selected Player Stats**")
        st.table(row[['Team','Batter','PA','AB','Hits','2B','3B','HR','HBP','BB','K','BA','OBP','SLG','OPS']])
    else:
        st.info("Player not found in computed stats.")
