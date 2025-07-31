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
from scipy.stats import chi2
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError
from matplotlib import colors

# ─── CONFIG / PATHS ───────────────────────────────────────────────────────────
CSV_PATH  = "5.31.2025 v HC.csv"  # adjust to your repo location
LOGO_PATH = "Nebraska-Cornhuskers-Logo.png"  # adjust to your repo location

st.title("Post-Game Hitter & Pitcher Reports")

# ─── CUSTOM COLORMAP FOR HEATMAP REPORT ────────────────────────────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [
        (0.0, "white"),
        (0.2, "deepskyblue"),
        (0.3, "white"),
        (0.7, "red"),
        (1.0, "red"),
    ],
)

# ─── STRIKEZONE / GRID CONSTANTS ───────────────────────────────────────────────
SZ_LEFT, SZ_RIGHT = -0.83, 0.83
SZ_BOTTOM, SZ_TOP = 1.17, 3.92
GRID_SIZE = 100

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def get_pitch_color(ptype):
    if ptype.lower().startswith("four-seam fastball") or ptype.lower() == "fastball":
        return "#E60026"
    savant = {
        "sinker": "#FF9300",
        "cutter": "#800080",
        "changeup": "#008000",
        "curveball": "#0033CC",
        "slider": "#CCCC00",
        "splitter": "#00CCCC",
        "knuckle curve": "#000000",
        "screwball": "#CC0066",
        "eephus": "#666666",
    }
    return savant.get(ptype.lower(), "#E60026")

def compute_density(x, y, grid_coords, mesh_shape):
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 2:
        return np.zeros(mesh_shape)
    try:
        kde = gaussian_kde(np.vstack([x_clean, y_clean]))
        return kde(grid_coords).reshape(mesh_shape)
    except LinAlgError:
        return np.zeros(mesh_shape)

def draw_strikezone(ax, sz_left, sz_bottom, sz_width, sz_height):
    zone = Rectangle((sz_left, sz_bottom), sz_width, sz_height,
                     fill=False, linewidth=2, linestyle='-', color='black')
    ax.add_patch(zone)
    for frac in (1/3, 2/3):
        ax.vlines(sz_left + sz_width * frac, sz_bottom, sz_bottom + sz_height,
                  colors='gray', linestyles='--', linewidth=1)
        ax.hlines(sz_bottom + sz_height * frac, sz_left, sz_left + sz_width,
                  colors='gray', linestyles='--', linewidth=1)

def strike_rate(sub_df):
    if len(sub_df) == 0:
        return np.nan
    strike_calls = ['StrikeCalled', 'StrikeSwinging', 'FoulBallNotFieldable', 'FoulBallFieldable', 'InPlay']
    return sub_df['PitchCall'].isin(strike_calls).sum() / len(sub_df) * 100

# ─── STANDARD PITCHER REPORT ───────────────────────────────────────────────────
def combined_pitcher_report(df, pitcher_name, logo_img, coverage=0.8):
    df_p = df[df['Pitcher'] == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' on that date.")
        return None

    total     = len(df_p)
    is_strike = df_p['PitchCall'].isin([
        'StrikeCalled','StrikeSwinging',
        'FoulBallNotFieldable','FoulBallFieldable','InPlay'
    ])
    grp = df_p.groupby('AutoPitchType')

    summary = pd.DataFrame({
        'Pitch Type': grp.size().index,
        'Pitches':    grp.size().values,
        'Usage %':    grp.size().values / total * 100,
        'Strike %':   grp.apply(lambda g: is_strike.loc[g.index].sum()/len(g)*100).values,
        'Rel Speed':  grp['RelSpeed'].mean().values,
        'Spin Rate':  grp['SpinRate'].mean().values,
        'IVB':        grp['InducedVertBreak'].mean().values,
        'HB':         grp['HorzBreak'].mean().values,
        'Rel Height': grp['RelHeight'].mean().values,
        'VAA':        grp['VertApprAngle'].mean().values,
        'Extension':  grp['Extension'].mean().values
    }).round({
        'Usage %':1,'Strike %':1,'Rel Speed':1,'Spin Rate':1,
        'IVB':1,'HB':1,'Rel Height':2,'VAA':1,'Extension':2
    }).sort_values('Pitches', ascending=False)

    cols = ['Pitch Type','Pitches','Usage %','Strike %',
            'Rel Speed','Spin Rate','IVB','HB','Rel Height','VAA','Extension']
    summary = summary[cols]

    fig = plt.figure(figsize=(8,12))
    gs  = GridSpec(2,1,figure=fig,height_ratios=[1.5,0.7],hspace=0.3)

    # Movement plot
    axm = fig.add_subplot(gs[0,0])
    axm.set_title('Movement Plot')
    chi2v = chi2.ppf(coverage, df=2)
    axm.axhline(0, linestyle='--', color='grey')
    axm.axvline(0, linestyle='--', color='grey')
    for ptype,g in grp:
        x, y = g['HorzBreak'], g['InducedVertBreak']
        clr = get_pitch_color(ptype)
        axm.scatter(x, y, label=ptype, color=clr, alpha=0.7)
        if len(g)>1:
            cov = np.cov(np.vstack((x,y)))
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:,order]
            ang = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            w, h = 2*np.sqrt(vals * chi2v)
            ell = Ellipse((x.mean(), y.mean()), w, h,
                          angle=ang, edgecolor=clr, facecolor=clr,
                          alpha=0.2, linestyle='--', linewidth=1.5)
            axm.add_patch(ell)
    axm.set_xlim(-30,30); axm.set_ylim(-30,30)
    axm.set_aspect('equal','box')
    axm.set_xlabel('Horizontal Break')
    axm.set_ylabel('Induced Vertical Break')
    axm.legend(title='Pitch Type',fontsize=8,title_fontsize=9,loc='upper right')

    # Summary table
    axt = fig.add_subplot(gs[1,0]); axt.axis('off')
    tbl = axt.table(cellText=summary.values,
                    colLabels=summary.columns,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5,1.5)
    axt.set_title('Summary Metrics', fontweight='bold', y=0.87)

    # Logo
    if logo_img is not None:
        axl = fig.add_axes([1,0.88,0.12,0.12],anchor='NE',zorder=10)
        axl.imshow(logo_img); axl.axis('off')
    elif os.path.exists(LOGO_PATH):
        logo = mpimg.imread(LOGO_PATH)
        axl  = fig.add_axes([1,0.88,0.12,0.12],anchor='NE',zorder=10)
        axl.imshow(logo); axl.axis('off')

    fig.suptitle(f"{pitcher_name} – Full Report", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0,0,1,0.95])
    return fig, summary

# ─── HEATMAP-STYLE PITCHER REPORT ──────────────────────────────────────────────
def combined_pitcher_heatmap_report(df, pitcher_name, logo_path,
                                    sz_left=SZ_LEFT, sz_right=SZ_RIGHT,
                                    sz_bottom=SZ_BOTTOM, sz_top=SZ_TOP,
                                    grid_size=GRID_SIZE):
    df_p = df[df['Pitcher'] == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' on that date.")
        return None

    sz_w, sz_h = sz_right - sz_left, sz_top - sz_bottom
    margin_x, margin_y = sz_w * 0.8, sz_h * 0.4
    x_min, x_max = sz_left - margin_x, sz_right + margin_x
    y_min, y_max = sz_bottom - margin_y, sz_top + margin_y
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi)
    grid_coords = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()])

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 0.6], hspace=0.4, wspace=0.3)

    top4 = list(df_p['AutoPitchType'].value_counts().index[:4])

    # Top row: top 4 pitch types
    for i, pitch in enumerate(top4):
        ax = fig.add_subplot(gs[0, i])
        sub = df_p[df_p['AutoPitchType'] == pitch]
        x = sub['PlateLocSide'].to_numpy()
        y = sub['PlateLocHeight'].to_numpy()
        if len(sub) < 15:
            ax.scatter(x, y, s=30, alpha=0.7, color='deepskyblue', edgecolors='black')
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        else:
            zi = compute_density(x, y, grid_coords, xi_mesh.shape)
            ax.imshow(zi, origin='lower',
                      extent=[x_min, x_max, y_min, y_max],
                      aspect='equal', cmap=custom_cmap)
        draw_strikezone(ax, sz_left, sz_bottom, sz_w, sz_h)
        ax.set_title(f"{pitch} (n={len(sub)})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.add_subplot(gs[0, 4]).axis('off')  # filler

    # Second row panels
    sub_vl = df_p[df_p['BatterSide'] == 'Left']
    ax_vl = fig.add_subplot(gs[1, 0])
    zi_vl = compute_density(sub_vl['PlateLocSide'], sub_vl['PlateLocHeight'], grid_coords, xi_mesh.shape)
    ax_vl.imshow(zi_vl, origin='lower',
                 extent=[x_min, x_max, y_min, y_max],
                 aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_vl, sz_left, sz_bottom, sz_w, sz_h)
    ax_vl.set_title(f"vs Left-Handed (n={len(sub_vl)})")
    ax_vl.set_xticks([]); ax_vl.set_yticks([])

    sub_vr = df_p[df_p['BatterSide'] == 'Right']
    ax_vr = fig.add_subplot(gs[1, 1])
    zi_vr = compute_density(sub_vr['PlateLocSide'], sub_vr['PlateLocHeight'], grid_coords, xi_mesh.shape)
    ax_vr.imshow(zi_vr, origin='lower',
                 extent=[x_min, x_max, y_min, y_max],
                 aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_vr, sz_left, sz_bottom, sz_w, sz_h)
    ax_vr.set_title(f"vs Right-Handed (n={len(sub_vr)})")
    ax_vr.set_xticks([]); ax_vr.set_yticks([])

    sub_whiff = df_p[df_p['PitchCall'] == 'StrikeSwinging']
    ax_whiff = fig.add_subplot(gs[1, 2])
    zi_whiff = compute_density(sub_whiff['PlateLocSide'], sub_whiff['PlateLocHeight'], grid_coords, xi_mesh.shape)
    ax_whiff.imshow(zi_whiff, origin='lower',
                    extent=[x_min, x_max, y_min, y_max],
                    aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_whiff, sz_left, sz_bottom, sz_w, sz_h)
    ax_whiff.set_title(f"Whiffs (n={len(sub_whiff)})")
    ax_whiff.set_xticks([]); ax_whiff.set_yticks([])

    sub_ks = df_p[df_p['KorBB'] == 'Strikeout']
    ax_ks = fig.add_subplot(gs[1, 3])
    zi_ks = compute_density(sub_ks['PlateLocSide'], sub_ks['PlateLocHeight'], grid_coords, xi_mesh.shape)
    ax_ks.imshow(zi_ks, origin='lower',
                 extent=[x_min, x_max, y_min, y_max],
                 aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_ks, sz_left, sz_bottom, sz_w, sz_h)
    ax_ks.set_title(f"Strikeouts (n={len(sub_ks)})")
    ax_ks.set_xticks([]); ax_ks.set_yticks([])

    sub_dmg = df_p[df_p['ExitSpeed'] >= 95]
    ax_dmg = fig.add_subplot(gs[1, 4])
    x_dmg = sub_dmg['PlateLocSide'].to_numpy()
    y_dmg = sub_dmg['PlateLocHeight'].to_numpy()
    if len(sub_dmg) < 15:
        ax_dmg.scatter(x_dmg, y_dmg, s=30, alpha=0.7, color='orange', edgecolors='black')
        ax_dmg.set_xlim(x_min, x_max); ax_dmg.set_ylim(y_min, y_max)
    else:
        zi_dmg = compute_density(x_dmg, y_dmg, grid_coords, xi_mesh.shape)
        ax_dmg.imshow(zi_dmg, origin='lower',
                      extent=[x_min, x_max, y_min, y_max],
                      aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_dmg, sz_left, sz_bottom, sz_w, sz_h)
    ax_dmg.set_title(f"Damage (n={len(sub_dmg)})")
    ax_dmg.set_xticks([]); ax_dmg.set_yticks([])

    # Summary metrics row
    fp = strike_rate(df_p[(df_p['Balls'] == 0) & (df_p['Strikes'] == 0)])
    mix = strike_rate(df_p[((df_p['Balls'] == 1) & (df_p['Strikes'] == 0)) |
                            ((df_p['Balls'] == 0) & (df_p['Strikes'] == 1)) |
                            ((df_p['Balls'] == 1) & (df_p['Strikes'] == 1))])
    hp = strike_rate(df_p[((df_p['Balls'] == 2) & (df_p['Strikes'] == 0)) |
                           ((df_p['Balls'] == 2) & (df_p['Strikes'] == 1)) |
                           ((df_p['Balls'] == 3) & (df_p['Strikes'] == 1))])
    two = strike_rate(df_p[(df_p['Strikes'] == 2) & (df_p['Balls'] < 3)])
    summary = pd.DataFrame({
        '1st Pitch %': [fp],
        'Mix Count %': [mix],
        'Hitter+ %': [hp],
        '2-Strike %': [two]
    }).round(1)
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis('off')
    tbl = ax_tbl.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.5, 1.5)
    ax_tbl.set_title('Strike Percentage by Count', y=0.75, fontweight='bold')

    # Logo
    if os.path.exists(logo_path):
        logo = mpimg.imread(logo_path)
        ax_logo = fig.add_axes([0.88, 0.92, 0.10, 0.10], anchor='NE', zorder=10)
        ax_logo.imshow(logo)
        ax_logo.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"{pitcher_name} – Heatmap Report", fontsize=18, y=0.98, fontweight='bold')
    return fig

# ─── HITTER REPORT ────────────────────────────────────────────────────────────
def create_hitter_report(df,batter,ncols=3):
    bdf=df[df['Batter']==batter]
    pa=list(bdf.groupby(['GameID','Inning','Top/Bottom','PAofInning']))
    n_pa=len(pa); nrows=math.ceil(n_pa/ncols)
    descs=[]
    for _,padf in pa:
        lines=[]
        for _,p in padf.iterrows():
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType} {p.EffectiveVelo:.1f} MPH / {p.PitchCall}")
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
    fig.suptitle(f"{batter} Hitter Report for {date}",fontsize=16,x=0.55,y=1.0,fontweight='bold')
    gd=pd.concat([grp for _,grp in pa])
    whiffs=(gd['PitchCall']=='StrikeSwinging').sum()
    hardhits=(gd['ExitSpeed']>95).sum()
    chases=gd[(gd['PitchCall']=='StrikeSwinging')&(((gd['PlateLocSide']<-0.83)|(gd['PlateLocSide']>0.83))|((gd['PlateLocHeight']<1.5)|(gd['PlateLocHeight']>3.5)))].shape[0]
    fig.text(0.55,0.96,f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",ha='center',va='top',fontsize=12)
    for idx,((_,inn,tb,_),padf) in enumerate(pa):
        row,col=divmod(idx,ncols)
        ax=fig.add_subplot(gs[row,col+1])
        ax.add_patch(Rectangle((-0.83,1.5),1.66,2.0,fill=False,linewidth=2,color='black'))
        hand='LHP' if padf['PitcherThrows'].iloc[0].startswith('L') else 'RHP'
        pitchr=padf['Pitcher'].iloc[0]
        for _,p in padf.iterrows():
            mk={'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}.get(p.AutoPitchType,'o')
            clr={'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan','InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.get(p.PitchCall,'black')
            sz=200 if p.AutoPitchType=='Slider' else 150
            ax.scatter(p.PlateLocSide,p.PlateLocHeight,marker=mk,c=clr,s=sz,edgecolor='white',linewidth=1,zorder=2)
            yoff=-0.05 if p.AutoPitchType=='Slider' else 0
            ax.text(p.PlateLocSide,p.PlateLocHeight+yoff,str(int(p.PitchofPA)),ha='center',va='center',fontsize=6,fontweight='bold',zorder=3)
        ax.set_xlim(-3,3);ax.set_ylim(0,5);ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}",fontsize=10,fontweight='bold')
        ax.text(0.5,0.1,f"vs {pitchr} ({hand})",transform=ax.transAxes,ha='center',va='top',fontsize=9,style='italic')
    axd=fig.add_subplot(gs[:,0]); axd.axis('off')
    y0=1.0; dy=1.0/(n_pa*5.0)
    for i,lines in enumerate(descs,1):
        axd.hlines(y0-dy*0.1,0,1,transform=axd.transAxes,color='black',linewidth=1)
        axd.text(0.02,y0,f"PA {i}",fontsize=6,fontweight='bold',transform=axd.transAxes)
        yln=y0-dy
        for ln in lines:
            axd.text(0.02,yln,ln,fontsize=6,transform=axd.transAxes)
            yln-=dy
        y0=yln-dy*0.05
    res_handles=[Line2D([0],[0],marker='o',color='w',label=k,markerfacecolor=v,markersize=10,markeredgecolor='k') for k,v in {'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan','InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'}.items()]
    fig.legend(res_handles,[h.get_label() for h in res_handles],title='Result',loc='lower right',bbox_to_anchor=(0.90,0.02))
    pitch_handles=[Line2D([0],[0],marker=m,color='w',label=k,markerfacecolor='gray',markersize=10,markeredgecolor='k') for k,m in {'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}.items()]
    fig.legend(pitch_handles,[h.get_label() for h in pitch_handles],title='Pitches',loc='lower right',bbox_to_anchor=(0.98,0.02))
    plt.tight_layout(rect=[0.12,0.05,1,0.88])
    return fig

# ─── STREAMLIT APP LOGIC ──────────────────────────────────────────────────────
# Load CSV from repo path
try:
    df_all = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"CSV not found at {CSV_PATH}")
    st.stop()

# validate presence
required = ['Date', 'PitcherTeam', 'BatterTeam', 'Pitcher', 'Batter']
missing = [c for c in required if c not in df_all.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df_all['Date'] = pd.to_datetime(df_all['Date']).dt.date
all_dates = sorted(df_all['Date'].unique())

col1, col2, col3, col4 = st.columns(4)
report        = col1.selectbox("Report Type", ["Pitcher Report","Hitter Report"], key="report_type")
variant       = col2.selectbox("Pitcher Variant", ["Standard","Heatmap"], key="variant") if report=="Pitcher Report" else None
selected_date = col3.selectbox("Game Date", all_dates, key="game_date")

df_date = df_all[df_all['Date'] == selected_date]

# load logo once
logo_img = None
if os.path.exists(LOGO_PATH):
    logo_img = mpimg.imread(LOGO_PATH)
else:
    st.warning(f"Logo not found at {LOGO_PATH}")

if report == "Pitcher Report":
    df_p     = df_date[df_date['PitcherTeam'] == 'NEB']
    pitchers = sorted(df_p['Pitcher'].unique())
    player   = col4.selectbox("Pitcher", pitchers, key="pitcher_name")
    st.subheader(f"{player} — {selected_date}")

    if variant == "Heatmap":
        fig = combined_pitcher_heatmap_report(df_p, player, LOGO_PATH)
        if fig:
            st.pyplot(fig=fig)
    else:
        result = combined_pitcher_report(df_p, player, logo_img, coverage=0.8)
        if result:
            fig, summary = result
            st.pyplot(fig=fig)
            st.table(summary)

else:
    df_b    = df_date[df_date['BatterTeam'] == 'NEB']
    batters = sorted(df_b['Batter'].unique())
    player  = col4.selectbox("Batter", batters, key="batter_name")
    st.subheader(f"{player} — {selected_date}")
    fig = create_hitter_report(df_b, player, ncols=3)
    if fig:
        st.pyplot(fig=fig)
