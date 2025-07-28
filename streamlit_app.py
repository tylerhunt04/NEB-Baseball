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

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_FILE   = '5.31.2025 v HC.csv'
LOGO_PATH   = 'Nebraska-Cornhuskers-Logo.png'

st.title("Post-Game Hitter & Pitcher Reports")

uploaded_file = st.file_uploader("Upload Pitch Data CSV", type="csv")
logo_file     = st.file_uploader("Upload Team Logo", type=["png","jpg","jpeg"])


# ─── PITCHER REPORT FUNCTION ─────────────────────────────────────────────────
def get_pitch_color(ptype):
    if ptype.lower().startswith('four-seam fastball') or ptype.lower()=='fastball':
        return '#E60026'
    savant_colors = {
        'sinker':'#FF9300','cutter':'#800080','changeup':'#008000',
        'curveball':'#0033CC','slider':'#CCCC00','splitter':'#00CCCC',
        'knuckle curve':'#000000','screwball':'#CC0066','eephus':'#666666'
    }
    return savant_colors.get(ptype.lower(), '#E60026')

def combined_pitcher_report(df, pitcher_name, logo_path, coverage=0.8):
    df_p = df[df['Pitcher'] == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' on that date.")
        return None

    total   = len(df_p)
    is_strike = df_p['PitchCall'].isin([
        'StrikeCalled','StrikeSwinging',
        'FoulBallNotFieldable','FoulBallFieldable','InPlay'
    ])

    grp = df_p.groupby('AutoPitchType')
    summary = pd.DataFrame({
        'Pitch Type': grp.size().index,
        'Pitches':    grp.size().values,
        'Usage %':    grp.size().values / total * 100,
        'Strike %':   grp.apply(lambda g: is_strike.loc[g.index].sum() / len(g) * 100).values,
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
    })

    # sort by Pitches
    summary = summary.sort_values('Pitches', ascending=False)
    cols    = ['Pitch Type','Pitches','Usage %','Strike %',
               'Rel Speed','Spin Rate','IVB','HB','Rel Height','VAA','Extension']
    summary = summary[cols]

    # build figure
    fig = plt.figure(figsize=(8, 12))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[1.5,0.7], hspace=0.3)

    # movement plot
    axm = fig.add_subplot(gs[0,0])
    axm.set_title('Movement Plot')
    chi2v = chi2.ppf(coverage, df=2)
    axm.axhline(0, linestyle='--', color='grey')
    axm.axvline(0, linestyle='--', color='grey')

    for ptype, g in grp:
        x = g['HorzBreak']; y = g['InducedVertBreak']
        clr = get_pitch_color(ptype)
        axm.scatter(x, y, label=ptype, color=clr, alpha=0.7)
        if len(g) > 1:
            cov = np.cov(np.vstack((x, y)))
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:,order]
            ang = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            w,h = 2*np.sqrt(vals * chi2v)
            ell = Ellipse((x.mean(), y.mean()), w, h,
                          angle=ang, edgecolor=clr, facecolor=clr,
                          alpha=0.2, linestyle='--', linewidth=1.5)
            axm.add_patch(ell)

    axm.set_xlim(-30,30); axm.set_ylim(-30,30)
    axm.set_aspect('equal','box')
    axm.set_xlabel('Horizontal Break')
    axm.set_ylabel('Induced Vertical Break')
    axm.legend(title='Pitch Type', fontsize=8, title_fontsize=9, loc='upper right')

    # summary table
    axt = fig.add_subplot(gs[1,0])
    axt.axis('off')
    tbl = axt.table(cellText=summary.values,
                    colLabels=summary.columns,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.5, 1.5)
    axt.set_title('Summary Metrics', fontweight='bold', y=0.87)

    # logo
    if os.path.exists(logo_path):
        logo = mpimg.imread(logo_path)
        axl  = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10)
        axl.imshow(logo); axl.axis('off')

    fig.suptitle(f"{pitcher_name} – Full Report", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0,0,1,0.95])
    return fig, summary


# ─── STREAMLIT APP LOGIC ──────────────────────────────────────────────────────
if uploaded_file:
    df_all = pd.read_csv(uploaded_file)
    df     = df_all[(df_all['PitcherTeam']=='NEB') & (df_all['BatterTeam']=='NEB')]

    col1, col2, col3 = st.columns(3)
    report = col1.selectbox('Report Type', ['Pitcher Report','Hitter Report'])
    date   = col2.selectbox('Date', sorted(df['Date'].unique()))
    df_d   = df[df['Date']==date]

    if report=='Pitcher Report':
        names   = sorted(df_d['Pitcher'].unique())
        player  = col3.selectbox('Pitcher', names)
        st.subheader(f"{player} – {date}")
        fig_sum = combined_pitcher_report(df_d, player,
                                          logo_path=LOGO_PATH, coverage=0.8)
        if fig_sum:
            fig, summary = fig_sum
            st.pyplot(fig)
            st.table(summary)

    else:
        # your existing create_hitter_report branch...
        pass

else:
    st.info("Please upload a CSV to begin.")
