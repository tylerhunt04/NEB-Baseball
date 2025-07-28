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
LOGO_PATH = 'Nebraska-Cornhuskers-Logo.png'

st.title("Post-Game Hitter & Pitcher Reports")

uploaded_file = st.file_uploader("Upload Pitch Data CSV", type="csv")
logo_file     = st.file_uploader("Upload Team Logo", type=["png","jpg","jpeg"])

# ─── HELPER: PITCHER COLORS ────────────────────────────────────────────────────
def get_pitch_color(ptype):
    if ptype.lower().startswith('four-seam fastball') or ptype.lower() == 'fastball':
        return '#E60026'
    savant = {
        'sinker':'#FF9300','cutter':'#800080','changeup':'#008000',
        'curveball':'#0033CC','slider':'#CCCC00','splitter':'#00CCCC',
        'knuckle curve':'#000000','screwball':'#CC0066','eephus':'#666666'
    }
    return savant.get(ptype.lower(), '#E60026')

# ─── PITCHER REPORT ────────────────────────────────────────────────────────────
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
    for ptype, g in grp:
        x, y = g['HorzBreak'], g['InducedVertBreak']
        clr   = get_pitch_color(ptype)
        axm.scatter(x, y, label=ptype, color=clr, alpha=0.7)
        if len(g) > 1:
            cov = np.cov(np.vstack((x,y)))
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
    axm.legend(title='Pitch Type',fontsize=8,title_fontsize=9,loc='upper right')

    # Summary table
    axt = fig.add_subplot(gs[1,0])
    axt.axis('off')
    tbl = axt.table(cellText=summary.values,
                    colLabels=summary.columns,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.5,1.5)
    axt.set_title('Summary Metrics', fontweight='bold', y=0.87)

    # Logo overlay (from uploaded file if provided)
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

# ─── HITTER REPORT (UNCHANGED) ────────────────────────────────────────────────
def create_hitter_report(df, batter, ncols=3):
    # … existing hitter report code …
    pass

# ─── STREAMLIT APP ────────────────────────────────────────────────────────────
if uploaded_file:
    df_all = pd.read_csv(uploaded_file)

    if 'Date' not in df_all.columns:
        st.error("Your CSV has no 'Date' column!")
        st.stop()

    # parse dates
    df_all['Date'] = pd.to_datetime(df_all['Date']).dt.date
    all_dates = sorted(df_all['Date'].unique())

    col1, col2, col3 = st.columns(3)
    report       = col1.selectbox("Report Type",
                                  ["Pitcher Report","Hitter Report"],
                                  key="report_type")
    selected_date = col2.selectbox("Game Date", all_dates, key="game_date")

    # filter for NEB pitchers and batters on that date
    df = df_all[
        (df_all['PitcherTeam']=='NEB') &
        (df_all['BatterTeam']=='NEB') &
        (df_all['Date']==selected_date)
    ]

    if report == "Pitcher Report":
        # only NEB pitchers
        pitchers = sorted(df['Pitcher'].unique())
        player   = col3.selectbox("Pitcher", pitchers, key="pitcher_name")
        st.subheader(f"{player} — {selected_date}")

        # load uploaded logo if provided
        logo_img = mpimg.imread(logo_file) if logo_file else None

        result = combined_pitcher_report(df, player, logo_img, coverage=0.8)
        if result:
            fig, summary = result
            st.pyplot(fig)
            st.table(summary)

    else:
        # only NEB batters
        batters = sorted(df['Batter'].unique())
        player  = col3.selectbox("Batter", batters, key="batter_name")
        st.subheader(f"{player} — {selected_date}")
        fig = create_hitter_report(df, player, ncols=3)
        st.pyplot(fig)

else:
    st.info("Please upload a CSV to begin.")
