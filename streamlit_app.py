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
logo_file = st.file_uploader("Upload Team Logo", type=["png","jpg","jpeg"])

# Helper functions

def get_pitch_color(ptype):
    if ptype.lower().startswith('four-seam fastball') or ptype.lower()=='fastball':
        return '#E60026'
    savant = {
        'sinker':'#FF9300','cutter':'#800080','changeup':'#008000',
        'curveball':'#0033CC','slider':'#CCCC00','splitter':'#00CCCC',
        'knuckle curve':'#000000','screwball':'#CC0066','eephus':'#666666'
    }
    return savant.get(ptype.lower(), '#E60026')

shape_map = {'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}
color_map = {
    'StrikeCalled':'#CCCC00','BallCalled':'green','FoulBallNotFieldable':'tan',
    'InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'
}

def draw_strikezone(ax,left=-0.83,right=0.83,bottom=1.5,top=3.5):
    ax.add_patch(Rectangle((left,bottom),right-left,top-bottom,
                           fill=False,linewidth=2,color='black'))
    dx,dy=(right-left)/3,(top-bottom)/3
    for i in (1,2):
        ax.add_line(Line2D([left+i*dx]*2,[bottom,top],linestyle='--',color='gray'))
        ax.add_line(Line2D([left,right],[bottom+i*dy]*2,linestyle='--',color='gray'))

# Hitter report

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
    chases=gd[(gd['PitchCall']=='StrikeSwinging')&(
        (gd['PlateLocSide']<-0.83)|(gd['PlateLocSide']>0.83)|
        (gd['PlateLocHeight']<1.5)|(gd['PlateLocHeight']>3.5)
    )].shape[0]
    fig.text(0.55,0.96,f"Whiffs: {whiffs} Hard Hits: {hardhits} Chases: {chases}",ha='center',va='top',fontsize=12)
    for idx,((_,inn,tb,_),padf) in enumerate(pa):
        row,col=divmod(idx,ncols)
        ax=fig.add_subplot(gs[row,col+1])
        draw_strikezone(ax)
        hand='LHP' if padf['PitcherThrows'].iloc[0].startswith('L') else 'RHP'
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
    res_handles=[Line2D([0],[0],marker='o',color='w',label=k,markerfacecolor=v,markersize=10,markeredgecolor='k') for k,v in color_map.items()]
    fig.legend(res_handles,color_map.keys(),title='Result',loc='lower right',bbox_to_anchor=(0.90,0.02))
    pitch_handles=[Line2D([0],[0],marker=m,color='w',label=k,markerfacecolor='gray',markersize=10,markeredgecolor='k') for k,m in shape_map.items()]
    fig.legend(pitch_handles,shape_map.keys(),title='Pitches',loc='lower right',bbox_to_anchor=(0.98,0.02))
    plt.tight_layout(rect=[0.12,0.05,1,0.88])
    return fig

# Streamlit app logic
if uploaded_file:
    df_all=pd.read_csv(uploaded_file)
    df=df_all[(df_all['PitcherTeam']=='NEB')&(df_all['BatterTeam']=='NEB')]
    col1,col2,col3=st.columns(3)
    report=col1.selectbox('Report Type',['Pitcher Report','Hitter Report'])
    date=col2.selectbox('Date',sorted(df['Date'].unique()))
    df_date=df[df['Date']==date]
    if report=='Pitcher Report':
        names=sorted(df_date['Pitcher'].unique())
    else:
        names=sorted(df_date['Batter'].unique())
    player=col3.selectbox('Player',names)
    if report=='Pitcher Report':
        df_sel = df_date[df_date['Pitcher'] == player]
    else:
        df_sel = df_date[df_date['Batter'] == player]
    if logo_file: st.image(logo_file,width=150)
    if report=='Hitter Report':
        st.subheader(f"{player} Hitter — {date}")
        fig=create_hitter_report(df_sel,player,3)
        st.pyplot(fig)
    else:
        st.subheader(f"{player} Pitcher — {date}")
        # Pitcher viz
        grp=df_sel.groupby('AutoPitchType')
        total=len(df_sel)
        fig=plt.figure(figsize=(8,12))
        gs=GridSpec(2,1,figure=fig,height_ratios=[1.5,0.7],hspace=0.3)
        axm=fig.add_subplot(gs[0,0]); axm.set_title('Movement Plot')
        axm.axhline(0,linestyle='--',color='grey'); axm.axvline(0,linestyle='--',color='grey')
        chi2v=chi2.ppf(0.8,df=2)
        for ptype,g in grp:
            x,y=g['HorzBreak'],g['InducedVertBreak']; clr=get_pitch_color(ptype)
            axm.scatter(x,y,color=clr,alpha=0.7,label=ptype)
            if len(g)>1:
                cov=np.cov(np.vstack((x,y)))
                vals,vecs=np.linalg.eigh(cov)
                ord=vals.argsort()[::-1]; vals,vecs=vals[ord],vecs[:,ord]
                ang=np.degrees(np.arctan2(*vecs[:,0][::-1]))
                w,h=2*np.sqrt(vals*chi2v)
                ell=Ellipse((x.mean(),y.mean()),w,h,angle=ang,
                            edgecolor=clr,facecolor=clr,alpha=0.2,linestyle='--',linewidth=1.5)
                axm.add_patch(ell)
        axm.set_xlim(-30,30);axm.set_ylim(-30,30);axm.set_aspect('equal','box')
        axm.set_xlabel('Horizontal Break'); axm.set_ylabel('Induced Vertical Break')
        axm.legend(title='Pitch Type',fontsize=8,title_fontsize=9)
                # build summary as list of dicts for one-dimensional columns
        summary_rows = []
        for ptype, g in grp:
            summary_rows.append({
                'Pitch Type': ptype,
                'Pitches': len(g),
                'Usage %': len(g) / total * 100,
                'Strike %': g['PitchCall'].isin([
                    'StrikeCalled','StrikeSwinging',
                    'FoulBallNotFieldable','FoulBallFieldable','InPlay'
                ]).mean() * 100,
                'Rel Speed': g['RelSpeed'].mean(),
                'Spin Rate': g['SpinRate'].mean(),
                'IVB': g['InducedVertBreak'].mean(),
                'HB': g['HorzBreak'].mean(),
                'Rel Height': g['RelHeight'].mean(),
                'VAA': g['VertApprAngle'].mean(),
                'Extension': g['Extension'].mean()
            })
        summary = pd.DataFrame(summary_rows)
        summary = summary.round({
            'Usage %':1, 'Strike %':1, 'Rel Speed':1, 'Spin Rate':1,
            'IVB':1, 'HB':1, 'Rel Height':2, 'VAA':1, 'Extension':2
        })
        # Sort summary by Pitches
        summary = summary.sort_values('Pitches', ascending=False)
        summary=summary.sort_values('Pitches',ascending=False)
        axt=fig.add_subplot(gs[1,0]);axt.axis('off')
        tbl=axt.table(cellText=summary.values,colLabels=summary.columns,cellLoc='center',loc='center')
        tbl.set_fontsize(10);tbl.scale(1.5,1.5)
        axt.set_title('Summary Metrics',fontweight='bold',y=0.87)
        fig.suptitle(f"{player} – Full Report",fontweight='bold',fontsize=16,y=0.98)
        st.pyplot(fig)
else:
    st.info("Please upload a CSV to begin.")
