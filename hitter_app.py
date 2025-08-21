# hitter_app.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from matplotlib import colors
import altair as alt

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nebraska Hitter Reports", layout="centered")  # wide OFF

DATA_PATH = "B10C25_hitter_app_columns.csv"  # update if needed
BANNER_CANDIDATES = [
    "NebraskaChampions.jpg",
    
]

HUSKER_RED = "#E60026"

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
# FORMATTERS
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

def themed_styler(df: pd.DataFrame, nowrap=True) -> pd.io.formats.style.Styler:
    styles = [
        {'selector': 'thead th', 'props': f'background-color: {HUSKER_RED}; color: white;'},
        {'selector': 'th.col_heading', 'props': f'background-color: {HUSKER_RED}; color: white;'},
        {'selector': 'th', 'props': f'background-color: {HUSKER_RED}; color: white;'},
    ]
    if nowrap:
        styles.append({'selector': 'td', 'props': 'white-space: nowrap;'})
    return (df.style
            .hide(axis="index")
            .set_table_styles(styles))

def pick_col(df: pd.DataFrame, *cands) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in cands:
        if name and name.lower() in lower:
            return lower[name.lower()]
    return None

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
    pitchcall_col = pick_col(df, "PitchCall", "Pitch Call", "Pitch_Call", "Pitch_Result")
    tag_col       = pick_col(df, "TaggedHitType", "Tagged Hit Type", "BBType", "HitType", "BattedBallType")
    bearing_col   = pick_col(df, "Bearing", "SprayAngle", "Spray Angle", "Spray")
    batside_col   = pick_col(df, "BatterSide", "Batter Side", "Bats", "Stand", "BatterBatSide", "BatterBats")

    s_call = df.get(pitchcall_col, pd.Series(dtype=object)).astype(str).str.replace(r"\s+", "", regex=True).str.lower()
    inplay = df[s_call.eq("inplay")].copy()

    if tag_col is None:
        inplay["__tag"] = pd.NA
        tag_use = "__tag"
    else:
        tag_use = tag_col

    if bearing_col is None:
        inplay["__bearing"] = np.nan
        bearing_use = "__bearing"
    else:
        bearing_use = bearing_col

    if batside_col is None:
        inplay["__batside"] = ""
        batside_use = "__batside"
    else:
        batside_use = batside_col

    def norm_tag(x: str) -> str:
        t = "".join(ch for ch in str(x).lower() if ch.isalpha())
        if t.startswith("ground"): return "groundball"
        if t.startswith("fly"):    return "flyball"
        if t.startswith("line"):   return "linedrive"
        if "pop" in t or "infieldfly" in t: return "popup"
        return ""

    inplay["_tag_norm"] = inplay[tag_use].apply(norm_tag)
    inplay["_bearing"]  = pd.to_numeric(inplay[bearing_use], errors="coerce")
    inplay["_side1"]    = inplay[batside_use].astype(str).str.strip().str.upper().str[0]

    def assign_spray_category_row(row):
        ang  = row["_bearing"]
        side = row["_side1"]
        if not np.isfinite(ang):
            return np.nan
        if -15 <= ang <= 15:
            return "Straight"
        if ang < -15:
            return "Pull" if side == "R" else "Opposite"
        return "Opposite" if side == "R" else "Pull"

    inplay["spray_cat"] = inplay.apply(assign_spray_category_row, axis=1)

    tot = len(inplay)
    def pct(mask: pd.Series) -> float:
        return round(100 * float(mask.mean()), 1) if tot else 0.0

    bb = pd.DataFrame([{
        "Ground ball %": pct(inplay["_tag_norm"].eq("groundball")),
        "Fly ball %":    pct(inplay["_tag_norm"].eq("flyball")),
        "Line drive %":  pct(inplay["_tag_norm"].eq("linedrive")),
        "Popup %":       pct(inplay["_tag_norm"].eq("popup")),
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
    pitchcall_col = pick_col(df, "PitchCall", "Pitch Call", "Pitch_Call", "Pitch_Result")
    play_col      = pick_col(df, "PlayResult", "Play Result", "Play_Result")
    korbb_col     = pick_col(df, "KorBB", "K_or_BB", "K/BB", "BB/K")
    exit_col      = pick_col(df, "ExitSpeed", "ExitVelo", "ExitVelocity", "Exit_Speed", "EV")
    angle_col     = pick_col(df, "Angle", "LaunchAngle", "Launch_Angle", "Launch Angle", "LA")
    pitchofpa_col = pick_col(df, "PitchofPA", "PitchOfPA", "Pitch_of_PA", "Pitch # of PA")

    s_call    = df.get(pitchcall_col, pd.Series(dtype=object))
    play      = df.get(play_col,      pd.Series(dtype=object))
    korbb     = df.get(korbb_col,     pd.Series(dtype=object))
    exitv     = pd.to_numeric(df.get(exit_col,  pd.Series(dtype=float)), errors="coerce")
    angle     = pd.to_numeric(df.get(angle_col, pd.Series(dtype=float)), errors="coerce")
    pitchofpa = pd.to_numeric(df.get(pitchofpa_col, pd.Series(dtype=float)), errors="coerce")

    pa_mask   = pitchofpa.eq(1)
    hit_mask  = (s_call.astype(str).eq("InPlay") & play.astype(str).isin(["Single","Double","Triple","HomeRun"]))
    so_mask   = korbb.astype(str).eq("Strikeout")
    bbout     = s_call.astype(str).eq("InPlay") & play.astype(str).eq("Out")
    fc_mask   = play.astype(str).eq("FieldersChoice")
    err_mask  = play.astype(str).eq("Error")
    walk_mask = korbb.astype(str).eq("Walk")
    hbp_mask  = s_call.astype(str).eq("HitByPitch")

    hits   = int(hit_mask.sum())
    so     = int(so_mask.sum())
    bbouts = int(bbout.sum())
    fc     = int(fc_mask.sum())
    err    = int(err_mask.sum())
    ab     = hits + so + bbouts + fc + err

    walks = int(walk_mask.sum())
    hbp   = int(hbp_mask.sum())
    pa    = int(pa_mask.sum())

    inplay_mask = s_call.astype(str).eq("InPlay")
    bases = (play.astype(str).eq("Single").sum()
             + 2*play.astype(str).eq("Double").sum()
             + 3*play.astype(str).eq("Triple").sum()
             + 4*play.astype(str).eq("HomeRun").sum())

    avg_exit  = exitv[inplay_mask].mean()
    max_exit  = exitv[inplay_mask].max()
    avg_angle = angle[inplay_mask].mean()

    ba  = hits/ab if ab else 0.0
    obp = (hits + walks + hbp)/pa if pa else 0.0
    slg = bases/ab if ab else 0.0
    ops = obp + slg
    hard = (exitv[inplay_mask] >= 95).mean()*100 if inplay_mask.any() else 0.0
    k_pct  = (so/pa*100) if pa else 0.0
    bb_pct = (walks/pa*100) if pa else 0.0

    stats = pd.DataFrame([{
        "Avg Exit Vel": round(avg_exit,  2) if pd.notna(avg_exit)  else np.nan,
        "Max Exit Vel": round(max_exit,  2) if pd.notna(max_exit)  else np.nan,
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

    if pa_groups:
        date = pa_groups[0][1].get('Date').iloc[0]
        date_str = format_date_long(date)
        fig.suptitle(f"{batter} Hitter Report for {date_str}",
                     fontsize=16, x=0.55, y=1.0, fontweight='bold')

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
            x = p.get('PlateLocSide'); y = p.get('PlateLocHeight')
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
# STREAMLIT (ALTair) HEATMAPS — use Profiles filters (single panel each)
# ──────────────────────────────────────────────────────────────────────────────
def make_zone_layers():
    # strike zone rectangle + dashed thirds (Altair)
    zone_df = pd.DataFrame([{"x0": -0.83, "x1": 0.83, "y0": 1.5, "y1": 3.5}])
    rect = alt.Chart(zone_df).mark_rect(fillOpacity=0, stroke="black", strokeWidth=2).encode(
        x=alt.X("x0:Q", scale=alt.Scale(domain=list(X_LIM)), title=""),
        x2="x1:Q",
        y=alt.Y("y0:Q", scale=alt.Scale(domain=list(Y_LIM)), title=""),
        y2="y1:Q",
    )
    thirds = []
    for fx in (1/3, 2/3):
        x = -0.83 + (0.83 - (-0.83)) * fx
        thirds.append(alt.Chart(pd.DataFrame({"x": [x]})).mark_rule(strokeDash=[4,4], stroke="gray").encode(x="x:Q"))
    for fy in (1/3, 2/3):
        y = 1.5 + (3.5 - 1.5) * fy
        thirds.append(alt.Chart(pd.DataFrame({"y": [y]})).mark_rule(strokeDash=[4,4], stroke="gray").encode(y="y:Q"))
    return rect, thirds

def altair_heatmap(df_sub: pd.DataFrame, title: str):
    if df_sub is None or df_sub.empty:
        return alt.Chart(pd.DataFrame({"PlateLocSide": [], "PlateLocHeight": []})).mark_rect().properties(
            width=320, height=320, title=title
        )

    base = alt.Chart(df_sub).transform_filter(
        (alt.datum.PlateLocSide >= X_LIM[0]) & (alt.datum.PlateLocSide <= X_LIM[1]) &
        (alt.datum.PlateLocHeight >= Y_LIM[0]) & (alt.datum.PlateLocHeight <= Y_LIM[1])
    ).mark_rect().encode(
        x=alt.X("PlateLocSide:Q", bin=alt.Bin(maxbins=30), scale=alt.Scale(domain=list(X_LIM)), title=""),
        y=alt.Y("PlateLocHeight:Q", bin=alt.Bin(maxbins=30), scale=alt.Scale(domain=list(Y_LIM)), title=""),
        color=alt.Color("count():Q", title="Count", scale=alt.Scale(scheme="reds")),
        tooltip=[alt.Tooltip("count():Q", title="Count")]
    ).properties(width=320, height=320, title=title)

    rect, thirds = make_zone_layers()
    chart = alt.layer(base, rect, *thirds).resolve_scale(color='independent')
    return chart

def hitter_heatmaps_altair(df_b: pd.DataFrame, batter: str):
    sub = df_b[df_b.get('Batter') == batter].copy()
    if sub.empty:
        return None, None, None

    sub["PlateLocSide"]   = pd.to_numeric(sub.get("PlateLocSide"), errors="coerce")
    sub["PlateLocHeight"] = pd.to_numeric(sub.get("PlateLocHeight"), errors="coerce")
    sub["ExitSpeed"]      = pd.to_numeric(sub.get("ExitSpeed"), errors="coerce")

    contact = sub[sub.get("PitchCall").isin(["InPlay","FoulBallFieldable","FoulBallNotFieldable"])]
    whiffs  = sub[sub.get("PitchCall").eq("StrikeSwinging")]
    damage  = sub[sub.get("ExitSpeed") >= 95]

    return (
        altair_heatmap(contact, "Contact"),
        altair_heatmap(whiffs,  "Whiffs"),
        altair_heatmap(damage,  "Damage"),
    )

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
def show_banner():
    for p in BANNER_CANDIDATES:
        if os.path.exists(p):
            st.image(p, use_container_width=True)
            return

show_banner()
st.markdown("### Nebraska Hitter Reports")

# ──────────────────────────────────────────────────────────────────────────────
# TOP FILTERS — Standard Hitter Report (Player LEFT → Date RIGHT, single game)
# ──────────────────────────────────────────────────────────────────────────────
colB_player, colB_date = st.columns([1, 1])

all_batters = sorted(df_neb_bat.get('Batter').dropna().unique().tolist())
batter = colB_player.selectbox("Player", options=all_batters, index=0 if all_batters else None)

def opponent_for_date(df: pd.DataFrame, batter_name: str, date_val) -> str | None:
    if batter_name is None: return None
    day = pd.to_datetime(df['Date'], errors="coerce").dt.date
    rows = df[(df['Batter'] == batter_name) & (day == date_val)]
    if "PitcherTeam" in rows.columns and not rows.empty:
        opps = rows["PitcherTeam"].dropna()
        opps = opps[opps != "NEB"]
        if not opps.empty:
            return opps.mode().iloc[0]
    return None

# Dates just for selected batter
if batter:
    date_opts = sorted(pd.to_datetime(df_neb_bat[df_neb_bat['Batter'] == batter]['Date'],
                                      errors="coerce").dropna().dt.date.unique().tolist())
else:
    date_opts = []

def date_label(d):
    opp = opponent_for_date(df_neb_bat, batter, d)
    return f"{format_date_long(d)} ({opp})" if opp else format_date_long(d)

selected_date = colB_date.selectbox(
    "Game Date",
    options=date_opts,
    format_func=date_label,
    index=len(date_opts)-1 if date_opts else 0
)

df_date = df_neb_bat[pd.to_datetime(df_neb_bat['Date'], errors="coerce").dt.date == selected_date]

if not batter:
    st.info("Select a player to see the Standard Hitter Report.")
else:
    st.markdown("### Standard Hitter Report")
    fig_std = create_hitter_report(df_date, batter, ncols=3)
    if fig_std:
        st.pyplot(fig_std)

# ──────────────────────────────────────────────────────────────────────────────
# LOWER FILTERS — apply to Profiles & Altair Heatmaps
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Profiles & Heatmaps Filters")

colM, colD, colN, colH = st.columns([1.2, 1.2, 0.9, 1.9])

present_dates_all = (
    pd.to_datetime(df_neb_bat[df_neb_bat['Batter'] == batter]['Date'], errors="coerce")
      .dropna().dt.date if batter else pd.Series([], dtype="datetime64[ns]")
)
present_months = sorted(pd.Series(present_dates_all).map(lambda d: d.month).unique().tolist()) if len(present_dates_all) else []
sel_months = colM.multiselect(
    "Months",
    options=present_months,
    format_func=lambda n: MONTH_NAME_BY_NUM.get(n, str(n)),
    default=[],
    key="prof_months",
)

# Days for the chosen months
if batter:
    dser = pd.Series(present_dates_all)
    if sel_months:
        dser = dser[dser.map(lambda d: d.month).isin(sel_months)]
    present_days = sorted(pd.Series(dser).map(lambda d: d.day).unique().tolist())
else:
    present_days = []
sel_days = colD.multiselect("Days", options=present_days, default=[], key="prof_days")

lastN = int(colN.number_input("Last N games", min_value=0, max_value=50, step=1, value=0, format="%d", key="prof_lastn"))
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

# ──────────────────────────────────────────────────────────────────────────────
# PROFILES (stacked) + ALTAIR HEATMAPS on filtered data
# ──────────────────────────────────────────────────────────────────────────────
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
        season_year = int(pd.to_datetime(selected_date).year) if selected_date else "—"

    # Month column (if any months chosen)
    month_label = ", ".join(MONTH_NAME_BY_NUM.get(m, str(m)) for m in sorted(sel_months)) if sel_months else None

    # Batted Ball Profile
    bb_df = create_batted_ball_profile(df_profiles).copy()
    for c in bb_df.columns:
        bb_df[c] = bb_df[c].apply(lambda v: fmt_pct(v, decimals=1))
    bb_df.insert(0, "Season", season_year)
    if month_label:
        bb_df.insert(1, "Month", month_label)
    st.markdown("### Batted Ball Profile")
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
    st.markdown("### Plate Discipline Profile")
    st.table(themed_styler(pd_df))

    # Batting Statistics
    st_df, pa_cnt, ab_cnt = create_batting_stats_profile(df_profiles)
    st_df = st_df.copy()
    for c in ["AVG", "OBP", "SLG", "OPS"]:
        if c in st_df.columns:
            st_df[c] = st_df[c].apply(fmt_avg3)
    if "Avg Exit Vel" in st_df.columns:
        st_df["Avg Exit Vel"] = st_df["Avg Exit Vel"].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "—")
    if "Max Exit Vel" in st_df.columns:
        st_df["Max Exit Vel"] = st_df["Max Exit Vel"].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "—")
    if "Avg Angle" in st_df.columns:
        st_df["Avg Angle"] = st_df["Avg Angle"].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "—")
    if "HardHit %" in st_df.columns:
        st_df["HardHit %"] = st_df["HardHit %"].apply(lambda v: fmt_pct(v, decimals=1))
    if "K %" in st_df.columns:
        st_df["K %"] = st_df["K %"].apply(fmt_pct2)
    if "BB %" in st_df.columns:
        st_df["BB %"] = st_df["BB %"].apply(fmt_pct2)
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

    st.markdown("### Batting Statistics")
    st.table(themed_styler(st_df, nowrap=True))

    # Altair Heatmaps (use Profiles filters)
    st.markdown("### Hitter Heatmaps")
    c_chart, w_chart, d_chart = hitter_heatmaps_altair(df_profiles, batter)
    cols = st.columns(3)
    if c_chart is not None:
        cols[0].altair_chart(c_chart, use_container_width=True)
    if w_chart is not None:
        cols[1].altair_chart(w_chart, use_container_width=True)
    if d_chart is not None:
        cols[2].altair_chart(d_chart, use_container_width=True)
