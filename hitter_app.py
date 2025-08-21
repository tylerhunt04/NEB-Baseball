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
from matplotlib import colors

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG / PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "B10C25_hitter_app_columns.csv"  # Nebraska dataset (streamlit columns)
BANNER_PATH = "NebraskaChampions.jpg"                   # banner image
LOGO_PATH   = "Nebraska-Cornhuskers-Logo.png"                     # optional logo inside figures

st.set_page_config(layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# BANNER
# ──────────────────────────────────────────────────────────────────────────────
if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, use_container_width=True)
else:
    st.title("Nebraska Baseball — Hitter Reports")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def ensure_date(series_like):
    s = pd.to_datetime(series_like, errors="coerce")
    return s.dt.date

def format_name(name: str) -> str:
    if isinstance(name, str) and "," in name:
        last, first = [t.strip() for t in name.split(",", 1)]
        return f"{first} {last}"
    return str(name)

def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"

def format_date_long(d) -> str:
    if d is None or pd.isna(d): return ""
    d = pd.to_datetime(d).date()
    return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"

# Strike zone bounds (fixed)
def get_zone_bounds():
    left, bottom = -0.83, 1.17
    width, height = 1.66, 2.75
    return left, bottom, width, height

def get_view_bounds():
    left, bottom, width, height = get_zone_bounds()
    mx, my = width * 0.8, height * 0.6
    return left - mx, left + width + mx, bottom - my, bottom + height + my

def draw_strikezone(ax, sz_left=None, sz_bottom=None, sz_width=None, sz_height=None):
    left, bottom, width, height = get_zone_bounds()
    if sz_left   is None: sz_left   = left
    if sz_bottom is None: sz_bottom = bottom
    if sz_width  is None: sz_width  = width
    if sz_height is None: sz_height = height
    ax.add_patch(Rectangle((sz_left, sz_bottom), sz_width, sz_height, fill=False, linewidth=2, color="black"))
    for f in (1/3, 2/3):
        ax.vlines(sz_left + sz_width*f,  sz_bottom, sz_bottom+sz_height, colors="gray", ls="--", lw=1)
        ax.hlines(sz_bottom + sz_height*f, sz_left, sz_left+sz_width,     colors="gray", ls="--", lw=1)

# Density helpers
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

# Pitch/result glyph maps (standard report)
SHAPE_MAP = {'Fastball':'o','Curveball':'s','Slider':'^','Changeup':'D'}
COLOR_MAP = {
    'StrikeCalled':'#CCCC00','BallCalled':'green',
    'FoulBallNotFieldable':'tan','FoulBallFieldable':'tan',
    'InPlay':'#6699CC','StrikeSwinging':'red','HitByPitch':'lime'
}

# Heatmap colormap
CUSTOM_CMAP = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0,"white"),(0.2,"deepskyblue"),(0.3,"white"),(0.7,"red"),(1.0,"red")],
    N=256,
)

# ──────────────────────────────────────────────────────────────────────────────
# BATTED BALL / PLATE DISCIPLINE / BATTING STATS (stacked tables)
# ──────────────────────────────────────────────────────────────────────────────
def _spray_cat(row):
    ang = row.get("Bearing", np.nan)
    side = str(row.get("BatterSide","")).strip().upper()[:1]
    if pd.isna(ang):
        return np.nan
    if -15 <= ang <= 15:
        return "Straight"
    if ang < -15:
        return "Pull" if side == "R" else "Opposite"
    return "Opposite" if side == "R" else "Pull"

def pct(series_bool):
    if series_bool is None or len(series_bool) == 0:
        return 0.0
    return round(float(series_bool.mean()) * 100.0, 1)

def create_batted_ball_profile(df: pd.DataFrame) -> pd.DataFrame:
    inplay = df[df.get("PitchCall","") == "InPlay"].copy()
    if "TaggedHitType" not in inplay.columns:
        return pd.DataFrame([{
            "Ground ball %": 0.0, "Fly ball %": 0.0, "Line drive %": 0.0, "Popup %": 0.0,
            "Pull %": 0.0, "Straight %": 0.0, "Opposite %": 0.0
        }])
    if "Bearing" in inplay.columns and "BatterSide" in inplay.columns:
        inplay["spray_cat"] = inplay.apply(_spray_cat, axis=1)
    else:
        inplay["spray_cat"] = np.nan

    out = pd.DataFrame([{
        "Ground ball %": pct(inplay["TaggedHitType"].eq("GroundBall")) if "TaggedHitType" in inplay else 0.0,
        "Fly ball %":    pct(inplay["TaggedHitType"].eq("FlyBall")) if "TaggedHitType" in inplay else 0.0,
        "Line drive %":  pct(inplay["TaggedHitType"].eq("LineDrive")) if "TaggedHitType" in inplay else 0.0,
        "Popup %":       pct(inplay["TaggedHitType"].eq("Popup")) if "TaggedHitType" in inplay else 0.0,
        "Pull %":        pct(inplay["spray_cat"].eq("Pull")),
        "Straight %":    pct(inplay["spray_cat"].eq("Straight")),
        "Opposite %":    pct(inplay["spray_cat"].eq("Opposite")),
    }])
    return out

def create_plate_discipline_profile(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    pc = d.get("PitchCall", pd.Series(dtype=object))
    d["isswing"]   = pc.isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"])
    d["iswhiff"]   = pc.eq("StrikeSwinging")
    d["iscontact"] = pc.isin(["InPlay","FoulBallNotFieldable","FoulBallFieldable"])
    pls = pd.to_numeric(d.get("PlateLocSide", pd.Series(np.nan, index=d.index)), errors="coerce")
    plh = pd.to_numeric(d.get("PlateLocHeight", pd.Series(np.nan, index=d.index)), errors="coerce")
    d["isinzone"]  = pls.between(-0.83,0.83) & plh.between(1.5,3.5)

    zp = d["isinzone"]
    sw = d["isswing"]

    zone_swing = pct(sw[zp]) if zp.any() else 0.0
    zone_contact = 0.0
    denom = int(sw[zp].sum())
    if denom > 0:
        num = int(d.loc[zp & d["iscontact"], "iscontact"].sum())
        zone_contact = round(num / denom * 100.0, 1)

    return pd.DataFrame([{
        "Zone Pitches":   int(zp.sum()),
        "Zone %":         pct(zp),
        "Zone Swing %":   zone_swing,
        "Zone Contact %": zone_contact,
        "Chase %":        pct(sw & ~zp),
        "Swing %":        pct(sw),
        "Whiff %":        round((int(d["iswhiff"].sum()) / int(sw.sum()) * 100.0), 1) if int(sw.sum()) > 0 else 0.0,
    }])

def create_batting_stats_profile(df: pd.DataFrame):
    d = df.copy()
    d["PA"]        = d.get("PitchofPA", pd.Series(0, index=d.index)).fillna(0).astype(float).eq(1)
    d["Hit"]       = (d.get("PitchCall","") == "InPlay") & d.get("PlayResult","").isin(["Single","Double","Triple","HomeRun"])
    d["StrikeOut"] = d.get("KorBB","").eq("Strikeout")
    d["Walk"]      = d.get("KorBB","").eq("Walk")
    d["HBP"]       = d.get("PitchCall","").eq("HitByPitch")
    d["BBOut"]     = (d.get("PitchCall","") == "InPlay") & d.get("PlayResult","").eq("Out")
    d["FC"]        = d.get("PlayResult","").eq("FieldersChoice")
    d["Error"]     = d.get("PlayResult","").eq("Error")

    hits   = int(d["Hit"].sum())
    so     = int(d["StrikeOut"].sum())
    bbouts = int(d["BBOut"].sum())
    fc     = int(d["FC"].sum())
    err    = int(d["Error"].sum())
    ab     = hits + so + bbouts + fc + err

    walks = int(d["Walk"].sum())
    hbp   = int(d["HBP"].sum())
    pa    = int(d["PA"].sum())

    inplay = d[d.get("PitchCall","") == "InPlay"].copy()
    es = pd.to_numeric(inplay.get("ExitSpeed", pd.Series(np.nan, index=inplay.index)), errors="coerce")
    la = pd.to_numeric(inplay.get("Angle", pd.Series(np.nan, index=inplay.index)), errors="coerce")

    bases = (
        int(d.get("PlayResult","").eq("Single").sum()) +
        2*int(d.get("PlayResult","").eq("Double").sum()) +
        3*int(d.get("PlayResult","").eq("Triple").sum()) +
        4*int(d.get("PlayResult","").eq("HomeRun").sum())
    )

    avg_exit  = float(es.mean()) if not es.empty else np.nan
    max_exit  = float(es.max()) if not es.empty else np.nan
    avg_angle = float(la.mean()) if not la.empty else np.nan
    ba  = (hits / ab) if ab else 0.0
    obp = ((hits + walks + hbp) / pa) if pa else 0.0
    slg = (bases / ab) if ab else 0.0
    ops = obp + slg
    hard = (int((es >= 95).sum()) / len(inplay) * 100.0) if len(inplay) else 0.0

    out = pd.DataFrame([{
        "Avg Exit Vel": round(avg_exit,1) if not np.isnan(avg_exit) else "—",
        "Max Exit Vel": round(max_exit,1) if not np.isnan(max_exit) else "—",
        "Avg Angle":    round(avg_angle,1) if not np.isnan(avg_angle) else "—",
        "Hits":         hits,
        "SO":           so,
        "AVG":          round(ba,3),
        "OBP":          round(obp,3),
        "SLG":          round(slg,3),
        "OPS":          round(ops,3),
        "HardHit %":    round(hard,1),
        "K %":          round((so/pa)*100.0,1) if pa else 0.0,
        "BB %":         round((walks/pa)*100.0,1) if pa else 0.0,
    }])

    return out, pa, ab

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD HITTER REPORT (single-game, post-game) — FIXED PANEL SIZE
# ──────────────────────────────────────────────────────────────────────────────
def hitter_standard_report(df_date: pd.DataFrame, batter: str, ncols=3):
    bdf = df_date[df_date["Batter"] == batter]
    required_cols = ["GameID","Inning","Top/Bottom","PAofInning"]
    missing = [c for c in required_cols if c not in bdf.columns]
    if missing:
        st.error(f"Missing columns for hitter report: {missing}")
        return None

    # Group by plate appearance
    pa = list(bdf.groupby(required_cols))
    n_pa = len(pa)
    nrows = max(1, math.ceil(n_pa / ncols))

    # Build descriptions (left column)
    descs = []
    for _, padf in pa:
        lines = []
        for _, p in padf.iterrows():
            vel = pd.to_numeric(p.get("EffectiveVelo", np.nan), errors="coerce")
            vel_str = f"{vel:.1f}" if pd.notna(vel) else "—"
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType} {vel_str} MPH / {p.PitchCall}")
        ip = padf[padf.get("PitchCall","") == "InPlay"]
        if not ip.empty:
            last = ip.iloc[-1]
            res = last.get("PlayResult", "InPlay") or "InPlay"
            es  = pd.to_numeric(last.get("ExitSpeed", np.nan), errors="coerce")
            if pd.notna(es):
                res += f" ({es:.1f} MPH)"
            lines.append(f"▶ PA Result: {res}")
        else:
            balls = (padf.get("PitchCall","") == "BallCalled").sum()
            strikes = padf.get("PitchCall","").isin(["StrikeCalled","StrikeSwinging"]).sum()
            if balls >= 4:
                lines.append("▶ PA Result: Walk")
            elif strikes >= 3:
                lines.append("▶ PA Result: Strikeout")
        descs.append(lines)

    # ── FIXED per-panel sizing so zones NEVER change size ─────────────────────
    PANEL_W_IN = 3.8   # width of each strikezone panel (inches)
    PANEL_H_IN = 3.8   # height of each strikezone panel (inches)
    DESC_W_IN  = 3.0   # description column width (inches)

    # Figure size scales with number of rows, keeping each panel constant size
    fig_w = DESC_W_IN + ncols * PANEL_W_IN + 1.0  # + small margin
    fig_h = nrows * PANEL_H_IN + 2.0              # + small top/bottom margin
    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec uses proportional ratios that match our inch targets
    width_ratios  = [DESC_W_IN] + [PANEL_W_IN]*ncols
    height_ratios = [PANEL_H_IN]*nrows
    gs = GridSpec(nrows, ncols+1, width_ratios=width_ratios, height_ratios=height_ratios,
                  wspace=0.10, hspace=0.20)

    # Optional logo in the figure
    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88, 0.88, 0.12, 0.12], anchor="NE")
        axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis("off")

    # Title & mini summary
    game_date = df_date["Date"].iloc[0] if not df_date.empty else None
    fig.suptitle(f"{format_name(batter)} Hitter Report for {format_date_long(game_date)}",
                 fontsize=16, x=0.55, y=1.0, fontweight="bold")

    gd = pd.concat([grp for _, grp in pa]) if pa else pd.DataFrame()
    whiffs   = (gd.get("PitchCall","") == "StrikeSwinging").sum() if not gd.empty else 0
    hardhits = (pd.to_numeric(gd.get("ExitSpeed", pd.Series(dtype=float)), errors="coerce") > 95).sum() if not gd.empty else 0
    chases   = 0
    if not gd.empty:
        pls = pd.to_numeric(gd.get("PlateLocSide", pd.Series(np.nan, index=gd.index)), errors="coerce")
        plh = pd.to_numeric(gd.get("PlateLocHeight", pd.Series(np.nan, index=gd.index)), errors="coerce")
        chases = ((gd.get("PitchCall","") == "StrikeSwinging") &
                  ((pls < -0.83) | (pls > 0.83) | (plh < 1.5) | (plh > 3.5))).sum()
    fig.text(0.55, 0.965, f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {int(chases)}",
             ha="center", va="top", fontsize=12)

    # Plot each PA panel (fixed axes limits so the box size is constant)
    for idx, ((_, inn, tb, _), padf) in enumerate(pa):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col+1])
        draw_strikezone(ax)
        hand = padf.get("PitcherThrows","R").iloc[0]
        hand_label = "LHP" if str(hand).upper().startswith("L") else "RHP"
        pitcher = padf.get("Pitcher","").iloc[0]
        for _, p in padf.iterrows():
            mk = SHAPE_MAP.get(p.get("AutoPitchType",""), "o")
            clr = COLOR_MAP.get(p.get("PitchCall",""), "black")
            sz = 200 if p.get("AutoPitchType","") == "Slider" else 150
            ax.scatter(p.get("PlateLocSide", np.nan),
                       p.get("PlateLocHeight", np.nan),
                       marker=mk, c=clr, s=sz, edgecolor="white", linewidth=1, zorder=2)
            yoff = -0.05 if p.get("AutoPitchType","") == "Slider" else 0.0
            ax.text(p.get("PlateLocSide", np.nan),
                    p.get("PlateLocHeight", np.nan) + yoff,
                    str(int(p.get("PitchofPA", 0))), ha="center", va="center",
                    fontsize=6, fontweight="bold", zorder=3)
        # CONSTANT limits & aspect -> constant box size
        ax.set_xlim(-3, 3); ax.set_ylim(0, 5); ax.set_aspect("equal", "box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}", fontsize=10, fontweight="bold")
        ax.text(0.5, 0.1, f"vs {pitcher} ({hand_label})", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, style="italic")

    # Description column
    axd = fig.add_subplot(gs[:, 0]); axd.axis("off")
    y0 = 1.0
    dy = 1.0 / (max(1, n_pa) * 5.0)
    for i, lines in enumerate(descs, 1):
        axd.hlines(y0 - dy*0.1, 0, 1, transform=axd.transAxes, color="black", lw=1)
        axd.text(0.02, y0, f"PA {i}", fontsize=6, fontweight="bold", transform=axd.transAxes)
        yln = y0 - dy
        for ln in lines:
            axd.text(0.02, yln, ln, fontsize=6, transform=axd.transAxes); yln -= dy
        y0 = yln - dy*0.05

    # Legends
    res_handles = [Line2D([0],[0], marker="o", color="w", label=k,
                          markerfacecolor=v, ms=10, markeredgecolor="k")
                   for k, v in COLOR_MAP.items()]
    fig.legend(res_handles, [h.get_label() for h in res_handles],
               title="Result", loc="lower right", bbox_to_anchor=(0.90, 0.02))

    pitch_handles = [Line2D([0],[0], marker=m, color="w", label=k,
                            markerfacecolor="gray", ms=10, markeredgecolor="k")
                     for k, m in SHAPE_MAP.items()]
    fig.legend(pitch_handles, [h.get_label() for h in pitch_handles],
               title="Pitches", loc="lower right", bbox_to_anchor=(0.98, 0.02))

    plt.tight_layout(rect=[0.12, 0.05, 1, 0.90])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# HITTER HEATMAPS (same single-game filter)
# ──────────────────────────────────────────────────────────────────────────────
def _plot_conditional_panel(ax, title, sub_df: pd.DataFrame):
    x_min, x_max, y_min, y_max = get_view_bounds()
    draw_strikezone(ax)
    x = pd.to_numeric(sub_df.get("PlateLocSide", pd.Series(dtype=float)), errors="coerce").to_numpy()
    y = pd.to_numeric(sub_df.get("PlateLocHeight", pd.Series(dtype=float)), errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 10:
        ax.plot(x, y, "o", color="gray", alpha=0.8, markersize=6)
    else:
        xi = np.linspace(x_min, x_max, 200)
        yi = np.linspace(y_min, y_max, 200)
        xi_m, yi_m = np.meshgrid(xi, yi)
        zi = compute_density_hitter(x, y, xi_m, yi_m)
        ax.imshow(zi, origin="lower", extent=[x_min, x_max, y_min, y_max],
                  aspect="equal", cmap=CUSTOM_CMAP)
        draw_strikezone(ax)

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.set_title(title, fontsize=10, pad=6, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

def hitter_heatmaps(df_date: pd.DataFrame, batter: str, hand_choice="Both"):
    df_b = df_date[df_date["Batter"] == batter].copy()
    if df_b.empty:
        st.warning("No Nebraska hitter data for that selection.")
        return None

    df_b["iscontact"] = df_b.get("PitchCall","").isin(["InPlay","FoulBallNotFieldable","FoulBallFieldable"])
    df_b["iswhiff"]   = df_b.get("PitchCall","").eq("StrikeSwinging")
    df_b["is95plus"]  = pd.to_numeric(df_b.get("ExitSpeed", pd.Series(np.nan)), errors="coerce").ge(95.0)

    if hand_choice in ("LHH","RHH"):
        throws = "Left" if hand_choice == "LHH" else "Right"
        df_b = df_b[df_b.get("PitcherThrows","") == throws]

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

    sub_contact_l = df_b[df_b["iscontact"] & (df_b.get("PitcherThrows","") == "Left")]
    sub_contact_r = df_b[df_b["iscontact"] & (df_b.get("PitcherThrows","") == "Right")]
    ax1 = fig.add_subplot(gs[0, 0]); _plot_conditional_panel(ax1, "Contact vs LHP", sub_contact_l)
    ax2 = fig.add_subplot(gs[0, 2]); _plot_conditional_panel(ax2, "Contact vs RHP", sub_contact_r)

    sub_wh_l = df_b[df_b["iswhiff"] & (df_b.get("PitcherThrows","") == "Left")]
    sub_wh_r = df_b[df_b["iswhiff"] & (df_b.get("PitcherThrows","") == "Right")]
    ax3 = fig.add_subplot(gs[0, 3]); _plot_conditional_panel(ax3, "Whiffs vs LHP", sub_wh_l)
    ax4 = fig.add_subplot(gs[0, 5]); _plot_conditional_panel(ax4, "Whiffs vs RHP", sub_wh_r)

    sub_95_l = df_b[df_b["is95plus"] & (df_b.get("PitcherThrows","") == "Left")]
    sub_95_r = df_b[df_b["is95plus"] & (df_b.get("PitcherThrows","") == "Right")]
    ax5 = fig.add_subplot(gs[0, 6]); _plot_conditional_panel(ax5, "Exit ≥95 vs LHP", sub_95_l)
    ax6 = fig.add_subplot(gs[0, 8]); _plot_conditional_panel(ax6, "Exit ≥95 vs RHP", sub_95_r)

    formatted = format_name(batter)
    fig.suptitle(formatted, fontsize=22, x=0.5, y=0.87)
    plt.tight_layout(rect=[0, 0, 1, 0.78])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv_norm(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    if "Date" in df.columns:
        df["Date"] = ensure_date(df["Date"])
    else:
        df["Date"] = pd.NaT
    return df

if not os.path.exists(DATA_PATH):
    st.error(f"Data not found at {DATA_PATH}")
    st.stop()

df_all = load_csv_norm(DATA_PATH)

# Nebraska hitters only
df_neb_hit = df_all[df_all.get("BatterTeam","") == "NEB"].copy()
if df_neb_hit.empty:
    st.warning("No Nebraska hitter rows in the dataset.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# TOP CONTROLS (single-game, post-game only)
# ──────────────────────────────────────────────────────────────────────────────
dates = sorted([d for d in df_neb_hit["Date"].dropna().unique().tolist() if pd.notna(d)])
col1, col2, col3 = st.columns([1, 2, 2])

selected_date = col1.selectbox(
    "Game Date",
    options=dates,
    format_func=lambda d: format_date_long(d),
    index=len(dates)-1 if dates else 0
)

bdf_date = df_neb_hit[df_neb_hit["Date"] == selected_date]

batters = sorted(bdf_date.get("Batter", pd.Series(dtype=object)).dropna().unique().tolist())
batter = col2.selectbox("Batter", options=batters, format_func=format_name)

hand_choice = col3.radio("Pitcher Hand Filter (Heatmaps)", options=["Both","LHH","RHH"], horizontal=True)

tabs = st.tabs(["Standard", "Heatmaps"])

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD TAB
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    if not batter:
        st.info("Select a batter.")
    else:
        fig_std = hitter_standard_report(bdf_date, batter, ncols=3)
        if fig_std:
            st.pyplot(fig_std)

        bdf_player = bdf_date[bdf_date["Batter"] == batter].copy()
        bb_df   = create_batted_ball_profile(bdf_player)
        pd_df   = create_plate_discipline_profile(bdf_player)
        stats_df, pa_cnt, ab_cnt = create_batting_stats_profile(bdf_player)

        st.markdown("### Batted Ball Profile")
        st.table(bb_df)

        st.markdown("---")
        st.markdown("### Plate Discipline Profile")
        st.table(pd_df)

        st.markdown("---")
        st.markdown("### Batting Statistics")
        st.table(stats_df)

# ──────────────────────────────────────────────────────────────────────────────
# HEATMAPS TAB
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    if not batter:
        st.info("Select a batter.")
    else:
        fig_hm = hitter_heatmaps(bdf_date, batter, hand_choice=hand_choice)
        if fig_hm:
            st.pyplot(fig_hm)
