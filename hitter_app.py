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

# ───────────────────────── CONFIG / PATHS ─────────────────────────
st.set_page_config(layout="wide", page_title="Nebraska Hitter Reports")

DATA_PATH   = "B10C25_hitter_app_columns.csv"  # streamlit-columns dataset
LOGO_PATH   = "Nebraska-Cornhuskers-Logo.png"
BANNER_PATH = "NebraskaChampions.jpg"  # Nebraska banner image

# ───────────────────────── BANNER ─────────────────────────
def show_banner():
    if os.path.exists(BANNER_PATH):
        st.image(BANNER_PATH, use_container_width=True)
    else:
        st.markdown("<h1 style='margin:0'>Nebraska Baseball</h1>", unsafe_allow_html=True)

show_banner()
st.markdown("")  # spacer

# ───────────────── CUSTOM COLORMAP & STRIKE ZONE ─────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
    N=256,
)

def get_zone_bounds():
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
    ax.add_patch(Rectangle((sz_left, sz_bottom), sz_width, sz_height,
                           fill=False, linewidth=2, linestyle="-", color="black"))
    for frac in (1/3, 2/3):
        ax.vlines(sz_left + sz_width * frac, sz_bottom, sz_bottom + sz_height,
                  colors="gray", linestyles="--", linewidth=1)
        ax.hlines(sz_bottom + sz_height * frac, sz_left, sz_left + sz_width,
                  colors="gray", linestyles="--", linewidth=1)

# ───────────────────────── LOAD DATA ─────────────────────────
@st.cache_data(show_spinner=True)
def load_csv_norm(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        rel = os.path.basename(path)
        if os.path.exists(rel):
            path = rel
        else:
            raise FileNotFoundError(f"CSV not found at {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    df["Date"] = pd.to_datetime(df.get("Date", pd.NaT), errors="coerce").dt.date
    return df

# ───────────── SHAPES & COLORS FOR STANDARD REPORT ─────────────
SHAPE_MAP = {
    "Fastball":  "o",
    "Curveball": "s",
    "Slider":    "^",
    "Changeup":  "D",
}
RESULT_COLORS = {
    "StrikeCalled":         "#CCCC00",
    "BallCalled":           "green",
    "FoulBallNotFieldable": "tan",
    "FoulBallFieldable":    "tan",
    "InPlay":               "#6699CC",
    "StrikeSwinging":       "red",
    "HitByPitch":           "lime",
}
PITCH_COLORS = {
    "Four-Seam": "#E60026",
    "Sinker": "#FF9300",
    "Cutter": "#800080",
    "Changeup": "#008000",
    "Curveball": "#0033CC",
    "Slider": "#CCCC00",
    "Splitter": "#00CCCC",
    "Knuckle Curve": "#000000",
    "Screwball": "#CC0066",
    "Eephus": "#666666",
}
def pitch_color(ptype: str) -> str:
    return PITCH_COLORS.get(str(ptype), "gray")

def format_name(name: str) -> str:
    if isinstance(name, str) and "," in name:
        last, first = [s.strip() for s in name.split(",", 1)]
        return f"{first} {last}"
    return str(name)

def compute_density_hitter(x, y, xi_m, yi_m):
    coords = np.vstack([x, y])
    mask = np.isfinite(coords).all(axis=0)
    if mask.sum() <= 1:
        return np.zeros(xi_m.shape)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(coords[:, mask])
        return kde(np.vstack([xi_m.ravel(), yi_m.ravel()])).reshape(xi_m.shape)
    except Exception:
        return np.zeros(xi_m.shape)

# ───────────── STANDARD (POST-GAME) HITTER REPORT ─────────────
def create_hitter_report(df: pd.DataFrame, batter: str, ncols: int = 3):
    bdf = df[df["Batter"] == batter].copy()
    pa_groups = list(bdf.groupby(["GameID", "Inning", "Top/Bottom", "PAofInning"]))
    n_pa = len(pa_groups)
    nrows = max(1, math.ceil(n_pa / ncols))

    # Build textual descriptions for each PA
    descs = []
    for _, pa_df in pa_groups:
        lines = []
        for _, p in pa_df.iterrows():
            eff = pd.to_numeric(p.get("EffectiveVelo", np.nan), errors="coerce")
            eff_s = f"{eff:.1f}" if np.isfinite(eff) else "—"
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType} {eff_s} MPH / {p.PitchCall}")

        inplay = pa_df[pa_df["PitchCall"] == "InPlay"]
        if not inplay.empty:
            last = inplay.iloc[-1]
            res = last.PlayResult if pd.notna(last.PlayResult) else "InPlay"
            ex = pd.to_numeric(last.get("ExitSpeed", np.nan), errors="coerce")
            if np.isfinite(ex):
                res += f" ({ex:.1f} MPH)"
            lines.append(f"▶ PA Result: {res}")
        else:
            balls = (pa_df["PitchCall"] == "BallCalled").sum()
            strikes = pa_df["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).sum()
            if balls >= 4:
                lines.append("▶ PA Result: Walk")
            elif strikes >= 3:
                lines.append("▶ PA Result: Strikeout")
        descs.append(lines)

    fig = plt.figure(figsize=(3 + 4 * ncols + 1, 4 * nrows))
    gs = GridSpec(nrows, ncols + 1, width_ratios=[0.8] + [1] * ncols, wspace=0.1)

    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88, 0.88, 0.12, 0.12], anchor="NE", zorder=5)
        axl.imshow(mpimg.imread(LOGO_PATH))
        axl.axis("off")

    game_date = None
    if not bdf.empty and "Date" in bdf.columns:
        dates = pd.unique(bdf["Date"])
        if len(dates) >= 1:
            game_date = str(dates[0])
    if game_date:
        fig.suptitle(f"{format_name(batter)} — Hitter Report ({game_date})",
                     fontsize=16, x=0.55, y=1.0, fontweight="bold")

    gd = pd.concat([grp for _, grp in pa_groups]) if pa_groups else pd.DataFrame()
    whiffs = int((gd["PitchCall"] == "StrikeSwinging").sum()) if not gd.empty else 0
    hard   = int((pd.to_numeric(gd.get("ExitSpeed", np.nan), errors="coerce") > 95).sum()) if not gd.empty else 0
    chases = 0
    if not gd.empty:
        x = pd.to_numeric(gd.get("PlateLocSide", np.nan), errors="coerce")
        y = pd.to_numeric(gd.get("PlateLocHeight", np.nan), errors="coerce")
        chases = int(((gd["PitchCall"] == "StrikeSwinging") &
                     ((x < -0.83) | (x > 0.83) | (y < 1.5) | (y > 3.5))).sum())
    fig.text(0.55, 0.965, f"Whiffs: {whiffs}   Hard Hits: {hard}   Chases: {chases}",
             ha="center", va="top", fontsize=12)

    for idx, ((_, inn, tb, _), pa_df) in enumerate(pa_groups):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col + 1])
        draw_strikezone(ax)
        throws = str(pa_df["PitcherThrows"].iloc[0]) if not pa_df.empty else "Right"
        hand_label = "LHP" if throws.upper().startswith("L") else "RHP"
        pitcher = format_name(str(pa_df["Pitcher"].iloc[0])) if not pa_df.empty else "—"

        for _, p in pa_df.iterrows():
            mk  = SHAPE_MAP.get(p.AutoPitchType, "o")
            clr = RESULT_COLORS.get(p.PitchCall, "black")
            sz  = 200 if p.AutoPitchType == "Slider" else 150
            x = pd.to_numeric(p.get("PlateLocSide", np.nan), errors="coerce")
            y = pd.to_numeric(p.get("PlateLocHeight", np.nan), errors="coerce")
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(x, y, marker=mk, c=clr, s=sz, edgecolor="white", linewidth=1, zorder=2)
            yoff = -0.05 if p.AutoPitchType == "Slider" else 0
            ax.text(x, y + yoff, str(int(p.PitchofPA)), ha="center", va="center",
                    fontsize=6, fontweight="bold", zorder=3)
        ax.set_xlim(-3, 3); ax.set_ylim(0, 5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}", fontsize=10, fontweight="bold")
        ax.text(0.5, 0.08, f"vs {pitcher} ({hand_label})", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, style="italic")

    axd = fig.add_subplot(gs[:, 0])
    axd.axis("off")
    denom = max(1, n_pa * 5.0)
    y0 = 1.0; dy = 1.0 / denom
    for i, lines in enumerate(descs, 1):
        axd.hlines(y0 - dy * 0.1, 0, 1, transform=axd.transAxes, color="black", linewidth=1)
        axd.text(0.02, y0, f"PA {i}", fontsize=6, fontweight="bold", transform=axd.transAxes)
        yln = y0 - dy
        for ln in lines:
            axd.text(0.02, yln, ln, fontsize=6, transform=axd.transAxes)
            yln -= dy
        y0 = yln - dy * 0.05

    # Legends
    res_handles = [Line2D([0],[0], marker="o", color="w", label=k,
                          markerfacecolor=v, markersize=10, markeredgecolor="k")
                   for k, v in {
                       "StrikeCalled": "#CCCC00", "BallCalled": "green",
                       "FoulBallNotFieldable": "tan", "InPlay": "#6699CC",
                       "StrikeSwinging": "red", "HitByPitch": "lime"
                   }.items()]
    fig.legend(res_handles, [h.get_label() for h in res_handles],
               title="Result", loc="lower right", bbox_to_anchor=(0.90, 0.02))

    pitch_handles = [Line2D([0],[0], marker=m, color="w", label=k,
                             markerfacecolor="gray", markersize=10, markeredgecolor="k")
                     for k, m in {"Fastball": "o", "Curveball": "s", "Slider": "^", "Changeup": "D"}.items()]
    fig.legend(pitch_handles, [h.get_label() for h in pitch_handles],
               title="Pitches", loc="lower right", bbox_to_anchor=(0.98, 0.02))

    plt.tight_layout(rect=[0.12, 0.04, 1, 0.88])
    return fig

# ───────────── HITTER HEATMAPS ─────────────
def _plot_conditional_panel(ax, sub: pd.DataFrame, title: str):
    x_min, x_max, y_min, y_max = get_view_bounds()
    draw_strikezone(ax)
    x = pd.to_numeric(sub.get("PlateLocSide", pd.Series(dtype=float)), errors="coerce").to_numpy()
    y = pd.to_numeric(sub.get("PlateLocHeight", pd.Series(dtype=float)), errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 10:
        for _, r in sub[mask].iterrows():
            ax.plot(r["PlateLocSide"], r["PlateLocHeight"], "o",
                    color=pitch_color(r.get("AutoPitchType", "")), alpha=0.8, markersize=6)
    else:
        xi = np.linspace(x_min, x_max, 200)
        yi = np.linspace(y_min, y_max, 200)
        xi_m, yi_m = np.meshgrid(xi, yi)
        zi = compute_density_hitter(x, y, xi_m, yi_m)
        ax.imshow(zi, origin="lower", extent=[x_min, x_max, y_min, y_max],
                  aspect="equal", cmap=custom_cmap)
        draw_strikezone(ax)

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect("equal", "box")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xticks([]); ax.set_yticks([])

def hitter_heatmaps(df_batter: pd.DataFrame, batter_name: str, hand_choice: str = "Both"):
    if df_batter.empty:
        return None
    df_b = df_batter.copy()
    df_b["iscontact"] = df_b["PitchCall"].isin(["InPlay","FoulBallNotFieldable","FoulBallFieldable"])
    df_b["iswhiff"]   = df_b["PitchCall"].eq("StrikeSwinging")
    df_b["is95plus"]  = pd.to_numeric(df_b.get("ExitSpeed", np.nan), errors="coerce") >= 95

    if hand_choice in {"LHP", "RHP"}:
        want = "Left" if hand_choice == "LHP" else "Right"
        df_b = df_b[df_b["PitcherThrows"].astype(str) == want]

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

    sub_contact_l = df_b[df_b["iscontact"] & (df_b["PitcherThrows"]=="Left")]
    sub_contact_r = df_b[df_b["iscontact"] & (df_b["PitcherThrows"]=="Right")]
    ax1 = fig.add_subplot(gs[0, 0]); _plot_conditional_panel(ax1, sub_contact_l, "Contact vs LHP")
    ax2 = fig.add_subplot(gs[0, 2]); _plot_conditional_panel(ax2, sub_contact_r, "Contact vs RHP")

    sub_wh_l = df_b[df_b["iswhiff"] & (df_b["PitcherThrows"]=="Left")]
    sub_wh_r = df_b[df_b["iswhiff"] & (df_b["PitcherThrows"]=="Right")]
    ax3 = fig.add_subplot(gs[0, 3]); _plot_conditional_panel(ax3, sub_wh_l, "Whiffs vs LHP")
    ax4 = fig.add_subplot(gs[0, 5]); _plot_conditional_panel(ax4, sub_wh_r, "Whiffs vs RHP")

    sub_95_l = df_b[df_b["is95plus"] & (df_b["PitcherThrows"]=="Left")]
    sub_95_r = df_b[df_b["is95plus"] & (df_b["PitcherThrows"]=="Right")]
    ax5 = fig.add_subplot(gs[0, 6]); _plot_conditional_panel(ax5, sub_95_l, "Exit ≥95 vs LHP")
    ax6 = fig.add_subplot(gs[0, 8]); _plot_conditional_panel(ax6, sub_95_r, "Exit ≥95 vs RHP")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

# ───────────── NEW: BATTED BALL / PLATE DISCIPLINE / BATTING STATS ─────────────
def _norm_side(s):
    s = str(s).strip().upper()
    return "L" if s.startswith("L") else ("R" if s.startswith("R") else "")

def assign_spray_category(row):
    ang  = pd.to_numeric(row.get("Bearing", np.nan), errors="coerce")
    side = _norm_side(row.get("BatterSide", ""))
    if not np.isfinite(ang) or side == "":
        return np.nan
    if -15 <= ang <= 15:
        return "Straight"
    if ang < -15:
        return "Pull" if side == "R" else "Opposite"
    return "Opposite" if side == "R" else "Pull"

def create_batted_ball_profile(df):
    inplay = df[df["PitchCall"] == "InPlay"].copy()
    if inplay.empty:
        return pd.DataFrame([{
            "Ground ball %": 0.0, "Fly ball %": 0.0, "Line drive %": 0.0, "Popup %": 0.0,
            "Pull %": 0.0, "Straight %": 0.0, "Opposite %": 0.0
        }])
    # normalize TaggedHitType casing
    th = inplay.get("TaggedHitType")
    if th is not None:
        inplay["TaggedHitType"] = th.astype(str).str.strip()
    inplay["spray_cat"] = inplay.apply(assign_spray_category, axis=1)

    def pct(mask): 
        denom = len(inplay)
        return round((mask.sum() / denom * 100) if denom else 0.0, 1)

    return pd.DataFrame([{
        "Ground ball %": pct(inplay["TaggedHitType"].eq("GroundBall")),
        "Fly ball %":    pct(inplay["TaggedHitType"].eq("FlyBall")),
        "Line drive %":  pct(inplay["TaggedHitType"].eq("LineDrive")),
        "Popup %":       pct(inplay["TaggedHitType"].eq("Popup")),
        "Pull %":        pct(inplay["spray_cat"].eq("Pull")),
        "Straight %":    pct(inplay["spray_cat"].eq("Straight")),
        "Opposite %":    pct(inplay["spray_cat"].eq("Opposite")),
    }])

def create_plate_discipline_profile(df):
    d = df.copy()
    d["isswing"]   = d["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"])
    d["iswhiff"]   = d["PitchCall"].eq("StrikeSwinging")
    d["iscontact"] = d["PitchCall"].isin(["InPlay","FoulBallNotFieldable","FoulBallFieldable"])
    x = pd.to_numeric(d.get("PlateLocSide", np.nan), errors="coerce")
    y = pd.to_numeric(d.get("PlateLocHeight", np.nan), errors="coerce")
    d["isinzone"]  = (x.between(-0.83, 0.83)) & (y.between(1.5, 3.5))

    zp = d["isinzone"]; sw = d["isswing"]
    total = len(d)
    zone_p = int(zp.sum())
    zone_pct = round((zone_p/total*100) if total else 0.0, 1)
    zone_sw = round((sw[zp].mean()*100) if zone_p else 0.0, 1)
    zone_ct = round((d.loc[zp & d["iscontact"], "iscontact"].sum() / sw[zp].sum()*100) if sw[zp].sum() else 0.0, 1)
    chase   = round((sw[~zp].mean()*100) if (~zp).sum() else 0.0, 1)
    swing   = round((sw.mean()*100) if total else 0.0, 1)
    whiff   = round((d["iswhiff"].sum()/sw.sum()*100) if sw.sum() else 0.0, 1)

    return pd.DataFrame([{
        "Zone Pitches": zone_p,
        "Zone %": zone_pct,
        "Zone Swing %": zone_sw,
        "Zone Contact %": zone_ct,
        "Chase %": chase,
        "Swing %": swing,
        "Whiff %": whiff,
    }])

def create_batting_stats_profile(df):
    d = df.copy()
    d["PA"]        = d.get("PitchofPA", 0) == 1
    d["Hit"]       = (d["PitchCall"].eq("InPlay") & d["PlayResult"].isin(["Single","Double","Triple","HomeRun"]))
    d["StrikeOut"] = d["KorBB"].eq("Strikeout")
    d["Walk"]      = d["KorBB"].eq("Walk")
    d["HBP"]       = d["PitchCall"].eq("HitByPitch")
    d["BBOut"]     = (d["PitchCall"].eq("InPlay") & d["PlayResult"].eq("Out"))
    d["FC"]        = d["PlayResult"].eq("FieldersChoice")
    d["Error"]     = d["PlayResult"].eq("Error")

    hits   = int(d["Hit"].sum())
    so     = int(d["StrikeOut"].sum())
    bbouts = int(d["BBOut"].sum())
    fc     = int(d["FC"].sum())
    err    = int(d["Error"].sum())
    walks  = int(d["Walk"].sum())
    hbp    = int(d["HBP"].sum())
    pa     = int(d["PA"].sum())

    ab = hits + so + bbouts + fc + err

    singles = int(d["PlayResult"].eq("Single").sum())
    doubles = int(d["PlayResult"].eq("Double").sum())
    triples = int(d["PlayResult"].eq("Triple").sum())
    hrs     = int(d["PlayResult"].eq("HomeRun").sum())
    total_bases = singles + 2*doubles + 3*triples + 4*hrs

    inplay = d[d["PitchCall"]=="InPlay"].copy()
    exit_speed = pd.to_numeric(inplay.get("ExitSpeed", np.nan), errors="coerce")
    angle      = pd.to_numeric(inplay.get("Angle", np.nan), errors="coerce")
    avg_exit   = float(exit_speed.mean()) if len(exit_speed) else np.nan
    max_exit   = float(exit_speed.max()) if len(exit_speed) else np.nan
    avg_angle  = float(angle.mean()) if len(angle) else np.nan
    hard_pct   = round(((exit_speed >= 95).mean()*100) if exit_speed.notna().any() else 0.0, 1)

    ba  = (hits / ab) if ab else 0.0
    obp = ((hits + walks + hbp) / pa) if pa else 0.0
    slg = (total_bases / ab) if ab else 0.0
    ops = obp + slg

    stats = pd.DataFrame([{
        "Avg Exit Vel": round(avg_exit, 1) if np.isfinite(avg_exit) else 0.0,
        "Max Exit Vel": round(max_exit, 1) if np.isfinite(max_exit) else 0.0,
        "Avg Angle":    round(avg_angle, 1) if np.isfinite(avg_angle) else 0.0,
        "Hits":         hits,
        "SO":           so,
        "AVG":          round(ba, 3),
        "OBP":          round(obp, 3),
        "SLG":          round(slg, 3),
        "OPS":          round(ops, 3),
        "HardHit %":    hard_pct,
        "K %":          round((so/pa*100), 1) if pa else 0.0,
        "BB %":         round((walks/pa*100), 1) if pa else 0.0,
    }])
    return stats, pa, ab, len(d)

# ───────────────────────── APP ─────────────────────────
try:
    df_all = load_csv_norm(DATA_PATH)
except Exception as e:
    st.error(f"Failed to read CSV at {DATA_PATH}: {e}")
    st.stop()

# Nebraska hitters only
df_neb_hit = df_all[df_all.get("BatterTeam","") == "NEB"].copy()
if df_neb_hit.empty:
    st.warning("No Nebraska hitter rows found in the dataset.")
    st.stop()

# Single-game selection (post-game)
date_options = sorted(pd.unique(df_neb_hit["Date"]))
colA, colB, colC = st.columns([1.2, 1.4, 2])
sel_date = colA.selectbox("Game Date", options=date_options, index=len(date_options)-1 if date_options else 0)
df_date = df_neb_hit[df_neb_hit["Date"] == sel_date]

batters = sorted(df_date["Batter"].dropna().unique().tolist())
if not batters:
    st.info("No Nebraska batters found for that date.")
    st.stop()
batter = colB.selectbox("Batter", options=batters)

# Tabs: Standard (single game) and Heatmaps
tab_std, tab_hm = st.tabs(["Standard", "Heatmaps"])

with tab_std:
    st.markdown(f"**{format_name(batter)} — {sel_date}**")
    # Main post-game report figure
    fig_std = create_hitter_report(df_date, batter, ncols=3)
    if fig_std:
        st.pyplot(fig=fig_std)

    # ── NEW: Metrics tables (no visuals, just calculations)
    bdf = df_date[df_date["Batter"] == batter].copy()
    st.markdown("### Batted Ball, Plate Discipline & Batting Stats")
    bb_df   = create_batted_ball_profile(bdf)
    pd_df   = create_plate_discipline_profile(bdf)
    stats_df, pa_cnt, ab_cnt, pitch_cnt = create_batting_stats_profile(bdf)

    st.caption(f"Pitches: {pitch_cnt} • PA: {pa_cnt} • AB: {ab_cnt}")
    c1, c2, c3 = st.columns([1,1,1.2])
    with c1:
        st.markdown("**Batted Ball Profile**")
        st.table(bb_df)
    with c2:
        st.markdown("**Plate Discipline Profile**")
        st.table(pd_df)
    with c3:
        st.markdown("**Batting Statistics**")
        st.table(stats_df)

with tab_hm:
    hand_choice = st.radio("Pitcher Hand", options=["Both","LHP","RHP"], index=0, horizontal=True)
    bdf_hm = df_date[df_date["Batter"] == batter].copy()
    fig_hm = hitter_heatmaps(bdf_hm, batter, hand_choice=hand_choice)
    if fig_hm:
        st.pyplot(fig=fig_hm)
