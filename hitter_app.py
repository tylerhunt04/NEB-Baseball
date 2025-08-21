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
# PAGE CONFIG (wide mode OFF)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="centered", page_title="Nebraska Hitter Reports")

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "B10C25_hitter_app_columns.csv"
BANNER_CANDIDATES = ["NebraskaChampions.jpg"]

# ──────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def load_banner():
    for p in BANNER_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"

def format_date_long(d) -> str:
    if d is None or pd.isna(d): 
        return ""
    d = pd.to_datetime(d).date()
    return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"

def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["DateOnly"] = df["Date"].dt.date
    else:
        df["Date"] = pd.NaT
        df["DateOnly"] = pd.NaT
    return df

def baseball_avg_str(x: float) -> str:
    """Format AVG/OBP/SLG/OPS to 3 decimals; show like .382 if < 1.000."""
    if x is None or pd.isna(x):
        return "—"
    s = f"{x:.3f}"
    return s[1:] if s.startswith("0") else s  # .382 style

def pct2(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.2f}%"

def fmt1(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.1f}"

def fmt2(x: float) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.2f}"

def season_label_from_dates(dates: pd.Series) -> str:
    dts = pd.to_datetime(dates, errors="coerce").dropna()
    if dts.empty:
        return "—"
    years = sorted(dts.dt.year.unique().tolist())
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}–{years[-1]}"

def month_names_from_numbers(months: list[int]) -> str:
    if not months:
        return ""
    names = [pd.Timestamp(year=2000, month=m, day=1).strftime("%B") for m in sorted(set(months))]
    return ", ".join(names)

# ──────────────────────────────────────────────────────────────────────────────
# STRIKE ZONE / VIEW
# ──────────────────────────────────────────────────────────────────────────────
def get_zone_bounds():
    left, bottom = -0.83, 1.50
    width, height = 1.66, 2.00
    return left, bottom, width, height

def draw_strikezone(ax, left=None, bottom=None, width=None, height=None):
    L, B, W, H = get_zone_bounds()
    left   = L if left   is None else left
    bottom = B if bottom is None else bottom
    width  = W if width  is None else width
    height = H if height is None else height
    ax.add_patch(Rectangle((left, bottom), width, height, fill=False, linewidth=2, color='black'))
    dx, dy = width/3, height/3
    for i in (1, 2):
        ax.add_line(Line2D([left+i*dx]*2, [bottom, bottom+height], linestyle='--', color='gray'))
        ax.add_line(Line2D([left, left+width], [bottom+i*dy]*2, linestyle='--', color='gray'))

# For heatmaps (consistent bounds around the zone)
def get_view_bounds():
    L, B, W, H = get_zone_bounds()
    mx, my = W * 0.8, H * 0.6
    return L - mx, L + W + mx, B - my, B + H + my

# ──────────────────────────────────────────────────────────────────────────────
# HEATMAP DENSITY
# ──────────────────────────────────────────────────────────────────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
    N=256,
)

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
# STANDARD HITTER REPORT (single game)
# ──────────────────────────────────────────────────────────────────────────────
def create_hitter_report(df_date_neb: pd.DataFrame, batter: str, ncols=3):
    bdf = df_date_neb[df_date_neb["Batter"] == batter].copy()
    pa_groups = list(bdf.groupby(["GameID", "Inning", "Top/Bottom", "PAofInning"], sort=False))
    n_pa = len(pa_groups)
    nrows = max(1, math.ceil(n_pa / ncols))

    # Build PA descriptions
    descs = []
    for _, pa_df in pa_groups:
        lines = []
        for _, p in pa_df.iterrows():
            velo = p.get("EffectiveVelo", np.nan)
            vel_str = fmt1(velo) if pd.notna(velo) else "—"
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType} {vel_str} MPH / {p.PitchCall}")
        inplay = pa_df[pa_df["PitchCall"] == "InPlay"]
        if not inplay.empty:
            last = inplay.iloc[-1]
            res = last.PlayResult if pd.notna(last.PlayResult) else "InPlay"
            es = last.get("ExitSpeed", np.nan)
            if pd.notna(es):
                res += f" ({fmt1(es)} MPH)"
            lines.append(f"▶ PA Result: {res}")
        else:
            balls = (pa_df["PitchCall"] == "BallCalled").sum()
            strikes = pa_df["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"]).sum()
            if balls >= 4:
                lines.append("▶ PA Result: Walk")
            elif strikes >= 3:
                lines.append("▶ PA Result: Strikeout")
        descs.append(lines)

    # Figure
    fig = plt.figure(figsize=(3 + 4*ncols + 1, 4*nrows))
    # NOTE: leave extra vertical space *beneath* the strike zones for better spacing
    gs = GridSpec(nrows, ncols + 1, width_ratios=[0.82] + [1] * ncols, wspace=0.10, hspace=0.45)

    # Title
    if pa_groups:
        dt = pa_groups[0][1]["DateOnly"].iloc[0]
        fig.suptitle(f"{batter} Hitter Report — {format_date_long(dt)}",
                     fontsize=16, x=0.55, y=1.0, fontweight='bold')

    # Summary line
    if pa_groups:
        game_df = pd.concat([grp for _, grp in pa_groups], ignore_index=True)
        whiffs = (game_df["PitchCall"] == "StrikeSwinging").sum()
        hardhits = (game_df.get("ExitSpeed", pd.Series(dtype=float)) > 95).sum()
        chases = game_df[
            (game_df["PitchCall"] == "StrikeSwinging") &
            ((game_df["PlateLocSide"] < -0.83) | (game_df["PlateLocSide"] > 0.83) |
             (game_df["PlateLocHeight"] < 1.50) | (game_df["PlateLocHeight"] > 3.50))
        ].shape[0]
        fig.text(0.55, 0.965, f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",
                 ha="center", va="top", fontsize=12)

    # Panels (fixed strike-zone dimensions)
    for idx, ((_, inn, tb, _), pa_df) in enumerate(pa_groups):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col + 1])
        draw_strikezone(ax)
        for _, p in pa_df.iterrows():
            mk = {"Fastball":"o", "Curveball":"s", "Slider":"^", "Changeup":"D"}.get(p.AutoPitchType, "o")
            clr = {"StrikeCalled":"#CCCC00", "BallCalled":"green", "FoulBallNotFieldable":"tan",
                   "InPlay":"#6699CC", "StrikeSwinging":"red", "HitByPitch":"lime"}.get(p.PitchCall, "black")
            sz = 200 if p.AutoPitchType == "Slider" else 150
            ax.scatter(p.PlateLocSide, p.PlateLocHeight, marker=mk, c=clr,
                       s=sz, edgecolor="white", linewidth=1, zorder=2)
            yoff = -0.05 if p.AutoPitchType == "Slider" else 0.0
            ax.text(p.PlateLocSide, p.PlateLocHeight + yoff, str(int(p.PitchofPA)),
                    ha="center", va="center", fontsize=6, fontweight="bold", zorder=3)
        # FIXED limits for consistent box size
        ax.set_xlim(-3, 3); ax.set_ylim(0, 5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}", fontsize=10, fontweight="bold")
        # add a small gap below each zone (label space)
        ax.text(0.5, -0.12, "", transform=ax.transAxes)

    # Description column
    axd = fig.add_subplot(gs[:, 0]); axd.axis("off")
    y0 = 1.0; dy = 1.0 / (max(1, n_pa) * 5.0)
    for i, lines in enumerate(descs, 1):
        axd.hlines(y0 - dy*0.12, 0, 1, transform=axd.transAxes, color="black", linewidth=1)
        axd.text(0.02, y0, f"PA {i}", fontsize=6, fontweight="bold", transform=axd.transAxes)
        yln = y0 - dy
        for ln in lines:
            axd.text(0.02, yln, ln, fontsize=6, transform=axd.transAxes)
            yln -= dy
        y0 = yln - dy*0.06

    # Legends (move slightly lower)
    res_handles = [
        Line2D([0],[0], marker='o', color='w', label=k,
               markerfacecolor=v, markersize=10, markeredgecolor='k')
        for k, v in {
            "StrikeCalled":"#CCCC00","BallCalled":"green","FoulBallNotFieldable":"tan",
            "InPlay":"#6699CC","StrikeSwinging":"red","HitByPitch":"lime"
        }.items()
    ]
    fig.legend(res_handles, [h.get_label() for h in res_handles],
               title="Result", loc="lower right", bbox_to_anchor=(0.90, 0.01))

    pitch_handles = [
        Line2D([0],[0], marker=m, color='w', label=k,
               markerfacecolor='gray', markersize=10, markeredgecolor='k')
        for k, m in {"Fastball":"o","Curveball":"s","Slider":"^","Changeup":"D"}.items()
    ]
    fig.legend(pitch_handles, [h.get_label() for h in pitch_handles],
               title="Pitches", loc="lower right", bbox_to_anchor=(0.98, 0.01))

    plt.tight_layout(rect=[0.12, 0.05, 1, 0.88])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# BATTED BALL / PLATE DISCIPLINE / BATTING STATS (stacked tables)
# ──────────────────────────────────────────────────────────────────────────────
def assign_spray_category(row):
    ang  = row.get("Bearing", np.nan)
    side = str(row.get("BatterSide", "")).upper()
    if pd.isna(ang):
        return np.nan
    if -15 <= ang <= 15:
        return "Straight"
    if ang < -15:
        return "Pull" if side == "R" else "Opposite"
    return "Opposite" if side == "R" else "Pull"

def build_batted_ball_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Returns 1-row DataFrame with required rounding."""
    need_cols = {"TaggedHitType","PitchCall"}
    sub = df.copy()
    if not need_cols.issubset(sub.columns):
        # Graceful fallback if trimmed file
        return pd.DataFrame([{
            "Season":"—","Ground ball %":"—","Fly ball %":"—","Line drive %":"—","Popup %":"—",
            "Pull %":"—","Straight %":"—","Opposite %":"—"
        }])
    inplay = sub[sub["PitchCall"] == "InPlay"].copy()
    # spray (if possible)
    if {"Bearing","BatterSide"}.issubset(sub.columns):
        inplay["spray_cat"] = inplay.apply(assign_spray_category, axis=1)
    else:
        inplay["spray_cat"] = np.nan

    def pct(series_bool):
        if len(inplay) == 0:
            return "—"
        return f"{series_bool.mean()*100:.2f}%"

    row = {
        "Ground ball %":  pct(inplay["TaggedHitType"].eq("GroundBall")),
        "Fly ball %":     pct(inplay["TaggedHitType"].eq("FlyBall")),
        "Line drive %":   pct(inplay["TaggedHitType"].eq("LineDrive")),
        "Popup %":        pct(inplay["TaggedHitType"].eq("Popup")),
        "Pull %":         pct(inplay["spray_cat"].eq("Pull")),
        "Straight %":     pct(inplay["spray_cat"].eq("Straight")),
        "Opposite %":     pct(inplay["spray_cat"].eq("Opposite")),
    }
    return pd.DataFrame([row])

def build_plate_discipline_profile(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    if sub.empty:
        return pd.DataFrame([{
            "Zone Pitches":0, "Zone %":"—", "Zone Swing %":"—", "Zone Contact %":"—",
            "Chase %":"—", "Swing %":"—", "Whiff %":"—"
        }])
    sub["isswing"]   = sub["PitchCall"].isin(["StrikeSwinging","FoulBallNotFieldable","FoulBallFieldable","InPlay"])
    sub["iswhiff"]   = sub["PitchCall"].eq("StrikeSwinging")
    sub["iscontact"] = sub["PitchCall"].isin(["InPlay","FoulBallNotFieldable","FoulBallFieldable"])
    sub["isinzone"]  = sub["PlateLocSide"].between(-0.83, 0.83) & sub["PlateLocHeight"].between(1.50, 3.50)

    zp = sub["isinzone"]
    sw = sub["isswing"]

    zone_pct   = f"{(zp.mean()*100):.2f}%" if len(sub) else "—"
    zswing_pct = f"{(sw[zp].mean()*100):.2f}%" if zp.any() else "—"
    zcontact_pct = "—"
    if sw[zp].sum() > 0:
        zcontact_pct = f"{(sub.loc[zp & sub['iscontact'], 'iscontact'].sum() / sw[zp].sum() * 100):.2f}%"
    chase_pct  = f"{(sw[~zp].mean()*100):.2f}%" if (~zp).any() else "—"
    swing_pct  = f"{(sw.mean()*100):.2f}%"
    whiff_pct  = f"{(sub['iswhiff'].sum() / sw.sum() * 100):.2f}%" if sw.sum() else "—"

    row = {
        "Zone Pitches": int(zp.sum()),
        "Zone %":       zone_pct,
        "Zone Swing %": zswing_pct,
        "Zone Contact %": zcontact_pct,
        "Chase %":      chase_pct,
        "Swing %":      swing_pct,
        "Whiff %":      whiff_pct,
    }
    return pd.DataFrame([row])

def build_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    if sub.empty:
        return pd.DataFrame([{
            "Avg Exit Velo":"—","Max Exit Velo":"—","Avg LA":"—",
            "Hits":0,"SO":0,"AVG":"—","OBP":"—","SLG":"—","OPS":"—","HardHit %":"—","K %":"—","BB %":"—"
        }])

    sub["PA_flag"]   = sub["PitchofPA"] == 1
    sub["Hit"]       = sub["PitchCall"].eq("InPlay") & sub["PlayResult"].isin(["Single","Double","Triple","HomeRun"])
    sub["SO_flag"]   = sub["KorBB"].eq("Strikeout")
    sub["Walk"]      = sub["KorBB"].eq("Walk")
    sub["HBP"]       = sub["PitchCall"].eq("HitByPitch")
    sub["BBOut"]     = sub["PitchCall"].eq("InPlay") & sub["PlayResult"].eq("Out")
    sub["FC"]        = sub["PlayResult"].eq("FieldersChoice")
    sub["Error"]     = sub["PlayResult"].eq("Error")

    hits = int(sub["Hit"].sum())
    so   = int(sub["SO_flag"].sum())
    bbouts = int(sub["BBOut"].sum())
    fc   = int(sub["FC"].sum())
    err  = int(sub["Error"].sum())
    ab   = hits + so + bbouts + fc + err

    walks = int(sub["Walk"].sum())
    hbp   = int(sub["HBP"].sum())
    pa    = int(sub["PA_flag"].sum())

    inplay = sub[sub["PitchCall"] == "InPlay"]
    bases = (
        sub["PlayResult"].eq("Single").sum() +
        2 * sub["PlayResult"].eq("Double").sum() +
        3 * sub["PlayResult"].eq("Triple").sum() +
        4 * sub["PlayResult"].eq("HomeRun").sum()
    )

    avg_exit  = inplay["ExitSpeed"].mean() if "ExitSpeed" in inplay.columns else np.nan
    max_exit  = inplay["ExitSpeed"].max() if "ExitSpeed" in inplay.columns else np.nan
    avg_la    = inplay["Angle"].mean() if "Angle" in inplay.columns else np.nan
    hard_pct  = (inplay["ExitSpeed"] >= 95).mean()*100 if "ExitSpeed" in inplay.columns and len(inplay) else np.nan

    ba  = hits / ab if ab else np.nan
    obp = (hits + walks + hbp) / pa if pa else np.nan
    slg = bases / ab if ab else np.nan
    ops = (obp + slg) if (pd.notna(obp) and pd.notna(slg)) else np.nan

    row = {
        "Avg Exit Velo": fmt2(avg_exit) if pd.notna(avg_exit) else "—",
        "Max Exit Velo": fmt2(max_exit) if pd.notna(max_exit) else "—",
        "Avg LA":        fmt2(avg_la)   if pd.notna(avg_la)   else "—",
        "Hits":          hits,
        "SO":            so,
        "AVG":           baseball_avg_str(ba)  if pd.notna(ba)  else "—",
        "OBP":           baseball_avg_str(obp) if pd.notna(obp) else "—",
        "SLG":           baseball_avg_str(slg) if pd.notna(slg) else "—",
        "OPS":           baseball_avg_str(ops) if pd.notna(ops) else "—",
        "HardHit %":     pct2(hard_pct) if pd.notna(hard_pct) else "—",
        "K %":           pct2((so/pa*100) if pa else np.nan),
        "BB %":          pct2((walks/pa*100) if pa else np.nan),
    }
    return pd.DataFrame([row])

def style_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Husker red header, white text; no index column."""
    red = "#E60026"
    sty = (
        df.reset_index(drop=True)
          .style
          .set_table_styles([
              {"selector": "th.col_heading", "props": [("background-color", red), ("color", "white")]},
              {"selector": "th.col_heading.level0", "props": [("background-color", red), ("color", "white")]},
              {"selector": "th.row_heading", "props": [("display", "none")]},
          ])
          .hide(axis="index")
    )
    return sty

# ──────────────────────────────────────────────────────────────────────────────
# HITTER HEATMAPS (with pitcher-hand filter)
# ──────────────────────────────────────────────────────────────────────────────
def _plot_panel(ax, title, sub):
    x_min, x_max, y_min, y_max = get_view_bounds()
    draw_strikezone(ax)
    x = pd.to_numeric(sub.get("PlateLocSide", pd.Series(dtype=float)), errors="coerce").to_numpy()
    y = pd.to_numeric(sub.get("PlateLocHeight", pd.Series(dtype=float)), errors="coerce").to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(sub) < 10:
        ax.plot(x, y, "o", color="gray", alpha=0.8, markersize=6)
    else:
        xi = np.linspace(x_min, x_max, 200)
        yi = np.linspace(y_min, y_max, 200)
        xi_m, yi_m = np.meshgrid(xi, yi)
        zi = compute_density_hitter(x, y, xi_m, yi_m)
        ax.imshow(zi, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal", cmap=custom_cmap)
        draw_strikezone(ax)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect("equal", "box")
    ax.set_title(title, fontsize=10, pad=6, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

def hitter_heatmaps(df_player_filtered: pd.DataFrame, batter: str, hand_choice: str = "Both"):
    df_b = df_player_filtered.copy()

    # Apply pitcher-hand filter if requested
    if hand_choice in ("LHP", "RHP"):
        want = "Left" if hand_choice == "LHP" else "Right"
        df_b = df_b[df_b.get("PitcherThrows", "").astype(str).str.startswith(want)]

    if df_b.empty:
        st.info("No rows after applying current filters.")
        return

    # Flags
    df_b["iscontact"] = df_b["PitchCall"].isin(["InPlay","FoulBallFieldable","FoulBallNotFieldable"])
    df_b["iswhiff"]   = df_b["PitchCall"].eq("StrikeSwinging")
    df_b["is95plus"]  = df_b.get("ExitSpeed", pd.Series(dtype=float)) >= 95

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

    # Contact vs LHP/RHP
    sub_contact_l = df_b[df_b["iscontact"] & (df_b["PitcherThrows"] == "Left")]
    sub_contact_r = df_b[df_b["iscontact"] & (df_b["PitcherThrows"] == "Right")]
    ax1 = fig.add_subplot(gs[0, 0]); _plot_panel(ax1, "Contact vs LHP", sub_contact_l)
    ax2 = fig.add_subplot(gs[0, 2]); _plot_panel(ax2, "Contact vs RHP", sub_contact_r)

    # Whiffs vs LHP/RHP
    sub_wh_l = df_b[df_b["iswhiff"] & (df_b["PitcherThrows"] == "Left")]
    sub_wh_r = df_b[df_b["iswhiff"] & (df_b["PitcherThrows"] == "Right")]
    ax3 = fig.add_subplot(gs[0, 3]); _plot_panel(ax3, "Whiffs vs LHP", sub_wh_l)
    ax4 = fig.add_subplot(gs[0, 5]); _plot_panel(ax4, "Whiffs vs RHP", sub_wh_r)

    # ≥95 EV vs LHP/RHP
    sub_95_l = df_b[df_b["is95plus"] & (df_b["PitcherThrows"] == "Left")]
    sub_95_r = df_b[df_b["is95plus"] & (df_b["PitcherThrows"] == "Right")]
    ax5 = fig.add_subplot(gs[0, 6]); _plot_panel(ax5, "Exit ≥95 vs LHP", sub_95_l)
    ax6 = fig.add_subplot(gs[0, 8]); _plot_panel(ax6, "Exit ≥95 vs RHP", sub_95_r)

    fig.suptitle(batter, fontsize=22, x=0.5, y=0.87)
    plt.tight_layout(rect=[0, 0, 1, 0.78])
    st.pyplot(fig=fig)

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Data not found at: {path}")
        st.stop()
    df = pd.read_csv(path, low_memory=False)
    df = ensure_date(df)
    return df

df_all = load_data(DATA_PATH)
df_neb = df_all[df_all.get("BatterTeam", "") == "NEB"].copy()

# ──────────────────────────────────────────────────────────────────────────────
# BANNER
# ──────────────────────────────────────────────────────────────────────────────
banner_path = load_banner()
if banner_path:
    st.image(banner_path, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD POST-GAME REPORT (one game only): Date → Batter
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Standard Hitter Report")

# Date options (Nebraska only)
date_options = sorted([d for d in df_neb["DateOnly"].dropna().unique().tolist()])
sel_date = st.selectbox("Game Date", date_options, format_func=format_date_long, index=len(date_options)-1 if date_options else 0)

if sel_date:
    df_date = df_neb[df_neb["DateOnly"] == sel_date].copy()
    batters = sorted(df_date["Batter"].dropna().unique().tolist())
    batter  = st.selectbox("Batter", batters)

    if batter:
        fig_std = create_hitter_report(df_date, batter, ncols=3)
        st.pyplot(fig=fig_std)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# FILTERS for Tables & Heatmaps (apply to the selected batter across season)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Filters for Batted Ball / Plate Discipline / Batting Statistics / Heatmaps")

colM, colD, colN, colH = st.columns([1.2, 1.0, 1.0, 1.2])

# Build batter-level base DF (across the season) once batter is chosen
df_player_all = df_neb[df_neb["Batter"] == batter].copy() if sel_date and batter else df_neb.iloc[0:0].copy()

# MONTHS (from this batter's appearances)
present_months = sorted(pd.to_datetime(df_player_all["Date"], errors="coerce").dt.month.dropna().astype(int).unique().tolist())
months_sel = colM.multiselect("Months", options=present_months,
                              format_func=lambda m: pd.Timestamp(year=2000, month=m, day=1).strftime("%B"),
                              default=[])

# DAYS (limit to days present in the chosen months if months are selected)
dser = pd.to_datetime(df_player_all["Date"], errors="coerce").dropna()
if months_sel:
    dser = dser[dser.dt.month.isin(months_sel)]
present_days = sorted(dser.dt.day.unique().tolist())
days_sel = colD.multiselect("Days", options=present_days, default=[])

# LAST N GAMES
last_n = int(colN.number_input("Last N Games", min_value=0, max_value=60, value=0, step=1))

# PITCHER HAND
hand_choice = colH.radio("Pitcher Hand", options=["Both", "LHP", "RHP"], index=0, horizontal=True)

# Apply filters
df_player_filt = df_player_all.copy()
if months_sel:
    df_player_filt = df_player_filt[pd.to_datetime(df_player_filt["Date"], errors="coerce").dt.month.isin(months_sel)]
if days_sel:
    df_player_filt = df_player_filt[pd.to_datetime(df_player_filt["Date"], errors="coerce").dt.day.isin(days_sel)]

# Last N unique game dates (for this batter)
if last_n and not df_player_all.empty:
    all_dates_player = sorted(df_player_all["DateOnly"].dropna().unique().tolist())
    take_dates = set(all_dates_player[-last_n:])
    df_player_filt = df_player_filt[df_player_filt["DateOnly"].isin(take_dates)]

# Also apply hand-choice for stat tables (makes profiles consistent with heatmaps)
if hand_choice in ("LHP", "RHP"):
    want = "Left" if hand_choice == "LHP" else "Right"
    df_player_filt = df_player_filt[df_player_filt.get("PitcherThrows", "").astype(str).str.startswith(want)]

# ──────────────────────────────────────────────────────────────────────────────
# STACKED TABLES: Batted Ball, Plate Discipline, Batting Statistics
# ──────────────────────────────────────────────────────────────────────────────
if batter:
    st.markdown("### Batted Ball Profile")
    bb = build_batted_ball_profile(df_player_filt)
    # prepend Season and (optional) Month column
    bb.insert(0, "Season", season_label_from_dates(df_player_filt["Date"]))
    if months_sel:
        bb.insert(1, "Month", month_names_from_numbers(months_sel))
    st.table(style_table(bb))

    st.markdown("### Plate Discipline Profile")
    pdp = build_plate_discipline_profile(df_player_filt)
    pdp.insert(0, "Season", season_label_from_dates(df_player_filt["Date"]))
    if months_sel:
        pdp.insert(1, "Month", month_names_from_numbers(months_sel))
    st.table(style_table(pdp))

    st.markdown("### Batting Statistics")
    bst = build_batting_stats(df_player_filt)
    bst.insert(0, "Season", season_label_from_dates(df_player_filt["Date"]))
    if months_sel:
        bst.insert(1, "Month", month_names_from_numbers(months_sel))
    st.table(style_table(bst))

# ──────────────────────────────────────────────────────────────────────────────
# HITTER HEATMAPS
# ──────────────────────────────────────────────────────────────────────────────
if batter:
    st.markdown("### Hitter Heatmaps")
    hitter_heatmaps(df_player_filt, batter=batter, hand_choice=hand_choice)
