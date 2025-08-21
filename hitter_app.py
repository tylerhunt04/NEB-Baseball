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

# Adjust these if paths differ in your deployment
DATA_PATH   = "B10C25_streamlit_streamlit_columns.csv"  # streamlit-columns dataset
LOGO_PATH   = "Nebraska-Cornhuskers-Logo.png"
BANNER_PATH = "NebraskaChampions.jpg"  # Nebraska banner image

# ───────────────────────── BANNER ─────────────────────────
def show_banner():
    # If the banner image exists, show it full width at top
    if os.path.exists(BANNER_PATH):
        st.image(BANNER_PATH, use_container_width=True)
    else:
        # Fallback heading if the banner isn’t found
        st.markdown("<h1 style='margin:0'>Nebraska Baseball</h1>", unsafe_allow_html=True)

show_banner()
st.markdown("")  # small spacer under banner

# ───────────────── CUSTOM COLORMAP & STRIKE ZONE ────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
    N=256,
)

def get_zone_bounds():
    # Fixed zone so size stays identical across panels
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

# ───────────────────────── HELPERS ─────────────────────────
def format_name(name: str) -> str:
    if isinstance(name, str) and "," in name:
        last, first = [s.strip() for s in name.split(",", 1)]
        return f"{first} {last}"
    return str(name)

@st.cache_data(show_spinner=True)
def load_csv_norm(path: str) -> pd.DataFrame:
    """Robust CSV loader; also normalizes a 'Date' column to date."""
    if not os.path.exists(path):
        # Try relative fallback if absolute path is missing
        rel = os.path.basename(path)
        if os.path.exists(rel):
            path = rel
        else:
            raise FileNotFoundError(f"CSV not found at {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")

    # Normalize a "Date" column to datetime.date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    else:
        df["Date"] = pd.NaT
    return df

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

# ───────────── STANDARD (POST-GAME) HITTER REPORT ─────────────
def create_hitter_report(df: pd.DataFrame, batter: str, ncols: int = 3):
    bdf = df[df["Batter"] == batter].copy()
    # Group by PAs
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

    # Figure & layout; keep zone size fixed, add a bit more gap under pads
    fig = plt.figure(figsize=(3 + 4 * ncols + 1, 4 * nrows))
    gs = GridSpec(nrows, ncols + 1, width_ratios=[0.8] + [1] * ncols, wspace=0.1)

    # Optional logo
    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88, 0.88, 0.12, 0.12], anchor="NE", zorder=5)
        axl.imshow(mpimg.imread(LOGO_PATH))
        axl.axis("off")

    # Title (single game date pulled from filtered dataframe)
    game_date = None
    if not bdf.empty and "Date" in bdf.columns:
        dates = pd.unique(bdf["Date"])
        if len(dates) >= 1:
            game_date = str(dates[0])
    if game_date:
        fig.suptitle(f"{format_name(batter)} — Hitter Report ({game_date})",
                     fontsize=16, x=0.55, y=1.0, fontweight="bold")

    # Quick summary
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

    # PA strike zone panels
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
                ha="center", va="top", fontsize=9, style="italic")  # slightly lower to add gap

    # Left description column
    axd = fig.add_subplot(gs[:, 0])
    axd.axis("off")
    # Space text comfortably even if only first row exists
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

    # Keep strike zone size but add a bit more bottom gap under them
    plt.tight_layout(rect=[0.12, 0.04, 1, 0.88])
    return fig

# ───────────── HITTER HEATMAPS (with bugfix) ─────────────
def _plot_conditional_panel(ax, sub: pd.DataFrame, title: str):
    x_min, x_max, y_min, y_max = get_view_bounds()
    draw_strikezone(ax)
    x = pd.to_numeric(sub.get("PlateLocSide", pd.Series(dtype=float)), errors="coerce").to_numpy()
    y = pd.to_numeric(sub.get("PlateLocHeight", pd.Series(dtype=float)), errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 10:
        # scatter colored by pitch type
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
    df_b["iscontact"] = df_b["PitchCall"].isin(["InPlay","FoulBallFieldable","FoulBallNotFieldable"])
    df_b["iswhiff"]   = df_b["PitchCall"].eq("StrikeSwinging")
    df_b["is95plus"]  = pd.to_numeric(df_b.get("ExitSpeed", np.nan), errors="coerce") >= 95

    # Filter by pitcher hand if requested
    if hand_choice in {"LHP", "RHP"}:
        want = "Left" if hand_choice == "LHP" else "Right"
        df_b = df_b[df_b["PitcherThrows"].astype(str) == want]

    if hand_choice == "Both":
        fig = plt.figure(figsize=(24, 6))
        gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

        sub_contact_l = df_b[df_b["iscontact"] & (df_b["PitcherThrows"]=="Left")]
        sub_contact_r = df_b[df_b["iscontact"] & (df_b["PitcherThrows"]=="Right")]
        ax1 = fig.add_subplot(gs[0, 0]); _plot_conditional_panel(ax1, sub_contact_l, "Contact vs LHP")
        ax2 = fig.add_subplot(gs[0, 2]); _plot_conditional_panel(ax2, sub_contact_r, "Contact vs RHP")

        sub_wh_l = df_b[df_b["iswhiff"] & (df_b["PitcherThrows"]=="Left")]
        sub_wh_r = df_b[df_b["iswhiff"] & (df_b["PitcherThrows"]=="Right")]
        ax3 = fig.add_subplot(gs[0, 3]); _plot_conditional_panel(ax3, sub_wh_l, "Whiffs vs LHP")
        ax4 = fig.add_subplot(gs[0, 5]); _plot_conditional_panel(ax4, sub_wh_r, "Whiffs vs RHP")  # ← fixed order

        sub_95_l = df_b[df_b["is95plus"] & (df_b["PitcherThrows"]=="Left")]
        sub_95_r = df_b[df_b["is95plus"] & (df_b["PitcherThrows"]=="Right")]
        ax5 = fig.add_subplot(gs[0, 6]); _plot_conditional_panel(ax5, sub_95_l, "Exit ≥95 vs LHP")
        ax6 = fig.add_subplot(gs[0, 8]); _plot_conditional_panel(ax6, sub_95_r, "Exit ≥95 vs RHP")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        return fig

    # Single hand (LHP/RHP): 3 panels (Contact, Whiffs, 95+)
    fig = plt.figure(figsize=(18, 5.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.25)
    sub_c = df_b[df_b["iscontact"]]
    sub_w = df_b[df_b["iswhiff"]]
    sub_9 = df_b[df_b["is95plus"]]
    ax1 = fig.add_subplot(gs[0, 0]); _plot_conditional_panel(ax1, sub_c, "Contact")
    ax2 = fig.add_subplot(gs[0, 1]); _plot_conditional_panel(ax2, sub_w, "Whiffs")
    ax3 = fig.add_subplot(gs[0, 2]); _plot_conditional_panel(ax3, sub_9, "Exit ≥95")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

# ───────────── LOAD DATA ─────────────
try:
    df_all = load_csv_norm(DATA_PATH)
except Exception as e:
    st.error(f"Failed to read CSV at {DATA_PATH}: {e}")
    st.stop()

# Ensure required columns exist
needed = ["Date","BatterTeam","Batter","Pitcher","PitcherThrows","GameID",
          "Inning","Top/Bottom","PAofInning","PitchofPA","AutoPitchType",
          "EffectiveVelo","PitchCall","ExitSpeed","PlateLocSide","PlateLocHeight"]
missing = [c for c in needed if c not in df_all.columns]
if missing:
    st.warning(f"Some expected columns are missing and will be treated as empty: {missing}")

# Nebraska hitters only
df_neb_hit = df_all[df_all.get("BatterTeam","") == "NEB"].copy()
if df_neb_hit.empty:
    st.warning("No Nebraska hitter rows found in the dataset.")
    st.stop()

# Available game dates (single-game selection for post-game report)
date_options = sorted(pd.unique(df_neb_hit["Date"]))
colA, colB, colC = st.columns([1.2, 1.4, 2])
sel_date = colA.selectbox("Game Date", options=date_options, index=len(date_options)-1 if date_options else 0)
df_date = df_neb_hit[df_neb_hit["Date"] == sel_date]

# Batter list for that date
batters = sorted(df_date["Batter"].dropna().unique().tolist())
if not batters:
    st.info("No Nebraska batters found for that date.")
    st.stop()
batter = colB.selectbox("Batter", options=batters)

# Tabs: Standard (single game) and Heatmaps (same selected game)
tab_std, tab_hm = st.tabs(["Standard", "Heatmaps"])

with tab_std:
    st.markdown(f"**{format_name(batter)} — {sel_date}**")
    fig_std = create_hitter_report(df_date, batter, ncols=3)
    if fig_std:
        st.pyplot(fig=fig_std)

with tab_hm:
    hand_choice = st.radio("Pitcher Hand", options=["Both","LHP","RHP"], index=0, horizontal=True)
    bdf_hm = df_date[df_date["Batter"] == batter].copy()
    fig_hm = hitter_heatmaps(bdf_hm, batter, hand_choice=hand_choice)
    if fig_hm:
        st.pyplot(fig=fig_hm)
