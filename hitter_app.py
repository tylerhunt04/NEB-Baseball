# hitter_app.py

import os
import io
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
from numpy.linalg import LinAlgError
from matplotlib import colors

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG / PATHS
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "B10C25_streamlit_streamlit_columns.csv"  # same file you use elsewhere
LOGO_PATH = "Nebraska-Cornhuskers-Logo.png"           # optional
APP_TITLE = "Nebraska Baseball — Hitter Reports"

st.set_page_config(layout="wide", page_title="Nebraska Hitter Reports")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS: dates, labels, caching
# ──────────────────────────────────────────────────────────────────────────────
DATE_CANDIDATES = [
    "Date","date","GameDate","GAME_DATE","Game Date","date_game","Datetime",
    "DateTime","game_datetime","GameDateTime"
]

@st.cache_data(show_spinner=True)
def load_csv_norm(path: str) -> pd.DataFrame:
    def _load(p):
        try:
            return pd.read_csv(p, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(p, low_memory=False, encoding="latin-1")
    if not os.path.exists(path):
        st.error(f"Data not found at {path}")
        st.stop()
    df = _load(path)
    # normalize Date -> datetime.date
    found = None
    lower = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand.lower() in lower:
            found = lower[cand.lower()]
            break
    if found is None:
        df["Date"] = pd.NaT
    else:
        dt = pd.to_datetime(df[found], errors="coerce")
        df["Date"] = pd.to_datetime(dt.dt.date, errors="coerce")
    return df

def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"

def fmt_long_date(d) -> str:
    if d is None or pd.isna(d): return ""
    d = pd.to_datetime(d).date()
    return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"

def summarize_dates_range(dates_like) -> str:
    if dates_like is None:
        return ""
    if not isinstance(dates_like, pd.Series):
        dates_like = pd.Series(dates_like)
    ser = pd.to_datetime(dates_like, errors="coerce").dropna()
    if ser.empty: return ""
    uniq = pd.to_datetime(ser.dt.date).unique()
    if len(uniq) == 1:
        return fmt_long_date(uniq[0])
    return f"{fmt_long_date(min(uniq))} – {fmt_long_date(max(uniq))}"

MONTH_CHOICES = [
    (1,"January"), (2,"February"), (3,"March"), (4,"April"),
    (5,"May"), (6,"June"), (7,"July"), (8,"August"),
    (9,"September"), (10,"October"), (11,"November"), (12,"December")
]
MONTH_NAME = {n:nm for n,nm in MONTH_CHOICES}

def build_label_from_selection(dates_series: pd.Series, months_sel, days_sel, last_n=None):
    """
    Pretty label for figure headers:
      - if last_n: 'Last N Games'
      - if neither months nor days: 'Season'
      - if exactly one month and no days: 'March'
      - else: date range from filtered rows
    """
    if last_n:
        return f"Last {last_n} Games"
    if not months_sel and not days_sel:
        return "Season"
    if months_sel and not days_sel and len(months_sel) == 1:
        return MONTH_NAME.get(months_sel[0], "Season")
    if dates_series is None or dates_series.empty:
        return "Season"
    return summarize_dates_range(dates_series)

# ──────────────────────────────────────────────────────────────────────────────
# STRIKE ZONE / VIEW / COLORS
# ──────────────────────────────────────────────────────────────────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
    N=256,
)

def get_zone_bounds():
    # fixed for identical size across heatmap/scatter
    left, bottom = -0.83, 1.17
    width, height = 1.66, 2.75
    return left, bottom, width, height

def get_view_bounds():
    l, b, w, h = get_zone_bounds()
    mx, my = w*0.8, h*0.6
    return l - mx, l + w + mx, b - my, b + h + my

def draw_strikezone(ax, sz_left=None, sz_bottom=None, sz_w=None, sz_h=None):
    l, b, w, h = get_zone_bounds()
    sz_left   = l if sz_left   is None else sz_left
    sz_bottom = b if sz_bottom is None else sz_bottom
    sz_w      = w if sz_w      is None else sz_w
    sz_h      = h if sz_h      is None else sz_h
    ax.add_patch(Rectangle((sz_left, sz_bottom), sz_w, sz_h, fill=False, lw=2, color="black"))
    for f in (1/3, 2/3):
        ax.vlines(sz_left + sz_w*f,  sz_bottom, sz_bottom+sz_h, colors="gray", ls="--", lw=1)
        ax.hlines(sz_bottom + sz_h*f, sz_left, sz_left+sz_w,     colors="gray", ls="--", lw=1)

def get_pitch_color(ptype):
    s = str(ptype).lower()
    if s.startswith("four-seam fastball") or s == "fastball":
        return "#E60026"
    savant = {
        "sinker": "#FF9300","cutter": "#800080","changeup": "#008000","curveball": "#0033CC",
        "slider": "#CCCC00","splitter": "#00CCCC","knuckle curve": "#000000","screwball": "#CC0066","eephus": "#666666",
    }
    return savant.get(s, "#E60026")

# ──────────────────────────────────────────────────────────────────────────────
# DENSITY & METRIC HELPERS
# ──────────────────────────────────────────────────────────────────────────────
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

def decision_quality_metrics(df: pd.DataFrame) -> dict:
    """Chase% (O-Swing%), Z-Swing%, Whiff% on swings."""
    if df.empty:
        return {"Chase%": np.nan, "Z-Swing%": np.nan, "Whiff% (swing)": np.nan}
    # zone
    l, b, w, h = get_zone_bounds()
    r, t = l + w, b + h
    in_zone = (
        df["PlateLocSide"].between(l, r, inclusive="both") &
        df["PlateLocHeight"].between(b, t, inclusive="both")
    )
    swings = df["PitchCall"].isin(["StrikeSwinging","FoulBallFieldable","FoulBallNotFieldable","InPlay"])
    # rates
    z_den = in_zone.sum()
    o_den = (~in_zone).sum()
    z_swing = (swings & in_zone).sum() / z_den * 100 if z_den > 0 else np.nan
    chase  = (swings & ~in_zone).sum() / o_den * 100 if o_den > 0 else np.nan
    whiff_on_swings = (df["PitchCall"].eq("StrikeSwinging").sum() / swings.sum() * 100) if swings.sum() > 0 else np.nan
    return {"Chase%": chase, "Z-Swing%": z_swing, "Whiff% (swing)": whiff_on_swings}

# ──────────────────────────────────────────────────────────────────────────────
# HITTER HEATMAPS (with LHP/RHP/Both toggle)
# ──────────────────────────────────────────────────────────────────────────────
def _plot_conditional_panel(ax, sub, title, grid_size=200):
    x_min, x_max, y_min, y_max = get_view_bounds()
    x = pd.to_numeric(sub.get("PlateLocSide", pd.Series(dtype=float)), errors="coerce").to_numpy()
    y = pd.to_numeric(sub.get("PlateLocHeight", pd.Series(dtype=float)), errors="coerce").to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10:
        # small sample → scatter with pitch colors
        for _, r in sub.loc[valid].iterrows():
            clr = get_pitch_color(r.get("AutoPitchType",""))
            ax.plot(r["PlateLocSide"], r["PlateLocHeight"], "o", color=clr, alpha=0.8, ms=6)
    else:
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        xi_m, yi_m = np.meshgrid(xi, yi)
        zi = compute_density_hitter(x, y, xi_m, yi_m)
        ax.imshow(zi, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="equal", cmap=custom_cmap)
    draw_strikezone(ax)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect("equal","box")
    ax.set_title(title, fontsize=10, pad=6, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

def hitter_heatmaps(df_batter: pd.DataFrame, batter_name: str, hand_choice: str):
    """
    hand_choice: 'Both', 'LHP', or 'RHP' (pitcher handedness).
    - Both: 6 panels (Contact/Whiffs/95+ split by LHP/RHP)
    - LHP/RHP: 3 panels (Contact, Whiffs, 95+) for that hand only
    """
    if df_batter.empty:
        return None

    df_b = df_batter.copy()
    df_b["iscontact"] = df_b["PitchCall"].isin(["InPlay","FoulBallFieldable","FoulBallNotFieldable"])
    df_b["iswhiff"]   = df_b["PitchCall"].eq("StrikeSwinging")
    df_b["is95plus"]  = pd.to_numeric(df_b.get("ExitSpeed", np.nan), errors="coerce") >= 95

    # Hand filter on pitcher throws
    hand_map = {"LHP":"Left","RHP":"Right"}
    if hand_choice in hand_map:
        df_b = df_b[df_b["PitcherThrows"].astype(str) == hand_map[hand_choice]]

    # Layout
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
        ax4 = fig.add_subplot(gs[0, 5]); _plot_conditional_panel(ax4, sub_wh_r, "Whiffs vs RHP")

        sub_95_l = df_b[df_b["is95plus"] & (df_b["PitcherThrows"]=="Left")]
        sub_95_r = df_b[df_b["is95plus"] & (df_b["PitcherThrows"]=="Right")]
        ax5 = fig.add_subplot(gs[0, 6]); _plot_conditional_panel(ax5, sub_95_l, "Exit ≥95 vs LHP")
        ax6 = fig.add_subplot(gs[0, 8]); _plot_conditional_panel(ax6, sub_95_r, "Exit ≥95 vs RHP")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        return fig

    # Single-hand (LHP or RHP) → 3 panels
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

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD HITTER REPORT (extra bottom padding for 1-row layout)
# ──────────────────────────────────────────────────────────────────────────────
def create_hitter_report(df: pd.DataFrame, batter: str, title_label: str, ncols=3):
    bdf = df[df["Batter"] == batter].copy()
    if bdf.empty:
        return None

    pa = list(bdf.groupby(["GameID","Inning","Top/Bottom","PAofInning"], sort=False))
    n_pa = len(pa)
    nrows = max(1, math.ceil(n_pa / ncols))

    # Build textual descriptions for each PA
    descs = []
    for _, padf in pa:
        lines = []
        for _, p in padf.iterrows():
            velo = p.get("EffectiveVelo", np.nan)
            vtxt = f"{float(velo):.1f} MPH" if pd.notna(velo) else "—"
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType}  {vtxt} / {p.PitchCall}")
        inplay = padf[padf["PitchCall"]=="InPlay"]
        if not inplay.empty:
            last = inplay.iloc[-1]
            res = last.PlayResult if pd.notna(last.PlayResult) else "InPlay"
            ev  = last.get("ExitSpeed", np.nan)
            if pd.notna(ev):
                res += f" ({float(ev):.1f} MPH)"
            lines.append(f"▶ PA Result: {res}")
        else:
            balls   = (padf["PitchCall"]=="BallCalled").sum()
            strikes = padf["PitchCall"].isin(["StrikeCalled","StrikeSwinging"]).sum()
            if balls >= 4:   lines.append("▶ PA Result: Walk")
            elif strikes>=3: lines.append("▶ PA Result: Strikeout")
        descs.append(lines)

    # Figure + layout
    # Keep zone size consistent; add more WHITE SPACE under when only one row
    base_h = 4 * nrows
    extra_pad = 1.2 if nrows == 1 else 0.0  # whitespace margin only
    fig = plt.figure(figsize=(3 + 4 * ncols + 1, base_h + extra_pad))
    gs = GridSpec(nrows, ncols+1, figure=fig, width_ratios=[0.8]+[1]*ncols, wspace=0.10)

    # Logo
    if os.path.exists(LOGO_PATH):
        axl = fig.add_axes([0.88, 0.88, 0.12, 0.12], anchor="NE")
        axl.imshow(mpimg.imread(LOGO_PATH)); axl.axis("off")

    # Title + small summary
    gd = pd.concat([grp for _, grp in pa]) if pa else pd.DataFrame()
    whiffs   = (gd["PitchCall"]=="StrikeSwinging").sum() if not gd.empty else 0
    hardhits = (pd.to_numeric(gd.get("ExitSpeed", np.nan), errors="coerce") > 95).sum() if not gd.empty else 0
    l,b,w,h = get_zone_bounds()
    chases = 0
    if not gd.empty:
        chases = gd[(gd["PitchCall"]=="StrikeSwinging") &
                    (~gd["PlateLocSide"].between(l, l+w)) |
                    (~gd["PlateLocHeight"].between(b, b+h))].shape[0]
    fig.suptitle(f"{batter} — Hitter Report\n({title_label})", fontsize=16, y=0.99, fontweight="bold")
    fig.text(0.56, 0.955, f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",
             ha="center", va="top", fontsize=12)

    # Panels
    for idx, ((_, inn, tb, _), padf) in enumerate(pa):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col+1])
        # strike zone
        draw_strikezone(ax)
        # pitcher hand tag
        hand = "LHP" if str(padf["PitcherThrows"].iloc[0]).upper().startswith("L") else "RHP"
        pitchr = str(padf["Pitcher"].iloc[0])

        for _, p in padf.iterrows():
            mk = {"Fastball":"o","Curveball":"s","Slider":"^","Changeup":"D"}.get(p.AutoPitchType, "o")
            clr= {"StrikeCalled":"#CCCC00","BallCalled":"green","FoulBallNotFieldable":"tan","InPlay":"#6699CC","StrikeSwinging":"red","HitByPitch":"lime"}.get(p.PitchCall, "black")
            sz = 200 if p.AutoPitchType == "Slider" else 150
            ax.scatter(p.PlateLocSide, p.PlateLocHeight, marker=mk, c=clr, s=sz, edgecolor="white", lw=1, zorder=2)
            yoff = -0.05 if p.AutoPitchType == "Slider" else 0
            ax.text(p.PlateLocSide, p.PlateLocHeight+yoff, str(int(p.PitchofPA)),
                    ha="center", va="center", fontsize=6, fontweight="bold", zorder=3)
        ax.set_xlim(-3,3); ax.set_ylim(0,5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}", fontsize=10, fontweight="bold")
        ax.text(0.5, 0.1, f"vs {pitchr} ({hand})", transform=ax.transAxes, ha="center", va="top", fontsize=9, style="italic")

    # Descriptions column
    axd = fig.add_subplot(gs[:,0]); axd.axis("off")
    y0 = 1.0; dy = 1.0 / (max(1, n_pa) * 5.0)
    for i, lines in enumerate(descs, 1):
        axd.hlines(y0 - dy*0.1, 0, 1, transform=axd.transAxes, color="black", lw=1)
        axd.text(0.02, y0, f"PA {i}", fontsize=6, fontweight="bold", transform=axd.transAxes)
        yln = y0 - dy
        for ln in lines:
            axd.text(0.02, yln, ln, fontsize=6, transform=axd.transAxes); yln -= dy
        y0 = yln - dy*0.05

    # Legends
    res_map = {"StrikeCalled":"#CCCC00","BallCalled":"green","FoulBallNotFieldable":"tan","InPlay":"#6699CC","StrikeSwinging":"red","HitByPitch":"lime"}
    res_handles = [Line2D([0],[0], marker="o", color="w", label=k, markerfacecolor=v, ms=10, markeredgecolor="k") for k,v in res_map.items()]
    fig.legend(res_handles, list(res_map.keys()), title="Result", loc="lower right", bbox_to_anchor=(0.90, 0.02))
    pitch_shapes = {"Fastball":"o","Curveball":"s","Slider":"^","Changeup":"D"}
    pitch_handles = [Line2D([0],[0], marker=m, color="w", label=k, markerfacecolor="gray", ms=10, markeredgecolor="k") for k,m in pitch_shapes.items()]
    fig.legend(pitch_handles, list(pitch_shapes.keys()), title="Pitches", loc="lower right", bbox_to_anchor=(0.98, 0.02))

    # Tight layout with extra bottom padding for 1-row case (whitespace only; zone size unchanged)
    rect_top = 0.90 if nrows == 1 else 0.88
    rect_bottom = 0.10 if nrows == 1 else 0.05
    plt.tight_layout(rect=[0.12, rect_bottom, 1, rect_top])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# CSV EXPORT HELPER
# ──────────────────────────────────────────────────────────────────────────────
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def fig_to_png_bytes(fig) -> bytes:
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=200, bbox_inches="tight")
    out.seek(0)
    return out.read()

# ──────────────────────────────────────────────────────────────────────────────
# LOAD / PREP DATA
# ──────────────────────────────────────────────────────────────────────────────
df_all = load_csv_norm(DATA_PATH)

# Nebraska hitters only
df_neb_hit = df_all[df_all.get("BatterTeam","") == "NEB"].copy()
if df_neb_hit.empty:
    st.error("No Nebraska hitter rows found in the dataset.")
    st.stop()

# Ensure numeric columns used in metrics
for col in ["PlateLocSide","PlateLocHeight","ExitSpeed","EffectiveVelo"]:
    if col in df_neb_hit.columns:
        df_neb_hit[col] = pd.to_numeric(df_neb_hit[col], errors="coerce")

# ──────────────────────────────────────────────────────────────────────────────
# UI — Filters
# ──────────────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)

# Top controls
c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.2])
batters = sorted(df_neb_hit["Batter"].dropna().unique().tolist())
batter = c1.selectbox("Batter", batters, index=0)

filter_mode = c2.selectbox("Filter", ["Season", "Last N Games", "Months/Days"], index=0)

last_n = None
months_sel, days_sel = [], []
if filter_mode == "Last N Games":
    last_n = c3.selectbox("N", [3,5,10,15,20], index=1)
elif filter_mode == "Months/Days":
    # derive options from this batter's dates
    dser_all = df_neb_hit[df_neb_hit["Batter"] == batter]["Date"].dropna()
    months_avail = sorted(dser_all.dt.month.unique().tolist())
    days_avail = sorted(dser_all.dt.day.unique().tolist())
    months_sel = c3.multiselect("Months", options=months_avail, format_func=lambda m: MONTH_NAME.get(m, str(m)))
    # days are across all selected months if any, else across all dates
    dser = dser_all
    if months_sel:
        dser = dser[dser.dt.month.isin(months_sel)]
    days_avail = sorted(dser.dt.day.unique().tolist())
    days_sel = c4.multiselect("Days", options=days_avail)

# Heatmap pitcher-hand filter
st.markdown("### Hitter Heatmaps — Pitcher Hand")
hand_choice = st.radio("Show:", ["Both","LHP","RHP"], horizontal=True, index=0, key="hh_hand")

# ──────────────────────────────────────────────────────────────────────────────
# SUBSET DATA per filter
# ──────────────────────────────────────────────────────────────────────────────
bdf_all = df_neb_hit[df_neb_hit["Batter"] == batter].copy()
# compute ordered unique dates for this batter
date_order = sorted(bdf_all["Date"].dropna().unique())
if last_n:
    use_dates = sorted(date_order)[-last_n:]
    bdf = bdf_all[bdf_all["Date"].isin(use_dates)].copy()
elif months_sel or days_sel:
    s = bdf_all["Date"].dropna()
    mask = pd.Series(True, index=bdf_all.index)
    if months_sel:
        mask &= s.dt.month.isin(months_sel)
    if days_sel:
        mask &= s.dt.day.isin(days_sel)
    bdf = bdf_all[mask].copy()
else:
    bdf = bdf_all.copy()

# Build label for figure headers
label_dates = bdf["Date"].dropna()
title_label = build_label_from_selection(label_dates, months_sel, days_sel, last_n=last_n)

if bdf.empty:
    st.info("No rows for the selected filters.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Decision-quality mini panel
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Decision Quality")
metrics = decision_quality_metrics(bdf)
m1, m2, m3 = st.columns(3)
m1.metric("Chase% (O-Swing%)", f"{metrics['Chase%']:.1f}%" if pd.notna(metrics['Chase%']) else "—")
m2.metric("Z-Swing%", f"{metrics['Z-Swing%']:.1f}%" if pd.notna(metrics['Z-Swing%']) else "—")
m3.metric("Whiff% on swings", f"{metrics['Whiff% (swing)']:.1f}%" if pd.notna(metrics['Whiff% (swing)']) else "—")

# ──────────────────────────────────────────────────────────────────────────────
# STANDARD REPORT (PA sequence panels)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Standard Hitter Report")
fig_std = create_hitter_report(bdf, batter, title_label=title_label, ncols=3)
if fig_std:
    st.pyplot(fig=fig_std)
    # downloads
    png_bytes = fig_to_png_bytes(fig_std)
    st.download_button("Download Report PNG", data=png_bytes, file_name=f"{batter.replace(' ','_')}_report.png", mime="image/png")

# per-PA CSV (export the subset)
export_cols = [
    "Date","GameID","Inning","Top/Bottom","PAofInning","PitchofPA",
    "AutoPitchType","EffectiveVelo","PitchCall","PlateLocSide","PlateLocHeight",
    "PlayResult","ExitSpeed","Pitcher","PitcherThrows"
]
cols_exist = [c for c in export_cols if c in bdf.columns]
csv_bytes = to_csv_bytes(bdf[cols_exist].copy())
st.download_button("Download Per-PA CSV", data=csv_bytes, file_name=f"{batter.replace(' ','_')}_pitches.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# HITTER HEATMAPS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Heatmaps")
fig_hm = hitter_heatmaps(bdf, batter, hand_choice=hand_choice)
if fig_hm:
    st.pyplot(fig=fig_hm)
    hm_png = fig_to_png_bytes(fig_hm)
    st.download_button("Download Heatmaps PNG", data=hm_png, file_name=f"{batter.replace(' ','_')}_heatmaps.png", mime="image/png")

# Done.
