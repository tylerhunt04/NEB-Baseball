# hitter_app.py — Nebraska-only Hitter Reports (Standard + Heatmaps)

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

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA_PATH  = "B10C25_streamlit_streamlit_columns.csv"  # your “streamlits columns” file
LOGO_PATH  = "Nebraska-Cornhuskers-Logo.png"                      # optional logo
BANNER_IMG = "NebraskaChampions.jpg"                    # optional banner

# ─── STREAMLIT PAGE ──────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Nebraska Baseball — Hitter Reports")

# Banner (optional)
if os.path.exists(BANNER_IMG):
    st.image(BANNER_IMG, use_container_width=True)

# ─── CUSTOM COLORMAP & STRIKE-ZONE GEOMETRY ──────────────────────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [(0.0, "white"), (0.2, "deepskyblue"), (0.3, "white"), (0.7, "red"), (1.0, "red")],
    N=256,
)

def get_zone_bounds():
    # fixed zone so sizes are consistent
    left, bottom = -0.83, 1.17
    width, height = 1.66, 2.75
    return left, bottom, width, height

def get_view_bounds():
    left, bottom, width, height = get_zone_bounds()
    mx = width * 0.8
    my = height * 0.6
    x_min = left - mx
    x_max = left + width + mx
    y_min = bottom - my
    y_max = bottom + height + my
    return x_min, x_max, y_min, y_max

def draw_strikezone(ax, sz_left=None, sz_bottom=None, sz_width=None, sz_height=None):
    left, bottom, width, height = get_zone_bounds()
    if sz_left is None: sz_left = left
    if sz_bottom is None: sz_bottom = bottom
    if sz_width is None: sz_width = width
    if sz_height is None: sz_height = height
    zone = Rectangle((sz_left, sz_bottom), sz_width, sz_height,
                     fill=False, linewidth=2, linestyle='-', color='black')
    ax.add_patch(zone)
    for frac in (1/3, 2/3):
        ax.vlines(sz_left + sz_width * frac, sz_bottom, sz_bottom + sz_height,
                  colors='gray', linestyles='--', linewidth=1)
        ax.hlines(sz_bottom + sz_height * frac, sz_left, sz_left + sz_width,
                  colors='gray', linestyles='--', linewidth=1)

# ─── COLOR MAPS & LABEL HELPERS ──────────────────────────────────────────────
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

def format_name(name: str) -> str:
    if isinstance(name, str) and ',' in name:
        last, first = [s.strip() for s in name.split(',', 1)]
        return f"{first} {last}"
    return str(name)

# ─── KERNEL DENSITY (HITTER) ─────────────────────────────────────────────────
def compute_density_hitter(x, y, xi_m, yi_m):
    coords = np.vstack([x, y])
    mask = np.isfinite(coords).all(axis=0)
    zi = np.zeros(xi_m.shape)
    if mask.sum() > 1:
        try:
            kde = gaussian_kde(coords[:, mask])
            zi = kde(np.vstack([xi_m.ravel(), yi_m.ravel()])).reshape(xi_m.shape)
        except Exception:
            zi = np.zeros(xi_m.shape)
    return zi

def plot_conditional(ax, sub, title):
    x_min, x_max, y_min, y_max = get_view_bounds()
    draw_strikezone(ax)
    x = sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy()
    y = sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy()
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]; y = y[valid_mask]

    if len(sub) < 10:
        # scatter fallback
        for _, r in sub.iterrows():
            if not (np.isfinite(r.get('PlateLocSide', np.nan)) and np.isfinite(r.get('PlateLocHeight', np.nan))):
                continue
            color = PITCH_COLORS.get(r.get('AutoPitchType', ''), 'gray')
            ax.plot(r['PlateLocSide'], r['PlateLocHeight'], 'o', color=color, alpha=0.8, markersize=6)
    else:
        # heatmap
        xi = np.linspace(x_min, x_max, 200)
        yi = np.linspace(y_min, y_max, 200)
        xi_m, yi_m = np.meshgrid(xi, yi)
        zi = compute_density_hitter(
            sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy(),
            sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy(),
            xi_m, yi_m
        )
        ax.imshow(zi, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap=custom_cmap)
        draw_strikezone(ax)

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal', 'box')
    ax.set_title(title, fontsize=10, pad=6, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

def place_between_with_offset(ax_left, ax_right, handles, labels, title, fig,
                              width=0.035, height=0.35, x_offset=0.0, y_offset=0.0):
    fig.canvas.draw()
    pos_l = ax_left.get_position()
    pos_r = ax_right.get_position()
    mid_x = (pos_l.x1 + pos_r.x0) / 2
    y_center = pos_l.y0 + pos_l.height / 2
    x0 = mid_x - width / 2 + x_offset
    y0 = y_center - height / 2 + y_offset
    ax_leg = fig.add_axes([x0, y0, width, height], zorder=6)
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, title=title, loc='upper left', fontsize=8, title_fontsize=9)
    return ax_leg

# ─── HITTER HEATMAP REPORT (from your code) ───────────────────────────────────
def combined_hitter_heatmap_report(df, batter, logo_img=None):
    df_b = df[df['Batter'] == batter].copy()
    if df_b.empty:
        st.error(f"No data for batter '{batter}' on that date.")
        return None

    df_b['iscontact'] = df_b['PitchCall'].isin(['InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable'])
    df_b['iswhiff'] = df_b['PitchCall'] == 'StrikeSwinging'
    df_b['is95plus'] = df_b['ExitSpeed'] >= 95

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

    # Contact L/R
    sub_contact_l = df_b[df_b['iscontact'] & (df_b['PitcherThrows'] == 'Left')]
    sub_contact_r = df_b[df_b['iscontact'] & (df_b['PitcherThrows'] == 'Right')]
    ax1 = fig.add_subplot(gs[0, 0]); plot_conditional(ax1, sub_contact_l, 'Contact vs LHP')
    ax2 = fig.add_subplot(gs[0, 2]); plot_conditional(ax2, sub_contact_r, 'Contact vs RHP')

    # Whiffs L/R
    sub_whiff_l = df_b[df_b['iswhiff'] & (df_b['PitcherThrows'] == 'Left')]
    sub_whiff_r = df_b[df_b['iswhiff'] & (df_b['PitcherThrows'] == 'Right')]
    ax3 = fig.add_subplot(gs[0, 3]); plot_conditional(ax3, sub_whiff_l, 'Whiffs vs LHP')
    ax4 = fig.add_subplot(gs[0, 5]); plot_conditional(ax4, sub_whiff_r, 'Whiffs vs RHP')

    # ≥95 L/R
    sub_95_l = df_b[df_b['is95plus'] & (df_b['PitcherThrows'] == 'Left')]
    sub_95_r = df_b[df_b['is95plus'] & (df_b['PitcherThrows'] == 'Right')]
    ax5 = fig.add_subplot(gs[0, 6]); plot_conditional(ax5, sub_95_l, 'Exit ≥95 vs LHP')
    ax6 = fig.add_subplot(gs[0, 8]); plot_conditional(ax6, sub_95_r, 'Exit ≥95 vs RHP')

    # Group boxes & mini legends (kept from your code)
    group_box_alpha = 0.18
    divider_color = '#444444'
    pad_extra = 0.05

    pos1 = ax1.get_position(); pos2 = ax2.get_position()
    contact_group_x0 = pos1.x0 - 0.005
    contact_group_width = (pos2.x1 + 0.005) - contact_group_x0
    contact_group_y0 = min(pos1.y0, pos2.y0) - pad_extra
    contact_group_height = max(pos1.y1, pos2.y1) - contact_group_y0 + pad_extra
    fig.patches.append(Rectangle((contact_group_x0, contact_group_y0),
                                 contact_group_width, contact_group_height,
                                 transform=fig.transFigure,
                                 facecolor='lightgray', alpha=group_box_alpha,
                                 edgecolor=divider_color, linewidth=1.5, zorder=1))

    pos3 = ax3.get_position(); pos4 = ax4.get_position()
    whiff_group_x0 = pos3.x0 - 0.005
    whiff_group_width = (pos4.x1 + 0.005) - whiff_group_x0
    whiff_group_y0 = min(pos3.y0, pos4.y0) - pad_extra
    whiff_group_height = max(pos3.y1, pos4.y1) - whiff_group_y0 + pad_extra
    fig.patches.append(Rectangle((whiff_group_x0, whiff_group_y0),
                                 whiff_group_width, whiff_group_height,
                                 transform=fig.transFigure,
                                 facecolor='lightgray', alpha=group_box_alpha,
                                 edgecolor=divider_color, linewidth=1.5, zorder=1))

    pos5 = ax5.get_position(); pos6 = ax6.get_position()
    high95_group_x0 = pos5.x0 - 0.005
    high95_group_width = (pos6.x1 + 0.005) - high95_group_x0
    high95_group_y0 = min(pos5.y0, pos6.y0) - pad_extra
    high95_group_height = max(pos5.y1, pos6.y1) - high95_group_y0 + pad_extra
    fig.patches.append(Rectangle((high95_group_x0, high95_group_y0),
                                 high95_group_width, high95_group_height,
                                 transform=fig.transFigure,
                                 facecolor='lightgray', alpha=group_box_alpha,
                                 edgecolor=divider_color, linewidth=1.5, zorder=1))

    sep_width = 0.006
    sep_x1 = (pos2.x1 + pos3.x0) / 2 - sep_width / 2
    sep_y0_1 = min(contact_group_y0, whiff_group_y0)
    sep_height_1 = max(contact_group_height, whiff_group_height)
    fig.patches.append(Rectangle((sep_x1, sep_y0_1), sep_width, sep_height_1,
                                 transform=fig.transFigure,
                                 facecolor=divider_color, alpha=0.9, zorder=2))
    sep_x2 = (pos4.x1 + pos5.x0) / 2 - sep_width / 2
    sep_y0_2 = min(whiff_group_y0, high95_group_y0)
    sep_height_2 = max(whiff_group_height, high95_group_height)
    fig.patches.append(Rectangle((sep_x2, sep_y0_2), sep_width, sep_height_2,
                                 transform=fig.transFigure,
                                 facecolor=divider_color, alpha=0.9, zorder=2))

    # Tallies between panels
    ct = df_b[df_b['iscontact']]
    cts = ct['AutoPitchType'].value_counts()
    if not cts.empty:
        contact_handles = [Line2D([0], [0], marker='o', color=PITCH_COLORS.get(pt, 'gray'),
                                  linestyle='', markersize=6) for pt in cts.index]
        contact_labels = [f"{pt} ({cts[pt]})" for pt in cts.index]
        place_between_with_offset(ax1, ax2, contact_handles, contact_labels,
                                  title=f'Contacts: {ct.shape[0]}', fig=fig,
                                  width=0.035, height=0.35, x_offset=-0.01, y_offset=-0.01)

    wf = df_b[df_b['iswhiff']]
    wfs = wf['AutoPitchType'].value_counts()
    if not wfs.empty:
        whiff_handles = [Line2D([0], [0], marker='o', color=PITCH_COLORS.get(pt, 'gray'),
                                linestyle='', markersize=6) for pt in wfs.index]
        whiff_labels = [f"{pt} ({wfs[pt]})" for pt in wfs.index]
        place_between_with_offset(ax3, ax4, whiff_handles, whiff_labels,
                                  title=f'Whiffs: {wf.shape[0]}', fig=fig,
                                  width=0.035, height=0.35, x_offset=-0.01, y_offset=-0.01)

    high95 = df_b[df_b['is95plus']]
    high95_types = high95['AutoPitchType'].value_counts()
    if not high95_types.empty:
        high95_handles = [Line2D([0], [0], marker='o', color=PITCH_COLORS.get(pt, 'gray'),
                                 linestyle='', markersize=6) for pt in high95_types.index]
        high95_labels = [f"{pt} ({high95_types[pt]})" for pt in high95_types.index]
        place_between_with_offset(ax5, ax6, high95_handles, high95_labels,
                                  title=f'Exit ≥95: {high95.shape[0]}', fig=fig,
                                  width=0.035, height=0.35, x_offset=-0.008, y_offset=-0.01)

    formatted = format_name(batter)
    fig.suptitle(formatted, fontsize=22, x=0.5, y=0.87)
    plt.tight_layout(rect=[0, 0, 1, 0.78])
    return fig

# ─── STANDARD HITTER REPORT (from your code) ──────────────────────────────────
def create_hitter_report(df, batter, ncols=3):
    bdf = df[df['Batter'] == batter]
    pa = list(bdf.groupby(['GameID', 'Inning', 'Top/Bottom', 'PAofInning']))
    n_pa = len(pa); nrows = max(1, math.ceil(n_pa / ncols))  # guard for 0

    # Build left-column descriptions
    descs = []
    for _, padf in pa:
        lines = []
        for _, p in padf.iterrows():
            vel = p.EffectiveVelo if pd.notna(p.EffectiveVelo) else np.nan
            vel_s = f"{vel:.1f}" if pd.notna(vel) else "—"
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType} {vel_s} MPH / {p.PitchCall}")
        ip = padf[padf['PitchCall'] == 'InPlay']
        if not ip.empty:
            last = ip.iloc[-1]; res = last.PlayResult or 'InPlay'
            if not pd.isna(last.ExitSpeed):
                res += f" ({last.ExitSpeed:.1f} MPH)"
            lines.append(f"▶ PA Result: {res}")
        else:
            balls = (padf['PitchCall'] == 'BallCalled').sum()
            strikes = padf['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging']).sum()
            if balls >= 4:
                lines.append('▶ PA Result: Walk')
            elif strikes >= 3:
                lines.append('▶ PA Result: Strikeout')
        descs.append(lines)

    # Figure & layout
    fig = plt.figure(figsize=(3 + 4 * ncols + 1, 4 * nrows))
    gs = GridSpec(nrows, ncols + 1, width_ratios=[0.8] + [1] * ncols, wspace=0.1)

    # Logo (optional)
    logo = mpimg.imread(LOGO_PATH) if os.path.exists(LOGO_PATH) else None
    if logo is not None:
        axl = fig.add_axes([0.88, 0.88, 0.12, 0.12], anchor='NE')
        axl.imshow(logo); axl.axis('off')

    # Title & summary metrics
    if pa:
        date = pa[0][1]['Date'].iloc[0]
        fig.suptitle(f"{batter} Hitter Report for {date}", fontsize=16, x=0.55, y=1.0, fontweight='bold')

    gd = pd.concat([grp for _, grp in pa]) if pa else pd.DataFrame()
    whiffs   = (gd['PitchCall'] == 'StrikeSwinging').sum() if not gd.empty else 0
    hardhits = (gd['ExitSpeed'] > 95).sum() if not gd.empty else 0
    chases   = 0
    if not gd.empty:
        chases = gd[
            (gd['PitchCall'] == 'StrikeSwinging') &
            ((gd['PlateLocSide'] < -0.83) | (gd['PlateLocSide'] > 0.83) |
             (gd['PlateLocHeight'] < 1.5) | (gd['PlateLocHeight'] > 3.5))
        ].shape[0]

    fig.text(0.55, 0.96, f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",
             ha='center', va='top', fontsize=12)

    # PA panels
    for idx, ((_, inn, tb, _), padf) in enumerate(pa):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col + 1])
        draw_strikezone(ax)
        hand = 'LHP' if str(padf['PitcherThrows'].iloc[0]).startswith('L') else 'RHP'
        pitchr = padf['Pitcher'].iloc[0]
        for _, p in padf.iterrows():
            mk = {'Fastball': 'o', 'Curveball': 's', 'Slider': '^', 'Changeup': 'D'}.get(p.AutoPitchType, 'o')
            clr = {'StrikeCalled': '#CCCC00', 'BallCalled': 'green',
                   'FoulBallNotFieldable': 'tan', 'InPlay': '#6699CC',
                   'StrikeSwinging': 'red', 'HitByPitch': 'lime'}.get(p.PitchCall, 'black')
            sz = 200 if p.AutoPitchType == 'Slider' else 150
            ax.scatter(p.PlateLocSide, p.PlateLocHeight, marker=mk, c=clr,
                       s=sz, edgecolor='white', linewidth=1, zorder=2)
            yoff = -0.05 if p.AutoPitchType == 'Slider' else 0
            ax.text(p.PlateLocSide, p.PlateLocHeight + yoff,
                    str(int(p.PitchofPA)), ha='center', va='center',
                    fontsize=6, fontweight='bold', zorder=3)
        ax.set_xlim(-3, 3); ax.set_ylim(0, 5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"PA {idx+1} | Inning {inn} {tb}", fontsize=10, fontweight='bold')
        ax.text(0.5, 0.1, f"vs {pitchr} ({hand})", transform=ax.transAxes, ha='center', va='top',
                fontsize=9, style='italic')

    # Left description column
    axd = fig.add_subplot(gs[:, 0]); axd.axis('off')
    y0 = 1.0; dy = 1.0 / (max(1, n_pa) * 5.0)
    for i, lines in enumerate(descs, 1):
        axd.hlines(y0 - dy * 0.1, 0, 1, transform=axd.transAxes, color='black', linewidth=1)
        axd.text(0.02, y0, f"PA {i}", fontsize=6, fontweight='bold', transform=axd.transAxes)
        yln = y0 - dy
        for ln in lines:
            axd.text(0.02, yln, ln, fontsize=6, transform=axd.transAxes)
            yln -= dy
        y0 = yln - dy * 0.05

    # Legends
    res_handles = [
        Line2D([0], [0], marker='o', color='w', label=k,
               markerfacecolor=v, markersize=10, markeredgecolor='k')
        for k, v in {
            'StrikeCalled': '#CCCC00', 'BallCalled': 'green',
            'FoulBallNotFieldable': 'tan', 'InPlay': '#6699CC',
            'StrikeSwinging': 'red', 'HitByPitch': 'lime'
        }.items()
    ]
    fig.legend(res_handles, [h.get_label() for h in res_handles],
               title='Result', loc='lower right', bbox_to_anchor=(0.90, 0.02))

    pitch_handles = [
        Line2D([0], [0], marker=m, color='w', label=k,
               markerfacecolor='gray', markersize=10, markeredgecolor='k')
        for k, m in {'Fastball': 'o', 'Curveball': 's', 'Slider': '^', 'Changeup': 'D'}.items()
    ]
    fig.legend(pitch_handles, [h.get_label() for h in pitch_handles],
               title='Pitches', loc='lower right', bbox_to_anchor=(0.98, 0.02))

    plt.tight_layout(rect=[0.12, 0.05, 1, 0.88])
    return fig

# ─── LOAD DATA (Nebraska-only hitters) ────────────────────────────────────────
@st.cache_data(show_spinner=True)
def _load_csv_norm(path: str, _mtime: float):
    df = pd.read_csv(path, low_memory=False)
    # normalize Date to date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    else:
        df["Date"] = pd.NaT
    return df

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

mtime = os.path.getmtime(DATA_PATH)
df_all = _load_csv_norm(DATA_PATH, mtime)

required_cols = ['Date', 'BatterTeam', 'Batter']
missing = [c for c in required_cols if c not in df_all.columns]
if missing:
    st.error(f"Missing required columns in CSV: {missing}")
    st.stop()

# Nebraska-only
df_all = df_all[df_all['BatterTeam'] == 'NEB'].copy()
if df_all.empty:
    st.info("No Nebraska hitter rows found.")
    st.stop()

# ─── UI: Date → Batter; Tabs for Standard vs Heatmaps ─────────────────────────
st.subheader("Hitter Reports")

available_dates = sorted(pd.Series(df_all['Date']).dropna().unique().tolist())
if not available_dates:
    st.info("No valid dates found for Nebraska hitters.")
    st.stop()

c1, c2 = st.columns([1.2, 2])

with c1:
    sel_date = st.selectbox("Game Date", options=available_dates)

df_date = df_all[df_all['Date'] == sel_date]
batters = sorted(df_date['Batter'].dropna().unique().tolist())

with c2:
    batter = st.selectbox("Batter (NEB)", batters) if batters else None

if not batter:
    st.info("Choose a Nebraska batter.")
    st.stop()

tabs = st.tabs(["Standard", "Heatmaps"])

with tabs[0]:
    fig = create_hitter_report(df_date, batter, ncols=3)
    if fig:
        st.pyplot(fig=fig)

with tabs[1]:
    fig = combined_hitter_heatmap_report(df_date, batter)
    if fig:
        st.pyplot(fig=fig)
