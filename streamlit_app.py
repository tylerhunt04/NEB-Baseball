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
from scipy.stats import chi2, gaussian_kde
from numpy.linalg import LinAlgError
from matplotlib import colors

# ─── CONFIG / PATHS ───────────────────────────────────────────────────────────
CSV_PATH  = "5.31.2025 v HC.csv"  # adjust to your repo path
LOGO_PATH = "Nebraska-Cornhuskers-Logo.png"  # adjust to your repo path

st.set_page_config(layout="wide")
st.title("Post-Game Hitter & Pitcher Reports")

# ─── CUSTOM COLORMAPS ─────────────────────────────────────────────────────────
custom_cmap = colors.LinearSegmentedColormap.from_list(
    "custom_cmap",
    [
        (0.0, "white"),
        (0.2, "deepskyblue"),
        (0.3, "white"),
        (0.7, "red"),
        (1.0, "red"),
    ],
    N=256,
)

# ─── STRIKEZONE / VIEW CONSTANTS ───────────────────────────────────────────────
def get_zone_bounds():
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

SZ_LEFT, _, = -0.83, None  # unused directly; use get_zone_bounds when needed
SZ_BOTTOM = 1.17

GRID_SIZE = 100

# ─── PITCH COLOR PALETTES ─────────────────────────────────────────────────────
def get_pitch_color(ptype):
    if ptype.lower().startswith("four-seam fastball") or ptype.lower() == "fastball":
        return "#E60026"
    savant = {
        "sinker": "#FF9300",
        "cutter": "#800080",
        "changeup": "#008000",
        "curveball": "#0033CC",
        "slider": "#CCCC00",
        "splitter": "#00CCCC",
        "knuckle curve": "#000000",
        "screwball": "#CC0066",
        "eephus": "#666666",
    }
    return savant.get(ptype.lower(), "#E60026")

# for hitter heatmap legend & scatter
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

# ─── GENERAL HELPERS ─────────────────────────────────────────────────────────
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

def compute_density(x, y, grid_coords, mesh_shape):
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 2:
        return np.zeros(mesh_shape)
    try:
        kde = gaussian_kde(np.vstack([x_clean, y_clean]))
        return kde(grid_coords).reshape(mesh_shape)
    except LinAlgError:
        return np.zeros(mesh_shape)

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

def strike_rate(sub_df):
    if len(sub_df) == 0:
        return np.nan
    strike_calls = ['StrikeCalled', 'StrikeSwinging', 'FoulBallNotFieldable', 'FoulBallFieldable', 'InPlay']
    return sub_df['PitchCall'].isin(strike_calls).sum() / len(sub_df) * 100

def format_name(name: str) -> str:
    if ',' in name:
        last, first = [s.strip() for s in name.split(',', 1)]
        return f"{first} {last}"
    return name

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

def plot_conditional(ax, sub, title):
    x_min, x_max, y_min, y_max = get_view_bounds()
    draw_strikezone(ax)
    x = sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy()
    y = sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy()
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]; y = y[valid_mask]

    if len(sub) < 10:
        for _, r in sub.iterrows():
            if not (np.isfinite(r.get('PlateLocSide', np.nan)) and np.isfinite(r.get('PlateLocHeight', np.nan))):
                continue
            color = PITCH_COLORS.get(r.get('AutoPitchType', ''), 'gray')
            ax.plot(r['PlateLocSide'], r['PlateLocHeight'], 'o', color=color, alpha=0.8, markersize=6)
    else:
        xi = np.linspace(x_min, x_max, 200)
        yi = np.linspace(y_min, y_max, 200)
        xi_m, yi_m = np.meshgrid(xi, yi)
        zi = compute_density_hitter(sub.get('PlateLocSide', pd.Series(dtype=float)).to_numpy(),
                                    sub.get('PlateLocHeight', pd.Series(dtype=float)).to_numpy(),
                                    xi_m, yi_m)
        ax.imshow(zi, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='equal', cmap=custom_cmap)
        draw_strikezone(ax)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', 'box')
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([]); ax.set_yticks([])

# ─── STANDARD PITCHER REPORT ───────────────────────────────────────────────────
def combined_pitcher_report(df, pitcher_name, logo_img, coverage=0.8):
    df_p = df[df['Pitcher'] == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' on that date.")
        return None

    total = len(df_p)
    is_strike = df_p['PitchCall'].isin([
        'StrikeCalled', 'StrikeSwinging',
        'FoulBallNotFieldable', 'FoulBallFieldable', 'InPlay'
    ])
    grp = df_p.groupby('AutoPitchType')

    summary = pd.DataFrame({
        'Pitch Type': grp.size().index,
        'Pitches': grp.size().values,
        'Usage %': grp.size().values / total * 100,
        'Strike %': grp.apply(lambda g: is_strike.loc[g.index].sum() / len(g) * 100).values,
        'Rel Speed': grp['RelSpeed'].mean().values,
        'Spin Rate': grp['SpinRate'].mean().values,
        'IVB': grp['InducedVertBreak'].mean().values,
        'HB': grp['HorzBreak'].mean().values,
        'Rel Height': grp['RelHeight'].mean().values,
        'VAA': grp['VertApprAngle'].mean().values,
        'Extension': grp['Extension'].mean().values
    }).round({
        'Usage %': 1, 'Strike %': 1, 'Rel Speed': 1, 'Spin Rate': 1,
        'IVB': 1, 'HB': 1, 'Rel Height': 2, 'VAA': 1, 'Extension': 2
    }).sort_values('Pitches', ascending=False)

    cols = ['Pitch Type', 'Pitches', 'Usage %', 'Strike %',
            'Rel Speed', 'Spin Rate', 'IVB', 'HB', 'Rel Height', 'VAA', 'Extension']
    summary = summary[cols]

    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 0.7], hspace=0.3)

    axm = fig.add_subplot(gs[0, 0])
    axm.set_title('Movement Plot')
    chi2v = chi2.ppf(coverage, df=2)
    axm.axhline(0, linestyle='--', color='grey')
    axm.axvline(0, linestyle='--', color='grey')
    for ptype, g in grp:
        x, y = g['HorzBreak'], g['InducedVertBreak']
        clr = get_pitch_color(ptype)
        axm.scatter(x, y, label=ptype, color=clr, alpha=0.7)
        if len(g) > 1:
            cov = np.cov(np.vstack((x, y)))
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * np.sqrt(vals * chi2v)
            ell = Ellipse((x.mean(), y.mean()), w, h,
                          angle=ang, edgecolor=clr, facecolor=clr,
                          alpha=0.2, linestyle='--', linewidth=1.5)
            axm.add_patch(ell)
    axm.set_xlim(-30, 30); axm.set_ylim(-30, 30)
    axm.set_aspect('equal', 'box')
    axm.set_xlabel('Horizontal Break')
    axm.set_ylabel('Induced Vertical Break')
    axm.legend(title='Pitch Type', fontsize=8, title_fontsize=9, loc='upper right')

    axt = fig.add_subplot(gs[1, 0]); axt.axis('off')
    tbl = axt.table(cellText=summary.values,
                    colLabels=summary.columns,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    axt.set_title('Summary Metrics', fontweight='bold', y=0.87)

    if logo_img is not None:
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10)
        axl.imshow(logo_img); axl.axis('off')
    elif os.path.exists(LOGO_PATH):
        logo = mpimg.imread(LOGO_PATH)
        axl = fig.add_axes([1, 0.88, 0.12, 0.12], anchor='NE', zorder=10)
        axl.imshow(logo); axl.axis('off')

    fig.suptitle(f"{pitcher_name} – Full Report", fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, summary

# ─── HEATMAP-STYLE PITCHER REPORT ──────────────────────────────────────────────
def combined_pitcher_heatmap_report(df, pitcher_name, logo_path,
                                    sz_left=None, sz_bottom=None,
                                    sz_width=None, sz_height=None,
                                    grid_size=GRID_SIZE):
    df_p = df[df['Pitcher'] == pitcher_name]
    if df_p.empty:
        st.error(f"No data for pitcher '{pitcher_name}' on that date.")
        return None

    # unify bounds
    x_min, x_max, y_min, y_max = get_view_bounds()
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi_mesh, yi_mesh = np.meshgrid(xi, yi)
    grid_coords = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()])

    # strikezone dims
    z_left, z_bottom, z_w, z_h = get_zone_bounds()

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 0.6], hspace=0.4, wspace=0.3)

    top4 = list(df_p['AutoPitchType'].value_counts().index[:4])

    # top pitch types
    for i, pitch in enumerate(top4):
        ax = fig.add_subplot(gs[0, i])
        sub = df_p[df_p['AutoPitchType'] == pitch]
        x = sub['PlateLocSide'].to_numpy()
        y = sub['PlateLocHeight'].to_numpy()
        if len(sub) < 15:
            ax.scatter(x, y, s=30, alpha=0.7, color='deepskyblue', edgecolors='black')
        else:
            zi = compute_density(x, y, grid_coords, xi_mesh.shape)
            ax.imshow(zi, origin='lower',
                      extent=[x_min, x_max, y_min, y_max],
                      aspect='equal', cmap=custom_cmap)
        draw_strikezone(ax, sz_left=z_left, sz_bottom=z_bottom, sz_width=z_w, sz_height=z_h)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal', 'box')
        ax.set_title(f"{pitch} (n={len(sub)})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.add_subplot(gs[0, 4]).axis('off')

    # vs Left
    sub_vl = df_p[df_p['BatterSide'] == 'Left']
    ax_vl = fig.add_subplot(gs[1, 0])
    zi_vl = compute_density(sub_vl['PlateLocSide'], sub_vl['PlateLocHeight'], grid_coords, xi_mesh.shape)
    ax_vl.imshow(zi_vl, origin='lower',
                 extent=[x_min, x_max, y_min, y_max],
                 aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_vl, sz_left=z_left, sz_bottom=z_bottom, sz_width=z_w, sz_height=z_h)
    ax_vl.set_xlim(x_min, x_max); ax_vl.set_ylim(y_min, y_max); ax_vl.set_aspect('equal', 'box')
    ax_vl.set_title(f"vs Left-Handed (n={len(sub_vl)})")
    ax_vl.set_xticks([]); ax_vl.set_yticks([])

    # vs Right
    sub_vr = df_p[df_p['BatterSide'] == 'Right']
    ax_vr = fig.add_subplot(gs[1, 1])
    zi_vr = compute_density(sub_vr['PlateLocSide'], sub_vr['PlateLocHeight'], grid_coords, xi_mesh.shape)
    ax_vr.imshow(zi_vr, origin='lower',
                 extent=[x_min, x_max, y_min, y_max],
                 aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_vr, sz_left=z_left, sz_bottom=z_bottom, sz_width=z_w, sz_height=z_h)
    ax_vr.set_xlim(x_min, x_max); ax_vr.set_ylim(y_min, y_max); ax_vr.set_aspect('equal', 'box')
    ax_vr.set_title(f"vs Right-Handed (n={len(sub_vr)})")
    ax_vr.set_xticks([]); ax_vr.set_yticks([])

    # Whiffs
    sub_whiff = df_p[df_p['PitchCall'] == 'StrikeSwinging']
    ax_whiff = fig.add_subplot(gs[1, 2])
    zi_whiff = compute_density(sub_whiff['PlateLocSide'], sub_whiff['PlateLocHeight'], grid_coords, xi_mesh.shape)
    ax_whiff.imshow(zi_whiff, origin='lower',
                    extent=[x_min, x_max, y_min, y_max],
                    aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_whiff, sz_left=z_left, sz_bottom=z_bottom, sz_width=z_w, sz_height=z_h)
    ax_whiff.set_xlim(x_min, x_max); ax_whiff.set_ylim(y_min, y_max); ax_whiff.set_aspect('equal', 'box')
    ax_whiff.set_title(f"Whiffs (n={len(sub_whiff)})")
    ax_whiff.set_xticks([]); ax_whiff.set_yticks([])

    # Strikeouts
    sub_ks = df_p[df_p['KorBB'] == 'Strikeout']
    ax_ks = fig.add_subplot(gs[1, 3])
    zi_ks = compute_density(sub_ks['PlateLocSide'], sub_ks['PlateLocHeight'], grid_coords, xi_mesh.shape)
    ax_ks.imshow(zi_ks, origin='lower',
                 extent=[x_min, x_max, y_min, y_max],
                 aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_ks, sz_left=z_left, sz_bottom=z_bottom, sz_width=z_w, sz_height=z_h)
    ax_ks.set_xlim(x_min, x_max); ax_ks.set_ylim(y_min, y_max); ax_ks.set_aspect('equal', 'box')
    ax_ks.set_title(f"Strikeouts (n={len(sub_ks)})")
    ax_ks.set_xticks([]); ax_ks.set_yticks([])

    # Damage
    sub_dmg = df_p[df_p['ExitSpeed'] >= 95]
    ax_dmg = fig.add_subplot(gs[1, 4])
    x_dmg = sub_dmg['PlateLocSide'].to_numpy()
    y_dmg = sub_dmg['PlateLocHeight'].to_numpy()
    if len(sub_dmg) < 15:
        ax_dmg.scatter(x_dmg, y_dmg, s=30, alpha=0.7, color='orange', edgecolors='black')
    else:
        zi_dmg = compute_density(x_dmg, y_dmg, grid_coords, xi_mesh.shape)
        ax_dmg.imshow(zi_dmg, origin='lower',
                      extent=[x_min, x_max, y_min, y_max],
                      aspect='equal', cmap=custom_cmap)
    draw_strikezone(ax_dmg, sz_left=z_left, sz_bottom=z_bottom, sz_width=z_w, sz_height=z_h)
    ax_dmg.set_xlim(x_min, x_max); ax_dmg.set_ylim(y_min, y_max); ax_dmg.set_aspect('equal', 'box')
    ax_dmg.set_title(f"Damage (n={len(sub_dmg)})")
    ax_dmg.set_xticks([]); ax_dmg.set_yticks([])

    # Summary metrics row
    fp = strike_rate(df_p[(df_p['Balls'] == 0) & (df_p['Strikes'] == 0)])
    mix = strike_rate(df_p[((df_p['Balls'] == 1) & (df_p['Strikes'] == 0)) |
                            ((df_p['Balls'] == 0) & (df_p['Strikes'] == 1)) |
                            ((df_p['Balls'] == 1) & (df_p['Strikes'] == 1))])
    hp = strike_rate(df_p[((df_p['Balls'] == 2) & (df_p['Strikes'] == 0)) |
                           ((df_p['Balls'] == 2) & (df_p['Strikes'] == 1)) |
                           ((df_p['Balls'] == 3) & (df_p['Strikes'] == 1))])
    two = strike_rate(df_p[(df_p['Strikes'] == 2) & (df_p['Balls'] < 3)])
    summary_metrics = pd.DataFrame({
        '1st Pitch %': [fp],
        'Mix Count %': [mix],
        'Hitter+ %': [hp],
        '2-Strike %': [two]
    }).round(1)
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis('off')
    tbl = ax_tbl.table(cellText=summary_metrics.values, colLabels=summary_metrics.columns,
                       cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.5, 1.5)
    ax_tbl.set_title('Strike Percentage by Count', y=0.75, fontweight='bold')

    # Logo
    if os.path.exists(logo_path):
        logo = mpimg.imread(logo_path)
        ax_logo = fig.add_axes([0.88, 0.92, 0.10, 0.10], anchor='NE', zorder=10)
        ax_logo.imshow(logo)
        ax_logo.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"{pitcher_name} – Heatmap Report", fontsize=18, y=0.98, fontweight='bold')
    return fig

# ─── HITTER HEATMAP REPORT ────────────────────────────────────────────────────
def combined_hitter_heatmap_report(df, batter, logo_img=None):
    df_b = df[df['Batter'] == batter].copy()
    if df_b.empty:
        st.error(f"No data for batter '{batter}' on that date.")
        return None

    df_b['iscontact'] = df_b['PitchCall'].isin(['InPlay', 'FoulBallFieldable', 'FoulBallNotFieldable'])
    df_b['iswhiff'] = df_b['PitchCall'] == 'StrikeSwinging'
    df_b['is95plus'] = df_b['ExitSpeed'] >= 95

    x_min, x_max, y_min, y_max = get_view_bounds()
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 200)
    xi_m, yi_m = np.meshgrid(xi, yi)

    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 9, figure=fig, wspace=0.05, hspace=0.15)

    sub_contact_l = df_b[df_b['iscontact'] & (df_b['PitcherThrows'] == 'Left')]
    sub_contact_r = df_b[df_b['iscontact'] & (df_b['PitcherThrows'] == 'Right')]
    ax1 = fig.add_subplot(gs[0, 0]); plot_conditional(ax1, sub_contact_l, 'Contact vs LHP')
    ax2 = fig.add_subplot(gs[0, 2]); plot_conditional(ax2, sub_contact_r, 'Contact vs RHP')

    sub_whiff_l = df_b[df_b['iswhiff'] & (df_b['PitcherThrows'] == 'Left')]
    sub_whiff_r = df_b[df_b['iswhiff'] & (df_b['PitcherThrows'] == 'Right')]
    ax3 = fig.add_subplot(gs[0, 3]); plot_conditional(ax3, sub_whiff_l, 'Whiffs vs LHP')
    ax4 = fig.add_subplot(gs[0, 5]); plot_conditional(ax4, sub_whiff_r, 'Whiffs vs RHP')

    sub_95_l = df_b[df_b['is95plus'] & (df_b['PitcherThrows'] == 'Left')]
    sub_95_r = df_b[df_b['is95plus'] & (df_b['PitcherThrows'] == 'Right')]
    ax5 = fig.add_subplot(gs[0, 6]); plot_conditional(ax5, sub_95_l, 'Exit ≥95 vs LHP')
    ax6 = fig.add_subplot(gs[0, 8]); plot_conditional(ax6, sub_95_r, 'Exit ≥95 vs RHP')

    # group background boxes and separators
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
    fig.patches.append(Rectangle((sep_x1, sep_y0_1),
                                 sep_width, sep_height_1,
                                 transform=fig.transFigure,
                                 facecolor=divider_color, alpha=0.9, zorder=2))
    sep_x2 = (pos4.x1 + pos5.x0) / 2 - sep_width / 2
    sep_y0_2 = min(whiff_group_y0, high95_group_y0)
    sep_height_2 = max(whiff_group_height, high95_group_height)
    fig.patches.append(Rectangle((sep_x2, sep_y0_2),
                                 sep_width, sep_height_2,
                                 transform=fig.transFigure,
                                 facecolor=divider_color, alpha=0.9, zorder=2))

    # legends
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

# ─── STANDARD HITTER REPORT ───────────────────────────────────────────────────
def create_hitter_report(df, batter, ncols=3):
    bdf = df[df['Batter'] == batter]
    pa = list(bdf.groupby(['GameID', 'Inning', 'Top/Bottom', 'PAofInning']))
    n_pa = len(pa); nrows = math.ceil(n_pa / ncols)
    descs = []
    for _, padf in pa:
        lines = []
        for _, p in padf.iterrows():
            lines.append(f"{int(p.PitchofPA)} / {p.AutoPitchType} {p.EffectiveVelo:.1f} MPH / {p.PitchCall}")
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

    fig = plt.figure(figsize=(3 + 4 * ncols + 1, 4 * nrows))
    gs = GridSpec(nrows, ncols + 1, width_ratios=[0.8] + [1] * ncols, wspace=0.1)
    logo = mpimg.imread(LOGO_PATH) if os.path.exists(LOGO_PATH) else None
    if logo is not None:
        axl = fig.add_axes([0.88, 0.88, 0.12, 0.12], anchor='NE')
        axl.imshow(logo); axl.axis('off')
    if pa:
        date = pa[0][1]['Date'].iloc[0]
        fig.suptitle(f"{batter} Hitter Report for {date}", fontsize=16, x=0.55, y=1.0, fontweight='bold')
    gd = pd.concat([grp for _, grp in pa]) if pa else pd.DataFrame()
    whiffs = (gd['PitchCall'] == 'StrikeSwinging').sum() if not gd.empty else 0
    hardhits = (gd['ExitSpeed'] > 95).sum() if not gd.empty else 0
    chases = 0
    if not gd.empty:
        chases = gd[(gd['PitchCall'] == 'StrikeSwinging') & (
            (gd['PlateLocSide'] < -0.83) | (gd['PlateLocSide'] > 0.83) |
            (gd['PlateLocHeight'] < 1.5) | (gd['PlateLocHeight'] > 3.5)
        )].shape[0]
    fig.text(0.55, 0.96, f"Whiffs: {whiffs}   Hard Hits: {hardhits}   Chases: {chases}",
             ha='center', va='top', fontsize=12)

    for idx, ((_, inn, tb, _), padf) in enumerate(pa):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col + 1])
        draw_strikezone(ax)
        hand = 'LHP' if padf['PitcherThrows'].iloc[0].startswith('L') else 'RHP'
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
        ax.text(0.5, 0.1, f"vs {pitchr} ({hand})",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=9, style='italic')

    axd = fig.add_subplot(gs[:, 0]); axd.axis('off')
    y0 = 1.0; dy = 1.0 / (n_pa * 5.0 if n_pa else 1)
    for i, lines in enumerate(descs, 1):
        axd.hlines(y0 - dy * 0.1, 0, 1, transform=axd.transAxes, color='black', linewidth=1)
        axd.text(0.02, y0, f"PA {i}", fontsize=6, fontweight='bold', transform=axd.transAxes)
        yln = y0 - dy
        for ln in lines:
            axd.text(0.02, yln, ln, fontsize=6, transform=axd.transAxes)
            yln -= dy
        y0 = yln - dy * 0.05

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

# ─── STREAMLIT APP LOGIC ──────────────────────────────────────────────────────
try:
    df_all = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Failed to read CSV at {CSV_PATH}: {e}")
    st.stop()

required = ['Date', 'PitcherTeam', 'BatterTeam', 'Pitcher', 'Batter']
missing = [c for c in required if c not in df_all.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df_all['Date'] = pd.to_datetime(df_all['Date']).dt.date
all_dates = sorted(df_all['Date'].unique())

col1, col2, col3, col4 = st.columns(4)
report = col1.selectbox("Report Type", ["Pitcher Report", "Hitter Report"], key="report_type")
variant = col2.selectbox("Variant", ["Standard", "Heatmap"], key="variant")
selected_date = col3.selectbox("Game Date", all_dates, key="game_date")

df_date = df_all[df_all['Date'] == selected_date]

# load logo
logo_img = None
if os.path.exists(LOGO_PATH):
    logo_img = mpimg.imread(LOGO_PATH)
else:
    st.warning(f"Logo not found at {LOGO_PATH}")

if report == "Pitcher Report":
    df_p = df_date[df_date['PitcherTeam'] == 'NEB']
    pitchers = sorted(df_p['Pitcher'].unique())
    if not pitchers:
        st.warning("No NEB pitchers for that date.")
    player = col4.selectbox("Pitcher", pitchers, key="pitcher_name")
    st.subheader(f"{player} — {selected_date}")
    if variant == "Heatmap":
        fig = combined_pitcher_heatmap_report(df_p, player, LOGO_PATH)
        if fig:
            st.pyplot(fig=fig)
    else:
        result = combined_pitcher_report(df_p, player, logo_img, coverage=0.8)
        if result:
            fig, summary = result
            st.pyplot(fig=fig)
            st.table(summary)
else:
    df_b = df_date[df_date['BatterTeam'] == 'NEB']
    batters = sorted(df_b['Batter'].unique())
    if not batters:
        st.warning("No NEB batters for that date.")
    player = col4.selectbox("Batter", batters, key="batter_name")
    st.subheader(f"{player} — {selected_date}")
    if variant == "Heatmap":
        fig = combined_hitter_heatmap_report(df_b, player, logo_img=logo_img)
        if fig:
            st.pyplot(fig=fig)
    else:
        fig = create_hitter_report(df_b, player, ncols=3)
        if fig:
            st.pyplot(fig=fig)
