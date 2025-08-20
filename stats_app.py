# d1_stats_app.py
import os
import base64
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="D1 Baseball — Team Stats",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("client.showErrorDetails", True)

DATA_PATH  = "B10C25_streamlit_streamlit_columns.csv"
BANNER_IMG = "NCAA_Baseball.jpg"   # ← new banner image

# ── Banner ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_banner_b64() -> str | None:
    if not os.path.exists(BANNER_IMG):
        return None
    with open(BANNER_IMG, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def hero_banner(title: str, *, subtitle: str | None = None, height_px: int = 260):
    b64 = load_banner_b64()
    bg_url = f"data:image/jpeg;base64,{b64}" if b64 else ""
    sub_html = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <style>
        .hero-wrap {{
            position: relative; width: 100%; height: {height_px}px;
            border-radius: 10px; overflow: hidden; margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }}
        .hero-bg {{
            position: absolute; inset: 0;
            background:
              linear-gradient(to bottom, rgba(0,0,0,0.45), rgba(0,0,0,0.60)),
              url('{bg_url}');
            background-size: cover; background-position: center;
            filter: saturate(105%);
        }}
        .hero-text {{
            position: absolute; inset: 0; display: flex;
            align-items: center; justify-content: center; flex-direction: column;
            color: #fff; text-align: center;
        }}
        .hero-title {{
            font-size: 40px; font-weight: 800; letter-spacing: .5px;
            text-shadow: 0 2px 8px rgba(0,0,0,.45); margin: 0;
        }}
        .hero-sub {{ font-size: 18px; font-weight: 600; opacity: .95; margin-top: 6px; }}
        </style>
        <div class="hero-wrap">
          <div class="hero-bg"></div>
          <div class="hero-text">
            <h1 class="hero-title">{title}</h1>
            {sub_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

hero_banner("D1 Baseball", subtitle=None, height_px=260)

# ── Date helpers / loader ─────────────────────────────────────────────────────
DATE_CANDIDATES = ["Date","date","GameDate","GAME_DATE","Game Date","date_game",
                   "Datetime","DateTime","game_datetime","GameDateTime"]

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    found = None
    lower = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand.lower() in lower:
            found = lower[cand.lower()]; break
    if found is None:
        df["Date"] = pd.NaT; return df
    dt = pd.to_datetime(df[found], errors="coerce")
    df["Date"] = pd.to_datetime(dt.dt.date, errors="coerce")
    return df

@st.cache_data(show_spinner=True)
def load_csv_norm(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    return ensure_date_column(df)

def filter_by_month_day(df, date_col="Date", months=None, days=None):
    if date_col not in df.columns or df.empty:
        return df
    s = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if months: mask &= s.dt.month.isin(months)
    if days:   mask &= s.dt.day.isin(days)
    return df[mask]

if not os.path.exists(DATA_PATH):
    st.error(f"Data not found at {DATA_PATH}"); st.stop()
df_all = load_csv_norm(DATA_PATH)

# ── Conference/team maps ──────────────────────────────────────────────────────
BIG_TEN_MAP = {
    'ILL_ILL': 'Illinois','MIC_SPA': 'Michigan State','UCLA': 'UCLA','IOW_HAW': 'Iowa',
    'IU': 'Indiana','MAR_TER': 'Maryland','MIC_WOL': 'Michigan','MIN_GOL': 'Minnesota',
    'NEB': 'Nebraska','NOR_CAT': 'Northwestern','ORE_DUC': 'Oregon','OSU_BUC': 'Ohio State',
    'PEN_NIT': 'Penn State','PUR_BOI': 'Purdue','RUT_SCA': 'Rutgers','SOU_TRO': 'USC','WAS_HUS': 'Washington'
}
BIG_12_MAP, SEC_MAP, ACC_MAP = {}, {}, {}
CONF_MAP = {"Big Ten": BIG_TEN_MAP, "Big 12": BIG_12_MAP, "SEC": SEC_MAP, "ACC": ACC_MAP}

# ── Stats builders ────────────────────────────────────────────────────────────
DISPLAY_COLS_H = ['Team','Batter','PA','AB','Hits','2B','3B','HR','HBP','BB','K','BA','OBP','SLG','OPS']
RATE_COLS_H    = ['BA','OBP','SLG','OPS']

def compute_hitter_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pr = df['PlayResult'].astype(str).str.lower()
    ab_values  = {'single','double','triple','homerun','out','error','fielderschoice'}
    hit_values = {'single','double','triple','homerun'}

    df['is_ab']   = pr.isin(ab_values).astype(int)
    df['is_hit']  = pr.isin(hit_values).astype(int)
    df['is_bb']   = df['KorBB'].astype(str).str.lower().eq('walk').astype(int)
    df['is_k']    = df['KorBB'].astype(str).str.contains('strikeout', case=False, na=False).astype(int)
    df['is_hbp']  = df['PitchCall'].astype(str).eq('HitByPitch').astype(int)
    df['is_sf']   = df['PlayResult'].astype(str).str.contains('Sacrifice', case=False, na=False).astype(int)
    df['is_1b']   = pr.eq('single').astype(int)
    df['is_2b']   = pr.eq('double').astype(int)
    df['is_3b']   = pr.eq('triple').astype(int)
    df['is_hr']   = pr.eq('homerun').astype(int)
    df['is_pa'] = (df['is_ab'] | df['is_bb'] | df['is_hbp'] | df['is_sf']).astype(int)

    agg = (df.groupby(['BatterTeam','Batter'], as_index=False)
             .agg(PA=('is_pa','sum'), AB=('is_ab','sum'), Hits=('is_hit','sum'),
                  Doubles=('is_2b','sum'), Triples=('is_3b','sum'), Homeruns=('is_hr','sum'),
                  HBP=('is_hbp','sum'), BB=('is_bb','sum'), K=('is_k','sum'),
                  Singles=('is_1b','sum'), SF=('is_sf','sum')))
    agg['TB'] = agg['Singles'] + 2*agg['Doubles'] + 3*agg['Triples'] + 4*agg['Homeruns']
    agg = agg.rename(columns={'Doubles':'2B', 'Triples':'3B', 'Homeruns':'HR'})

    with np.errstate(divide='ignore', invalid='ignore'):
        ba  = np.where(agg['AB'] > 0, agg['Hits'] / agg['AB'], 0.0)
        slg = np.where(agg['AB'] > 0, agg['TB'] / agg['AB'], 0.0)
        obp_den = agg['AB'] + agg['BB'] + agg['HBP'] + agg['SF']
        obp_num = agg['Hits'] + agg['BB'] + agg['HBP']
        obp = np.divide(obp_num, obp_den, out=np.zeros_like(obp_den, dtype=float), where=obp_den > 0)
        ops = obp + slg

    agg['BA_num'], agg['OBP_num'], agg['SLG_num'], agg['OPS_num'] = ba, obp, slg, ops
    agg['BA']  = [f"{x:.3f}" for x in ba]
    agg['OBP'] = [f"{x:.3f}" for x in obp]
    agg['SLG'] = [f"{x:.3f}" for x in slg]
    agg['OPS'] = [f"{x:.3f}" for x in ops]
    agg = agg.rename(columns={'BatterTeam':'Team'})
    team_map_all = {**BIG_TEN_MAP, **BIG_12_MAP, **SEC_MAP, **ACC_MAP}
    agg['Team'] = agg['Team'].replace(team_map_all)

    keep = DISPLAY_COLS_H + [c+'_num' for c in RATE_COLS_H]
    return agg[keep].sort_values('BA_num', ascending=False)

DISPLAY_COLS_P = ['Team','Name','IP','Hits','HR','BB','HBP','SO','WHIP','BB/9','H/9','HR/9','SO/9']
RATE_NUMS_P    = ['IP_num','WHIP_num','BB/9_num','H/9_num','HR/9_num','SO/9_num']

def compute_pitcher_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pr_low  = df['PlayResult'].astype(str).str.lower()
    korbb_s = df['KorBB'].astype(str)
    pitch_s = df['PitchCall'].astype(str)

    hit_values    = {'single','double','triple','homerun'}
    df['is_hit']  = pr_low.isin(hit_values).astype(int)
    df['is_hr']   = pr_low.eq('homerun').astype(int)
    df['is_bb']   = korbb_s.str.lower().eq('walk').astype(int)
    df['is_so']   = korbb_s.str.contains('strikeout', case=False, na=False).astype(int)
    df['is_hbp']  = pitch_s.eq('HitByPitch').astype(int)
    df['is_sf']   = df['PlayResult'].astype(str).str.contains('Sacrifice', case=False, na=False).astype(int)

    if 'OutsOnPlay' in df.columns:
        outs_on_play = pd.to_numeric(df['OutsOnPlay'], errors='coerce').fillna(0).astype(int)
        outs_from_k  = df['is_so'] * (outs_on_play == 0)
        df['outs']   = outs_on_play + outs_from_k
    else:
        is_out = pr_low.eq('out').astype(int)
        df['outs'] = is_out + df['is_sf'] + df['is_so']

    grouped = (df.groupby(['PitcherTeam','Pitcher'], as_index=False)
                 .agg(Hits=('is_hit','sum'),
                      HR=('is_hr','sum'),
                      BB=('is_bb','sum'),
                      HBP=('is_hbp','sum'),
                      SO=('is_so','sum'),
                      Outs=('outs','sum')))
    grouped['IP_num'] = grouped['Outs'] / 3.0
    def ip_display(ip_num: float) -> str:
        outs = int(round(ip_num * 3)); return f"{outs//3}.{outs%3}"
    grouped['IP'] = grouped['IP_num'].apply(ip_display)

    ip = grouped['IP_num'].replace(0, np.nan)
    grouped['WHIP_num'] = ((grouped['BB'] + grouped['Hits']) / ip).fillna(0.0)
    grouped['BB/9_num'] = (grouped['BB']  / ip * 9).fillna(0.0)
    grouped['H/9_num']  = (grouped['Hits']/ ip * 9).fillna(0.0)
    grouped['HR/9_num'] = (grouped['HR']  / ip * 9).fillna(0.0)
    grouped['SO/9_num'] = (grouped['SO']  / ip * 9).fillna(0.0)

    grouped['WHIP'] = grouped['WHIP_num'].map(lambda x: f"{x:.3f}")
    grouped['BB/9'] = grouped['BB/9_num'].map(lambda x: f"{x:.2f}")
    grouped['H/9']  = grouped['H/9_num'].map(lambda x: f"{x:.2f}")
    grouped['HR/9'] = grouped['HR/9_num'].map(lambda x: f"{x:.2f}")
    grouped['SO/9'] = grouped['SO/9_num'].map(lambda x: f"{x:.2f}")

    grouped = grouped.rename(columns={'PitcherTeam':'Team', 'Pitcher':'Name'})
    team_map_all = {**BIG_TEN_MAP, **BIG_12_MAP, **SEC_MAP, **ACC_MAP}
    grouped['Team'] = grouped['Team'].map(team_map_all).fillna(grouped['Team'])

    grouped = grouped.sort_values('WHIP_num', ascending=True)
    out = grouped[DISPLAY_COLS_P].copy()
    out = out.join(grouped[RATE_NUMS_P])
    return out

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("### Team Statistics")

col1, col2, col3 = st.columns([1,1,2])
conference = col1.selectbox("Conference", ["Big Ten","Big 12","SEC","ACC"], index=0)
team_map = {"Big Ten": BIG_TEN_MAP, "Big 12": {}, "SEC": {}, "ACC": {}}.get(conference, {})
present_codes = pd.unique(
    pd.concat(
        [df_all.get('BatterTeam', pd.Series(dtype=object)),
         df_all.get('PitcherTeam', pd.Series(dtype=object))],
        ignore_index=True
    ).dropna()
)
codes_in_conf = [code for code in team_map.keys() if code in present_codes]
if not codes_in_conf:
    codes_in_conf = sorted(present_codes.tolist())
    team_map = {code: code for code in codes_in_conf}
team_display = [team_map.get(code, code) for code in codes_in_conf]
team_sel_name = col2.selectbox("Team", team_display) if team_display else None

team_code = None
if team_sel_name:
    for code, name in [(c, team_map.get(c, c)) for c in codes_in_conf]:
        if name == team_sel_name:
            team_code = code; break

stats_type = col3.radio("Stats Type", ["Hitter Statistics","Pitcher Statistics"], index=0, horizontal=True)

all_dates = df_all['Date'].dropna()
month_options = sorted(all_dates.dt.month.unique().tolist())
day_options   = list(range(1,32))
fm, fd = st.columns(2)
months_sel = fm.multiselect("Months (optional)", options=month_options,
                            format_func=lambda n: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][n-1])
days_sel   = fd.multiselect("Days (optional)", options=day_options)

if not team_code:
    st.info("Choose a conference and team."); st.stop()

team_df = df_all[(df_all['BatterTeam'] == team_code) if stats_type=="Hitter Statistics"
                 else (df_all['PitcherTeam'] == team_code)].copy()
team_df = filter_by_month_day(team_df, months=months_sel, days=days_sel)
if team_df.empty:
    st.info("No rows after applying the selected filters."); st.stop()

if stats_type == "Hitter Statistics":
    ranked_h = compute_hitter_rates(team_df)
    st.dataframe(ranked_h[['Team','Batter','PA','AB','Hits','2B','3B','HR','HBP','BB','K','BA','OBP','SLG','OPS']],
                 use_container_width=True)
else:
    table_p = compute_pitcher_table(team_df)
    st.dataframe(table_p[['Team','Name','IP','Hits','HR','BB','HBP','SO','WHIP','BB/9','H/9','HR/9','SO/9']],
                 use_container_width=True)
