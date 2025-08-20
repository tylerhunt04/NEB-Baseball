# d1_app.py

import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="D1 Baseball Stats", initial_sidebar_state="expanded")

DATA_PATH = "B10C25_streamlit_streamlit_columns.csv"

DATE_CANDIDATES = ["Date","date","GameDate","GAME_DATE","Game Date","date_game","Datetime","DateTime","game_datetime","GameDateTime"]

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    found = None
    m = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand.lower() in m:
            found = m[cand.lower()]
            break
    if found is None:
        df["Date"] = pd.NaT
        return df
    dt = pd.to_datetime(df[found], errors="coerce")
    df["Date"] = pd.to_datetime(dt.dt.date, errors="coerce")
    return df

def filter_by_month_day(df, date_col="Date", months=None, days=None):
    if date_col not in df.columns or df.empty:
        return df
    s = pd.to_datetime(df[date_col], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if months:
        mask &= s.dt.month.isin(months)
    if days:
        mask &= s.dt.day.isin(days)
    return df[mask]

BIG_TEN_MAP = {
    'ILL_ILL': 'Illinois','MIC_SPA': 'Michigan State','UCLA': 'UCLA','IOW_HAW': 'Iowa',
    'IU': 'Indiana','MAR_TER': 'Maryland','MIC_WOL': 'Michigan','MIN_GOL': 'Minnesota',
    'NEB': 'Nebraska','NOR_CAT': 'Northwestern','ORE_DUC': 'Oregon','OSU_BUC': 'Ohio State',
    'PEN_NIT': 'Penn State','PUR_BOI': 'Purdue','RUT_SCA': 'Rutgers','SOU_TRO': 'USC','WAS_HUS': 'Washington'
}
BIG_12_MAP, SEC_MAP, ACC_MAP = {}, {}, {}
CONF_MAP = {"Big Ten": BIG_TEN_MAP, "Big 12": BIG_12_MAP, "SEC": SEC_MAP, "ACC": ACC_MAP}

DISPLAY_COLS_H = ['Team','Batter','PA','AB','Hits','2B','3B','HR','HBP','BB','K','BA','OBP','SLG','OPS']
RATE_COLS_H    = ['BA','OBP','SLG','OPS']
DISPLAY_COLS_P = ['Team','Name','IP','Hits','HR','BB','HBP','SO','WHIP','BB/9','H/9','HR/9','SO/9']
RATE_NUMS_P    = ['IP_num','WHIP_num','BB/9_num','H/9_num','HR/9_num','SO/9_num']

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

@st.cache_data(show_spinner=True)
def load_csv_norm(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    return ensure_date_column(df)

if not os.path.exists(DATA_PATH):
    st.error(f"Data not found at {DATA_PATH}")
    st.stop()
df_all = load_csv_norm(DATA_PATH)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.caption("Use the ◀ chevron to hide/show.")
    present_codes = pd.unique(pd.concat(
        [df_all.get('BatterTeam', pd.Series(dtype=object)),
         df_all.get('PitcherTeam', pd.Series(dtype=object))],
        ignore_index=True
    ).dropna())

    st.selectbox("Conference", ["Big Ten","Big 12","SEC","ACC"], index=0, key="d1_conference")
    team_map = CONF_MAP.get(st.session_state["d1_conference"], {})
    codes_in_conf = [code for code in team_map.keys() if code in present_codes]
    if not codes_in_conf:
        codes_in_conf = sorted(present_codes.tolist())
        team_map = {code: code for code in codes_in_conf}

    team_options = [(code, team_map.get(code, code)) for code in codes_in_conf]
    team_display = [name for _, name in team_options]
    team_sel_name = st.selectbox("Team", team_display, key="d1_team_name") if team_display else None

    team_code = None
    if team_sel_name:
        for code, name in team_options:
            if name == team_sel_name:
                team_code = code
                break
    st.session_state['d1_team_code'] = team_code

    MONTH_CHOICES = [(1,"January"),(2,"February"),(3,"March"),(4,"April"),(5,"May"),(6,"June"),
                     (7,"July"),(8,"August"),(9,"September"),(10,"October"),(11,"November"),(12,"December")]
    MONTH_NAME_BY_NUM = {n:name for n,name in MONTH_CHOICES}

    st.multiselect("Months (optional)", options=[n for n,_ in MONTH_CHOICES],
                   format_func=lambda n: MONTH_NAME_BY_NUM[n], default=[], key="d1_months")
    st.multiselect("Days (optional)", options=list(range(1,32)), default=[], key="d1_days")

    st.radio("Stats Type", ["Hitter Statistics","Pitcher Statistics"], index=0, key="d1_stats_type")

st.title("D1 Baseball")

team_code  = st.session_state.get('d1_team_code')
months_sel = st.session_state.get('d1_months', [])
days_sel   = st.session_state.get('d1_days', [])
stats_type = st.session_state.get('d1_stats_type', "Hitter Statistics")

if not team_code:
    st.warning("Choose a conference and team in the Filters drawer.")
    st.stop()

if stats_type == "Hitter Statistics":
    team_df = df_all[df_all['BatterTeam'] == team_code].copy()
else:
    team_df = df_all[df_all['PitcherTeam'] == team_code].copy()

team_df = filter_by_month_day(team_df, months=months_sel, days=days_sel)
if team_df.empty:
    st.info("No rows after applying the selected filters.")
    st.stop()

if stats_type == "Hitter Statistics":
    ranked_h = compute_hitter_rates(team_df)
    display_cols = DISPLAY_COLS_H + [c+'_num' for c in RATE_COLS_H]
    st.dataframe(ranked_h[display_cols], use_container_width=True)
else:
    table_p = compute_pitcher_table(team_df)
    display_cols_p = DISPLAY_COLS_P + RATE_NUMS_P
    st.dataframe(table_p[display_cols_p], use_container_width=True)
