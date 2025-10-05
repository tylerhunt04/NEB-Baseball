

import os
import json
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st

# =============== CONFIG & PATHS =================================================
APP_NAME = "LifeHub"
DATA_DIR = "lifehub_data"

FILES = {
    "schedule": os.path.join(DATA_DIR, "schedule.csv"),
    "workout_plan": os.path.join(DATA_DIR, "workout_plan.csv"),
    "workout_log": os.path.join(DATA_DIR, "workout_log.csv"),
    "weighins": os.path.join(DATA_DIR, "weighins.csv"),
    "journal": os.path.join(DATA_DIR, "journal.csv"),
    "habits": os.path.join(DATA_DIR, "habits.csv"),
    "settings": os.path.join(DATA_DIR, "settings.json"),
    "goals": os.path.join(DATA_DIR, "goals.csv"),
    # Finance datasets (manual)
    "accounts": os.path.join(DATA_DIR, "accounts.csv"),
    "transactions": os.path.join(DATA_DIR, "transactions.csv"),
    "paychecks": os.path.join(DATA_DIR, "paychecks.csv"),
    "budget": os.path.join(DATA_DIR, "budget.csv"),
}

DEFAULT_SETTINGS = {
    "units": "imperial",          # or "metric" (used for weigh-in label)
    "default_week_start": "Monday",
    "goal_weight": None,          # keeps To-goal metric in Weigh-ins
    "mission": "",                # mission text shown under header

    # Finance defaults
    "pay_frequency_days": 14,     # bi-weekly
    "last_pay_date": "",          # ISO date string
    "default_deposit_account": "Checking",
}

os.makedirs(DATA_DIR, exist_ok=True)

# =============== UTIL: LOAD / SAVE =============================================
def _ensure_csv(path: str, columns: List[str]):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def load_csv(path: str, columns: List[str]) -> pd.DataFrame:
    _ensure_csv(path, columns)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=columns)
    # fill missing columns if schema evolves
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[columns]

def save_csv(path: str, df: pd.DataFrame):
    df.to_csv(path, index=False)

def load_settings() -> Dict[str, Any]:
    if not os.path.exists(FILES["settings"]):
        with open(FILES["settings"], "w") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2)
        return DEFAULT_SETTINGS.copy()
    try:
        with open(FILES["settings"], "r") as f:
            s = json.load(f)
    except Exception:
        s = DEFAULT_SETTINGS.copy()
    # backfill keys
    for k, v in DEFAULT_SETTINGS.items():
        if k not in s:
            s[k] = v
    return s

def save_settings(s: Dict[str, Any]):
    with open(FILES["settings"], "w") as f:
        json.dump(s, f, indent=2)

# =============== DATA SCHEMAS ===================================================
SCHEDULE_COLS = [
    "id", "date", "start_time", "end_time", "title", "category", "priority", "notes", "status"
]
WORKOUT_PLAN_COLS = [
    "id", "day_of_week", "block", "exercise", "target_sets", "target_reps", "target_load", "notes"
]
WORKOUT_LOG_COLS = [
    "id", "date", "exercise", "set_number", "reps", "load", "rpe", "notes"
]
WEIGHIN_COLS = [
    "id", "date", "weight", "body_fat_pct", "waist", "notes"
]
JOURNAL_COLS = [
    "id", "date", "title", "mood", "tags", "content"
]
HABIT_COLS = [
    "id", "name", "weekly_target", "sun", "mon", "tue", "wed", "thu", "fri", "sat", "notes"
]
GOALS_COLS = [
    "id", "title", "category", "timeframe", "target_date", "status", "progress", "notes"
]

# Finance schemas
ACCOUNT_COLS = ["id", "name", "type", "opening_balance", "notes"]
TRANSACTION_COLS = ["id", "date", "account", "category", "description", "amount", "status", "notes"]
PAYCHECK_COLS = ["id", "date", "net_amount", "hours", "rate", "account", "notes"]
BUDGET_COLS = ["id", "category", "percent", "auto_account", "notes"]

# =============== SESSION STATE ==================================================
if "settings" not in st.session_state:
    st.session_state.settings = load_settings()

# =============== PAGE CONFIG & LIGHT THEME ENFORCE ==============================
st.set_page_config(page_title=f"{APP_NAME} — Self Management", layout="wide", initial_sidebar_state="collapsed")
# Force light-like appearance + hide sidebar
st.markdown(
    """
    <style>
    :root {
      --background-color: #ffffff;
      --secondary-background-color: #f6f6f6;
      --text-color: #111111;
    }
    [data-testid="stAppViewContainer"] {background: var(--background-color) !important;}
    [data-testid="stHeader"] {background: var(--background-color) !important;}
    [data-testid="stSidebar"] {display: none !important;}
    .stMarkdown, .stText, .stDataFrame, .stMetric { color: var(--text-color) !important; }
    .mission-box { background:#f6f6f6; border:1px solid #eee; padding:12px 14px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============== HEADER =========================================================
# Your name larger; Be Intentional smaller; mission displayed right below.
st.title("Tyler Hunt")                     # bigger
st.subheader("Be Intentional")             # smaller

# Show mission statement below the header (edit via Mission & Goals tab)
settings = st.session_state.settings
mission_text = settings.get("mission", "").strip()
if mission_text:
    st.markdown(f'<div class="mission-box">{mission_text}</div>', unsafe_allow_html=True)
else:
    st.info("Add your mission below (Mission & Goals tab) so it always appears here.")

# ---- TOP NAV TABS ----
(
    tab_dashboard,
    tab_schedule,
    tab_workouts,
    tab_weighins,
    tab_journal,
    tab_habits,
    tab_finance,          # NEW
    tab_mission_goals,
) = st.tabs([
    "Dashboard",
    "Schedule",
    "Workouts",
    "Weigh-ins",
    "Journal",
    "Habits",
    "Finance",            # NEW
    "Mission & Goals",
])

# Quick Add lives on the Dashboard
with tab_dashboard:
    with st.expander("➕ Quick Add", expanded=False):
        qa_choice = st.selectbox("Type", ["Task", "Weigh-in", "Journal"], key="qa_type_top")
        if qa_choice == "Task":
            qa_date = st.date_input("Date", value=date.today(), key="qa_task_date_top")
            qa_title = st.text_input("Title", key="qa_task_title_top")
            col1, col2 = st.columns(2)
            with col1:
                qa_start = st.time_input("Start", value=datetime.now().time(), key="qa_task_start_top")
            with col2:
                qa_end = st.time_input("End", value=(datetime.now() + timedelta(hours=1)).time(), key="qa_task_end_top")
            qa_cat = st.selectbox("Category", ["Work", "Study", "Health", "Personal", "Other"], index=2, key="qa_task_cat_top")
            qa_pri = st.selectbox("Priority", ["Low", "Med", "High"], index=1, key="qa_task_pri_top")
            if st.button("Add Task", use_container_width=True, key="qa_add_task_top"):
                df = load_csv(FILES["schedule"], SCHEDULE_COLS)
                new = {
                    "id": str(uuid.uuid4()),
                    "date": qa_date.isoformat(),
                    "start_time": qa_start.strftime("%H:%M"),
                    "end_time": qa_end.strftime("%H:%M"),
                    "title": qa_title,
                    "category": qa_cat,
                    "priority": qa_pri,
                    "notes": "",
                    "status": "Todo",
                }
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                save_csv(FILES["schedule"], df)
                st.success("Task added.")
        elif qa_choice == "Weigh-in":
            wi_date = st.date_input("Date", value=date.today(), key="qa_wi_date_top")
            wi_weight = st.number_input("Weight", min_value=0.0, step=0.1, key="qa_wi_weight_top")
            wi_bf = st.number_input("Body Fat %", min_value=0.0, max_value=100.0, step=0.1, key="qa_wi_bf_top")
            if st.button("Add Weigh-in", use_container_width=True, key="qa_add_wi_top"):
                df = load_csv(FILES["weighins"], WEIGHIN_COLS)
                new = {
                    "id": str(uuid.uuid4()),
                    "date": wi_date.isoformat(),
                    "weight": wi_weight,
                    "body_fat_pct": wi_bf if wi_bf else np.nan,
                    "waist": np.nan,
                    "notes": "",
                }
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                save_csv(FILES["weighins"], df)
                st.success("Weigh-in logged.")
        else:
            j_date = st.date_input("Date", value=date.today(), key="qa_j_date_top")
            j_title = st.text_input("Title", key="qa_j_title_top")
            j_mood = st.slider("Mood", 1, 10, 6, key="qa_j_mood_top")
            j_tags = st.text_input("Tags (comma-sep)", key="qa_j_tags_top")
            j_content = st.text_area("Entry", height=120, key="qa_j_content_top")
            if st.button("Save Journal", use_container_width=True, key="qa_add_journal_top"):
                df = load_csv(FILES["journal"], JOURNAL_COLS)
                new = {
                    "id": str(uuid.uuid4()),
                    "date": j_date.isoformat(),
                    "title": j_title,
                    "mood": j_mood,
                    "tags": j_tags,
                    "content": j_content,
                }
                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                save_csv(FILES["journal"], df)
                st.success("Journal saved.")

# =============== HELPERS ========================================================
def _week_bounds(d: date, week_start: str = "Monday"):
    weekday_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(week_start)
    current = d.weekday()
    delta = (current - weekday_idx) % 7
    start = d - timedelta(days=delta)
    end = start + timedelta(days=6)
    return start, end

def _human_duration(start_str: str, end_str: str) -> float:
    try:
        s = datetime.strptime(start_str, "%H:%M")
        e = datetime.strptime(end_str, "%H:%M")
        if e < s:
            e += timedelta(days=1)
        return (e - s).total_seconds() / 3600.0
    except Exception:
        return 0.0

# =============== PAGES ==========================================================

# ---- DASHBOARD ----
with tab_dashboard:
    st.subheader("Today")
    today = date.today().isoformat()

    # Today tasks
    sched = load_csv(FILES["schedule"], SCHEDULE_COLS)
    today_tasks = sched[sched["date"] == today].copy()
    today_tasks["hours"] = today_tasks.apply(lambda r: _human_duration(r["start_time"], r["end_time"]), axis=1)

    # Today workout plan (day_of_week)
    dow = date.today().strftime("%A")
    plan = load_csv(FILES["workout_plan"], WORKOUT_PLAN_COLS)
    today_plan = plan[plan["day_of_week"] == dow].copy()

    # Weigh-ins stats
    wi = load_csv(FILES["weighins"], WEIGHIN_COLS)
    wi_sorted = wi.dropna(subset=["weight"]).sort_values("date")
    latest_weight = wi_sorted["weight"].iloc[-1] if len(wi_sorted) else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Tasks today", int(today_tasks.shape[0]))
    with c2:
        st.metric("Planned workout blocks", int(today_plan.shape[0]))
    with c3:
        st.metric("Latest weight", f"{latest_weight:.1f}" if latest_weight else "—")

    st.markdown("### Today's Schedule")
    if today_tasks.empty:
        st.info("No tasks for today yet. Use Quick Add above or the Schedule tab.")
    else:
        st.dataframe(
            today_tasks[["start_time", "end_time", "title", "category", "priority", "status", "notes"]]
            .sort_values("start_time"),
            use_container_width=True,
        )

    st.markdown("### Today's Workout Plan")
    if today_plan.empty:
        st.info("No workout blocks planned for today. Add some on the Workouts tab.")
    else:
        st.dataframe(today_plan[["block", "exercise", "target_sets", "target_reps", "target_load", "notes"]], use_container_width=True)

    st.markdown("### Recent Weigh-ins")
    if wi_sorted.empty:
        st.info("Log your first weigh-in on the Weigh-ins tab or via Quick Add.")
    else:
        st.line_chart(wi_sorted.set_index("date")["weight"], height=220)

    # ---- GOALS SUMMARY (UNDER DASHBOARD) ----
    st.markdown("### Goals (Summary)")
    goals_df = load_csv(FILES["goals"], GOALS_COLS)
    with st.expander("➕ Quick add goal"):
        g_title = st.text_input("Title", key="dash_goal_title")
        c1, c2, c3 = st.columns(3)
        with c1:
            g_cat = st.selectbox("Category", ["Health", "Career", "School", "Finance", "Personal", "Other"], key="dash_goal_cat")
        with c2:
            g_timeframe = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly", "Long-term"], key="dash_goal_timeframe")
        with c3:
            g_target = st.date_input("Target date", value=date.today(), key="dash_goal_target")
        g_notes = st.text_area("Notes", key="dash_goal_notes")
        if st.button("Add Goal", key="dash_add_goal", type="primary") and g_title:
            new = {
                "id": str(uuid.uuid4()),
                "title": g_title,
                "category": g_cat,
                "timeframe": g_timeframe,
                "target_date": g_target.isoformat(),
                "status": "Active",
                "progress": 0,
                "notes": g_notes,
            }
            goals_df = pd.concat([goals_df, pd.DataFrame([new])], ignore_index=True)
            save_csv(FILES["goals"], goals_df)
            st.success("Goal added.")

    # show top/active goals
    if goals_df.empty:
        st.info("No goals yet.")
    else:
        view = goals_df.copy().sort_values(["status", "target_date"])
        st.dataframe(view[["title","category","timeframe","target_date","status","progress"]].head(10), use_container_width=True)

# ---- SCHEDULE ----
with tab_schedule:
    st.subheader("Schedule Builder")

    sched = load_csv(FILES["schedule"], SCHEDULE_COLS)

    col_a, col_b, col_c = st.columns([1,1,2])
    with col_a:
        d = st.date_input("Date", value=date.today(), key="schedule_date")
    with col_b:
        week_start = st.selectbox("Week starts", ["Monday", "Sunday"], index=0, key="schedule_week_start")
    with col_c:
        show_week = st.toggle("Show week overview", value=True)

    # Add task form (inline)
    with st.expander("➕ Add a task"):
        c1, c2 = st.columns(2)
        with c1:
            start_t = st.time_input("Start", value=datetime.now().time(), key="sch_start")
        with c2:
            end_t = st.time_input("End", value=(datetime.now() + timedelta(hours=1)).time(), key="sch_end")
        t_title = st.text_input("Title", key="sch_title")
        c3, c4, c5 = st.columns([1,1,2])
        with c3:
            t_cat = st.selectbox("Category", ["Work", "Study", "Health", "Personal", "Other"], index=2, key="schedule_task_cat")
        with c4:
            t_pri = st.selectbox("Priority", ["Low", "Med", "High"], index=1, key="schedule_task_pri")
        with c5:
            t_status = st.selectbox("Status", ["Todo", "Doing", "Done"], index=0, key="schedule_task_status")
        t_notes = st.text_area("Notes", height=80)
        if st.button("Add to schedule", type="primary"):
            new = {
                "id": str(uuid.uuid4()),
                "date": d.isoformat(),
                "start_time": start_t.strftime("%H:%M"),
                "end_time": end_t.strftime("%H:%M"),
                "title": t_title,
                "category": t_cat,
                "priority": t_pri,
                "notes": t_notes,
                "status": t_status,
            }
            sched = pd.concat([sched, pd.DataFrame([new])], ignore_index=True)
            save_csv(FILES["schedule"], sched)
            st.success("Task added.")

    # Day view
    st.markdown("### Day View")
    day_df = sched[sched["date"] == d.isoformat()].copy()
    if day_df.empty:
        st.info("No tasks for this day yet.")
    else:
        day_df = day_df.sort_values("start_time")
        edited = st.data_editor(
            day_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.TextColumn("id", disabled=True),
            },
        )
        if st.button("Save day changes"):
            # merge edits back
            others = sched[sched["date"] != d.isoformat()].copy()
            merged = pd.concat([others, edited], ignore_index=True)
            save_csv(FILES["schedule"], merged[SCHEDULE_COLS])
            st.success("Saved.")

    # Week overview
    if show_week:
        st.markdown("### Week Overview")
        start, end = _week_bounds(d, week_start)
        mask = (pd.to_datetime(sched["date"]).dt.date >= start) & (pd.to_datetime(sched["date"]).dt.date <= end)
        week_df = sched[mask].copy()
        if week_df.empty:
            st.info("No tasks for this week yet.")
        else:
            week_df["hours"] = week_df.apply(lambda r: _human_duration(r["start_time"], r["end_time"]), axis=1)
            agg = week_df.groupby(["date", "category"]).agg(
                tasks=("id", "count"), hours=("hours", "sum")
            ).reset_index()
            st.dataframe(agg.sort_values(["date", "category"]))

# ---- WORKOUTS ----
with tab_workouts:
    st.subheader("Workouts")

    plan = load_csv(FILES["workout_plan"], WORKOUT_PLAN_COLS)
    log = load_csv(FILES["workout_log"], WORKOUT_LOG_COLS)

    tab1, tab2, tab3 = st.tabs(["Plan Builder", "Workout Logger", "Progress"])

    with tab1:
        st.markdown("#### Weekly Plan Builder")
        st.caption("Create planned blocks per weekday. Blocks are arbitrary (e.g., AM/PM/Accessory)")
        if plan.empty:
            st.info("Start by adding rows with the plus button below.")
        edited = st.data_editor(
            plan if not plan.empty else pd.DataFrame(columns=WORKOUT_PLAN_COLS),
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.TextColumn("id", disabled=True),
                "day_of_week": st.column_config.SelectboxColumn(options=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]),
            },
        )
        if st.button("Save Plan", key="save_plan"):
            # ensure IDs
            if "id" in edited.columns:
                edited["id"] = edited["id"].fillna("").apply(lambda x: x if x else str(uuid.uuid4()))
            else:
                edited["id"] = [str(uuid.uuid4()) for _ in range(len(edited))]
            save_csv(FILES["workout_plan"], edited[WORKOUT_PLAN_COLS])
            st.success("Plan saved.")

    with tab2:
        st.markdown("#### Workout Logger")
        wl_date = st.date_input("Date", value=date.today(), key="log_date")
        wl_ex = st.text_input("Exercise", placeholder="e.g., Back Squat")
        c1, c2, c3 = st.columns(3)
        with c1:
            wl_set = st.number_input("Set #", 1, 20, 1)
        with c2:
            wl_reps = st.number_input("Reps", 0, 100, 5)
        with c3:
            wl_load = st.number_input("Load", 0.0, 2000.0, 135.0, step=5.0)
        wl_rpe = st.slider("RPE", 1.0, 10.0, 7.5, step=0.5)
        wl_notes = st.text_input("Notes", placeholder="Top set felt heavy …")
        if st.button("Log Set", type="primary"):
            new = {
                "id": str(uuid.uuid4()),
                "date": wl_date.isoformat(),
                "exercise": wl_ex,
                "set_number": wl_set,
                "reps": wl_reps,
                "load": wl_load,
                "rpe": wl_rpe,
                "notes": wl_notes,
            }
            log = pd.concat([log, pd.DataFrame([new])], ignore_index=True)
            save_csv(FILES["workout_log"], log)
            st.success("Logged.")

        st.markdown("#### Recent Sets")
        if log.empty:
            st.info("No sets logged yet.")
        else:
            st.dataframe(log.sort_values(["date", "exercise", "set_number"], ascending=[False, True, True]).tail(50), use_container_width=True)

    with tab3:
        st.markdown("#### Volume & Trends")
        if log.empty:
            st.info("Log workouts to see progress.")
        else:
            dfl = log.copy()
            dfl["volume"] = dfl["reps"].astype(float) * dfl["load"].astype(float)
            vol_daily = dfl.groupby("date")["volume"].sum().reset_index()
            st.line_chart(vol_daily.set_index("date")["volume"], height=240)

            st.markdown("##### By Exercise (Total Volume)")
            by_ex = dfl.groupby("exercise")["volume"].sum().sort_values(ascending=False).reset_index()
            st.dataframe(by_ex, use_container_width=True)

# ---- WEIGH-INS ----
with tab_weighins:
    st.subheader("Weekly Weigh-ins")

    settings = st.session_state.settings
    units = settings.get("units", "imperial")

    wi = load_csv(FILES["weighins"], WEIGHIN_COLS)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        wi_date = st.date_input("Date", value=date.today(), key="wi_date_main")
    with c2:
        weight = st.number_input("Weight" + (" (lb)" if units == "imperial" else " (kg)"), min_value=0.0, step=0.1, key="wi_weight_main")
    with c3:
        bf = st.number_input("Body Fat %", min_value=0.0, max_value=100.0, step=0.1, key="wi_bf_main")
    with c4:
        waist = st.number_input("Waist (in/cm)", min_value=0.0, step=0.1, key="wi_waist_main")
    notes = st.text_input("Notes", key="wi_notes_main")

    if st.button("Add Weigh-in", type="primary"):
        new = {
            "id": str(uuid.uuid4()),
            "date": wi_date.isoformat(),
            "weight": weight,
            "body_fat_pct": bf if bf else np.nan,
            "waist": waist if waist else np.nan,
            "notes": notes,
        }
        wi = pd.concat([wi, pd.DataFrame([new])], ignore_index=True)
        save_csv(FILES["weighins"], wi)
        st.success("Saved weigh-in.")

    st.markdown("### Trend")
    if wi.empty:
        st.info("Add weigh-ins to see your trend.")
    else:
        wi_sorted = wi.dropna(subset=["weight"]).sort_values("date")
        st.line_chart(wi_sorted.set_index("date")["weight"], height=260)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Latest", f"{wi_sorted['weight'].iloc[-1]:.1f}")
        with c2:
            delta_7 = wi_sorted.tail(7)["weight"].mean() - wi_sorted.head(7)["weight"].mean() if len(wi_sorted) >= 14 else np.nan
            st.metric("Δ (first 7 → last 7)", f"{delta_7:+.1f}" if not np.isnan(delta_7) else "—")
        with c3:
            goal = settings.get("goal_weight")
            latest = wi_sorted["weight"].iloc[-1]
            st.metric("To goal", f"{(latest - goal):+.1f}" if goal else "—")
# ---- JOURNAL ----
with tab_journal:
    st.subheader("Journal")

    jr = load_csv(FILES["journal"], JOURNAL_COLS)
    with st.form("journal_form"):
        j_date = st.date_input("Date", value=date.today(), key="journal_date_form")
        j_title = st.text_input("Title", placeholder="Daily reflection", key="journal_title_form")
        c1, c2 = st.columns([2,1])
        with c1:
            j_content = st.text_area("Content (Markdown supported)", height=200, key="journal_content_form")
        with c2:
            j_mood = st.slider("Mood", 1, 10, 6, key="journal_mood_form")
            j_tags = st.text_input("Tags (comma-sep)", key="journal_tags_form")
        submitted = st.form_submit_button("Save Entry", type="primary")
        if submitted:
            new = {
                "id": str(uuid.uuid4()),
                "date": j_date.isoformat(),
                "title": j_title,
                "mood": j_mood,
                "tags": j_tags,
                "content": j_content,
            }
            jr = pd.concat([jr, pd.DataFrame([new])], ignore_index=True)
            save_csv(FILES["journal"], jr)
            st.success("Saved.")

    st.markdown("### Entries")
    if jr.empty:
        st.info("No entries yet.")
    else:
        colf1, colf2 = st.columns(2)
        with colf1:
            tag_filter = st.text_input("Filter by tag contains", key="jr_filter_tag")
        with colf2:
            mood_min, mood_max = st.slider("Mood range", 1, 10, (1, 10), key="jr_filter_mood")
        view = jr.copy()
        if tag_filter:
            view = view[view["tags"].fillna("").str.contains(tag_filter, case=False)]
        view = view[(view["mood"].fillna(0) >= mood_min) & (view["mood"].fillna(0) <= mood_max)]
        view = view.sort_values("date", ascending=False)
        for _, r in view.iterrows():
            st.markdown(f"**{r['date']} — {r['title']}**  ")
            st.caption(f"Mood: {int(r['mood'])}  •  Tags: {r['tags'] or '—'}")
            st.markdown(r["content"] or "")
            st.divider()

# ---- HABITS ----
with tab_habits:
    st.subheader("Habit Tracker")

    hb = load_csv(FILES["habits"], HABIT_COLS)

    with st.expander("➕ Add a habit"):
        h_name = st.text_input("Habit name", placeholder="e.g., Read 20 min", key="hab_add_name")
        h_weekly = st.number_input("Weekly target (times)", min_value=1, max_value=21, value=5, key="hab_add_weekly")
        if st.button("Add Habit", key="hab_add_btn"):
            new = {
                "id": str(uuid.uuid4()),
                "name": h_name,
                "weekly_target": h_weekly,
                "sun": 0, "mon": 0, "tue": 0, "wed": 0, "thu": 0, "fri": 0, "sat": 0,
                "notes": "",
            }
            hb = pd.concat([hb, pd.DataFrame([new])], ignore_index=True)
            save_csv(FILES["habits"], hb)
            st.success("Habit added.")

    if hb.empty:
        st.info("Add a habit to begin tracking.")
    else:
        edited = st.data_editor(
            hb,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.TextColumn("id", disabled=True),
                "sun": st.column_config.NumberColumn(min_value=0),
                "mon": st.column_config.NumberColumn(min_value=0),
                "tue": st.column_config.NumberColumn(min_value=0),
                "wed": st.column_config.NumberColumn(min_value=0),
                "thu": st.column_config.NumberColumn(min_value=0),
                "fri": st.column_config.NumberColumn(min_value=0),
                "sat": st.column_config.NumberColumn(min_value=0),
            },
            key="hab_editor",
        )
        if st.button("Save Habits", key="hab_save_btn"):
            save_csv(FILES["habits"], edited[HABIT_COLS])
            st.success("Saved.")

        st.markdown("### Weekly Review")
        view = edited.copy() if isinstance(edited, pd.DataFrame) else hb.copy()
        days = ["sun","mon","tue","wed","thu","fri","sat"]
        view["done"] = view[days].sum(axis=1)
        view["progress"] = (view["done"] / view["weekly_target"]).clip(upper=1.0)
        st.dataframe(view[["name","weekly_target","done","progress","notes"]], use_container_width=True)

# ---- FINANCE ----
with tab_finance:
    st.subheader("Finance")
    import altair as alt

    # ---------- Ensure files/columns once (no output) ----------
    if "accounts" not in FILES:
        FILES.update({
            "accounts": os.path.join(DATA_DIR, "accounts.csv"),
            "transactions": os.path.join(DATA_DIR, "transactions.csv"),
            "paychecks": os.path.join(DATA_DIR, "paychecks.csv"),
            "budget": os.path.join(DATA_DIR, "budget.csv"),
        })
    if "ACCOUNT_COLS" not in globals():
        ACCOUNT_COLS = ["id", "name", "type", "opening_balance", "notes"]
    if "TRANSACTION_COLS" not in globals():
        TRANSACTION_COLS = ["id", "date", "account", "category", "description", "amount", "status", "notes"]
    if "PAYCHECK_COLS" not in globals():
        PAYCHECK_COLS = ["id", "date", "net_amount", "hours", "rate", "account", "notes"]
    if "BUDGET_COLS" not in globals():
        BUDGET_COLS = ["id", "category", "percent", "auto_account", "notes"]

    # ---------- Load data ----------
    accounts      = load_csv(FILES["accounts"], ACCOUNT_COLS)
    transactions  = load_csv(FILES["transactions"], TRANSACTION_COLS)
    paychecks     = load_csv(FILES["paychecks"], PAYCHECK_COLS)  # not displayed; just kept for history
    budget        = load_csv(FILES["budget"], BUDGET_COLS)

    # ---------- Core accounts ----------
    def ensure_account(df, name, acc_type):
        if df[df["name"] == name].empty:
            df = pd.concat([df, pd.DataFrame([{
                "id": str(uuid.uuid4()), "name": name, "type": acc_type,
                "opening_balance": 0.0, "notes": "",
            }])], ignore_index=True)
        return df

    if accounts.empty:
        accounts = pd.DataFrame(columns=ACCOUNT_COLS)
    accounts = ensure_account(accounts, "Checking", "bank")
    accounts = ensure_account(accounts, "Savings", "bank")
    accounts = ensure_account(accounts, "Credit Card", "credit")
    accounts["type"] = accounts["type"].fillna("bank").replace({"cash": "bank"})
    accounts["id"]   = accounts["id"].fillna("").apply(lambda x: x or str(uuid.uuid4()))
    save_csv(FILES["accounts"], accounts)

    acc_names    = accounts["name"].tolist()
    acc_type_map = dict(zip(accounts["name"], accounts["type"]))

    # ---------- Helpers ----------
    def compute_current_balances(acc_df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
        tx = tx_df.copy()
        if not tx.empty:
            tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
        sums = tx.groupby("account")["amount"].sum() if not tx.empty else pd.Series(dtype=float)
        rows = []
        for _, r in acc_df.iterrows():
            opening = float(r.get("opening_balance", 0) or 0)
            delta   = float(sums.get(r["name"], 0.0))
            rows.append({"Account": r["name"], "Balance": opening + delta})
        return pd.DataFrame(rows)

    # every-other-Thursday periods: start from "two weeks before tomorrow", then forward
    def next_or_same_weekday(d: date, weekday: int) -> date:  # Monday=0 ... Thu=3
        return d + timedelta(days=(weekday - d.weekday()) % 7)

    today = date.today()
    next_thu = next_or_same_weekday(today + timedelta(days=1), 3)
    first_start = next_thu - timedelta(days=14)
    period_starts = [first_start + timedelta(days=14 * i) for i in range(16)]
    periods = [(s, s + timedelta(days=13)) for s in period_starts]  # inclusive 14-day windows

    def fmt_period(p):
        s, e = p
        return f"{s.isoformat()} → {e.isoformat()}"

    def fmt_date_long(dval) -> str:
        dt = pd.to_datetime(dval, errors="coerce")
        if pd.isna(dt):
            return ""
        return f"{dt.strftime('%B')} {dt.day}, {dt.year}"  # e.g., "October 25, 2025"

    # ========================= TOP: Account Balances =========================
    st.markdown("### Account Balances")

    with st.expander("Initialize / adjust starting balances (one-time)"):
        init_cols = st.columns(3)
        for i, name in enumerate(["Checking", "Savings", "Credit Card"]):
            with init_cols[i]:
                cur = float(accounts.loc[accounts["name"] == name, "opening_balance"].iloc[0])
                new = st.number_input(f"{name} starting balance", value=cur, step=50.0, key=f"init_{name}")
                accounts.loc[accounts["name"] == name, "opening_balance"] = new
        if st.button("Save starting balances", key="fin_save_opening"):
            save_csv(FILES["accounts"], accounts)
            st.success("Starting balances saved.")

    balances_df = compute_current_balances(accounts, transactions)

    # Quick metrics
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Checking", f"${balances_df.loc[balances_df['Account']=='Checking','Balance'].sum():,.2f}")
    with c2: st.metric("Savings", f"${balances_df.loc[balances_df['Account']=='Savings','Balance'].sum():,.2f}")
    with c3: st.metric("Credit Card (owed)", f"${balances_df.loc[balances_df['Account']=='Credit Card','Balance'].sum():,.2f}")

    # Interactive bar (cap at $4,000; skinnier bars)
    bar = (
        alt.Chart(balances_df)
        .mark_bar(size=22)
        .encode(
            x=alt.X("Account:N", sort=None, title=""),
            y=alt.Y("Balance:Q", scale=alt.Scale(domain=[0, 4000]), title="Balance ($)"),
            tooltip=["Account:N", alt.Tooltip("Balance:Q", format="$,.2f")],
        )
        .properties(height=220)
        .interactive()
    )
    st.altair_chart(bar, use_container_width=True)

    st.divider()

    # ========================= TRANSACTIONS (form | table + filters) =========================
    st.markdown("### Transactions")
    col_tx_form, col_tx_table = st.columns([1, 1])

    # ------------ LEFT: Add transactions ------------
    with col_tx_form:
        flow = st.radio("Type", ["Expense", "Income", "Transfer"], horizontal=True, key="fin_tx_flow")

        if flow == "Expense":
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
            with c1: tx_date = st.date_input("Date", value=today, key="fin_tx_exp_date")
            with c2: from_acc = st.selectbox("From account", acc_names, index=0 if acc_names else None, key="fin_tx_exp_from")
            with c3: amount = st.number_input("Amount ($)", min_value=0.0, value=0.0, step=5.0, key="fin_tx_exp_amt")
            with c4: category = st.text_input("Category (e.g., Groceries)", key="fin_tx_exp_cat")
            notes = st.text_input("Notes", key="fin_tx_exp_notes")
            if st.button("Add Expense", key="fin_tx_exp_add", type="primary"):
                signed = -abs(amount) if acc_type_map.get(from_acc, "bank") != "credit" else +abs(amount)
                row = {
                    "id": str(uuid.uuid4()), "date": tx_date.isoformat(),
                    "account": from_acc, "category": category or "Expense", "description": "",
                    "amount": float(signed), "status": "Logged", "notes": notes or "",
                }
                transactions = pd.concat([transactions, pd.DataFrame([row])], ignore_index=True)
                save_csv(FILES["transactions"], transactions[TRANSACTION_COLS])
                st.success("Expense added.")

        elif flow == "Income":
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
            with c1: tx_date = st.date_input("Date", value=today, key="fin_tx_inc_date")
            with c2: to_acc = st.selectbox("To account", acc_names, index=0 if acc_names else None, key="fin_tx_inc_to")
            with c3: amount = st.number_input("Amount ($)", min_value=0.0, value=0.0, step=50.0, key="fin_tx_inc_amt")
            with c4: category = st.text_input("Category (e.g., Income: Paycheck)", value="Income: Paycheck", key="fin_tx_inc_cat")
            notes = st.text_input("Notes", key="fin_tx_inc_notes")
            if st.button("Add Income", key="fin_tx_inc_add", type="primary"):
                signed = +abs(amount) if acc_type_map.get(to_acc, "bank") != "credit" else -abs(amount)
                row = {
                    "id": str(uuid.uuid4()), "date": tx_date.isoformat(),
                    "account": to_acc, "category": category or "Income", "description": "",
                    "amount": float(signed), "status": "Logged", "notes": notes or "",
                }
                transactions = pd.concat([transactions, pd.DataFrame([row])], ignore_index=True)
                save_csv(FILES["transactions"], transactions[TRANSACTION_COLS])
                st.success("Income added.")

        else:  # Transfer
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
            with c1: tx_date = st.date_input("Date", value=today, key="fin_tx_trf_date")
            with c2: from_acc = st.selectbox("From", acc_names, index=0 if acc_names else None, key="fin_tx_trf_from")
            to_options = [a for a in acc_names if a != from_acc] or acc_names
            with c3: to_acc = st.selectbox("To", to_options, index=0 if to_options else None, key="fin_tx_trf_to")
            with c4: amount = st.number_input("Amount ($)", min_value=0.0, value=0.0, step=5.0, key="fin_tx_trf_amt")
            notes = st.text_input("Notes", value="Transfer", key="fin_tx_trf_notes")
            if st.button("Record Transfer", key="fin_tx_trf_add", type="primary"):
                amt_from = -abs(amount)
                amt_to   = +abs(amount) if acc_type_map.get(to_acc, "bank") != "credit" else -abs(amount)
                rows = [
                    {"id": str(uuid.uuid4()), "date": tx_date.isoformat(), "account": from_acc,
                     "category": "Transfer", "description": f"Transfer to {to_acc}", "amount": float(amt_from),
                     "status": "Logged", "notes": notes or ""},
                    {"id": str(uuid.uuid4()), "date": tx_date.isoformat(), "account": to_acc,
                     "category": "Transfer", "description": f"Transfer from {from_acc}", "amount": float(amt_to),
                     "status": "Logged", "notes": notes or ""},
                ]
                transactions = pd.concat([transactions, pd.DataFrame(rows)], ignore_index=True)
                save_csv(FILES["transactions"], transactions[TRANSACTION_COLS])
                st.success("Transfer recorded.")

    # ------------ RIGHT: Filters + table (no IDs visible, selectable delete) ------------
    with col_tx_table:
        st.markdown("#### All Transactions")

        # FILTERS (above table)
        tx_labels = [fmt_period(p) for p in periods]
        tx_label = st.selectbox("Pay period", options=tx_labels, index=0, key="fin_tx_period_dd")  # past period by default
        tx_start, tx_end = periods[tx_labels.index(tx_label)]
        search_q = st.text_input("Filter text (category/description contains)", key="fin_tx_search")

        # Filter to selected period + optional text search
        if transactions.empty:
            period_tx = pd.DataFrame(columns=TRANSACTION_COLS)
        else:
            tdf = transactions.copy()
            tdf["date"] = pd.to_datetime(tdf["date"], errors="coerce")
            mask = (tdf["date"].dt.date >= tx_start) & (tdf["date"].dt.date <= tx_end)
            period_tx = tdf[mask].copy()
            if search_q:
                q = search_q.strip().lower()
                period_tx = period_tx[
                    period_tx["category"].fillna("").str.lower().str.contains(q) |
                    period_tx["description"].fillna("").str.lower().str.contains(q) |
                    period_tx["account"].fillna("").str.lower().str.contains(q)
                ]
            period_tx = period_tx.sort_values("date", ascending=False)

        # Build display table (no IDs shown; nice date format; selectable delete)
        if not period_tx.empty:
            display = period_tx.copy()
            display["Date"] = display["date"].apply(fmt_date_long)
            display["select"] = False
            display["__rid"] = display["id"]  # keep hidden id for deletion mapping
            display = display[["select","Date","account","category","description","amount","status","notes","__rid"]]

            edited_tbl = st.data_editor(
                display,
                use_container_width=True, height=360, hide_index=True,
                column_config={
                    "select": st.column_config.CheckboxColumn("Select", help="Check to mark for deletion"),
                    "amount": st.column_config.NumberColumn("Amount ($)"),
                    "__rid": st.column_config.TextColumn("__rid", disabled=True),
                },
                column_order=["select","Date","account","category","description","amount","status","notes"],  # hides __rid
                key="fin_tx_editor_period",
            )

            # Delete selected
            to_delete_ids = edited_tbl.loc[edited_tbl["select"] == True, "__rid"].tolist()
            if st.button("Delete selected", key="fin_tx_delete_btn") and to_delete_ids:
                remaining = transactions[~transactions["id"].isin(to_delete_ids)].copy()
                save_csv(FILES["transactions"], remaining[TRANSACTION_COLS])
                st.success(f"Deleted {len(to_delete_ids)} transaction(s).")
                st.experimental_rerun()
        else:
            st.info("No transactions in this pay period.")

    # ========================= VISUALS (below the table) =========================

    # CASH FLOW (Selected Period) — spending by category
    st.markdown("#### Cash Flow — Spending by Category (Selected Period)")
    if not period_tx.empty:
        tx2 = period_tx.merge(accounts[["name","type"]], left_on="account", right_on="name", how="left") \
                       .rename(columns={"type":"acc_type"}).drop(columns=["name"])
        not_income   = ~tx2["category"].fillna("").str.strip().str.lower().str.startswith("income")
        not_transfer = tx2["category"].fillna("").str.strip().str.lower() != "transfer"
        spend = tx2[not_income & not_transfer].copy()
        if not spend.empty:
            spend["outflow"] = np.where(
                spend["acc_type"].fillna("bank") == "credit",
                spend["amount"].clip(lower=0),
                (-spend["amount"]).clip(lower=0),
            )
            by_cat = spend.groupby("category", dropna=False)["outflow"].sum().reset_index()
            by_cat = by_cat.sort_values("outflow", ascending=False)

            flow_bar = (
                alt.Chart(by_cat)
                .mark_bar()
                .encode(
                    x=alt.X("category:N", sort="-y", title="Category"),
                    y=alt.Y("outflow:Q", title="Spend ($)"),
                    tooltip=[alt.Tooltip("category:N", title="Category"),
                             alt.Tooltip("outflow:Q",  title="Spend ($)", format="$,.2f")],
                )
                .properties(height=240)
                .interactive()
            )
            st.altair_chart(flow_bar, use_container_width=True)
        else:
            st.info("No spending found in the selected period.")
    else:
        st.info("Add a few transactions in this period to see your cash flow.")

    # INCOME (This Month) — single growing bar for the current calendar month
    st.markdown("#### Income — This Month")
    if not transactions.empty:
        inc = transactions.copy()
        inc["date"] = pd.to_datetime(inc["date"], errors="coerce")
        mask_inc = inc["category"].fillna("").str.strip().str.lower().str.startswith("income")
        inc = inc[mask_inc]
        if not inc.empty:
            # current month total
            month_start = pd.Timestamp(today.replace(day=1))
            next_month = (month_start + pd.offsets.MonthBegin(1))
            this_month = inc[(inc["date"] >= month_start) & (inc["date"] < next_month)].copy()
            total = float(this_month["amount"].sum()) if not this_month.empty else 0.0
            month_label = today.strftime("%B %Y")  # e.g., "October 2025"
            month_df = pd.DataFrame({"Month": [month_label], "Income": [total]})

            st.metric(f"Total income in {month_label}", f"${total:,.2f}")

            income_month_bar = (
                alt.Chart(month_df)
                .mark_bar(size=60)
                .encode(
                    x=alt.X("Month:N", title=""),
                    y=alt.Y("Income:Q", title="Income ($)"),
                    tooltip=[alt.Tooltip("Income:Q", title="Income ($)", format="$,.2f")],
                )
                .properties(height=200)
                .interactive()
            )
            st.altair_chart(income_month_bar, use_container_width=True)
        else:
            st.info("No income logged yet.")
    else:
        st.info("Log income to see monthly totals grow.")

    # BUDGETING CIRCLE — planned allocation (uses saved percentages)
    st.markdown("#### Budgeting Circle (Planned Allocation)")

    # Ensure 3 rows exist; default to 20/50/30 (Savings/Expenses/Wants)
    def ensure_budget_rows(bdf: pd.DataFrame) -> pd.DataFrame:
        desired = [
            ("Savings/Investing", "Savings", 20, "Save & invest"),
            ("Expenses",          "Checking", 50, "Bills, groceries, etc."),
            ("Wants",             "Checking", 30, "Fun/Discretionary"),
        ]
        if bdf.empty:
            rows = []
            for cat, acct, pct, note in desired:
                rows.append({"id": str(uuid.uuid4()), "category": cat, "percent": pct,
                             "auto_account": acct if acct in acc_names else "Checking", "notes": note})
            return pd.DataFrame(rows, columns=BUDGET_COLS)
        # Backfill missing
        present = bdf["category"].str.lower().tolist()
        for cat, acct, pct, note in desired:
            if cat.lower() not in present:
                bdf = pd.concat([bdf, pd.DataFrame([{
                    "id": str(uuid.uuid4()), "category": cat, "percent": pct,
                    "auto_account": acct if acct in acc_names else "Checking", "notes": note
                }])], ignore_index=True)
        return bdf[BUDGET_COLS]

    budget = ensure_budget_rows(budget)
    save_csv(FILES["budget"], budget)  # persist defaults/backfill

    # Base the circle on income within the selected pay period (same filter as table)
    period_income = 0.0
    if not period_tx.empty:
        is_income = period_tx["category"].fillna("").str.strip().str.lower().str.startswith("income")
        period_income = float(period_tx.loc[is_income, "amount"].sum())

    if period_income <= 0:
        st.info("No income found in the selected pay period. Add an income transaction to populate the circle.")
    else:
        b = budget.copy()
        b = b[b["category"].isin(["Savings/Investing","Expenses","Wants"])].copy()
        b["amount"] = (b["percent"].fillna(0).astype(float) / 100.0) * period_income
        b = b[b["amount"] > 0]
        if b.empty:
            st.info("All budget percentages are 0%. Update your budget.csv if you want a split.")
        else:
            pie = (
                alt.Chart(b)
                .mark_arc()
                .encode(
                    theta=alt.Theta("amount:Q"),
                    color=alt.Color("category:N", legend=alt.Legend(title="Category")),
                    tooltip=[
                        "category:N",
                        alt.Tooltip("percent:Q", title="Percent", format=".0f"),
                        alt.Tooltip("amount:Q",  title="Amount",  format="$,.2f"),
                    ],
                )
                .properties(height=260)
                .interactive()
            )
            st.altair_chart(pie, use_container_width=True)




# ---- MISSION & GOALS (full-page editor) ----
with tab_mission_goals:
    st.subheader("Mission & Goals")

    # Mission editor
    st.markdown("#### Mission")
    mission_edit = st.text_area("Your mission statement", value=settings.get("mission", ""), height=180, key="mission_text_edit")
    if st.button("Save Mission", type="primary", key="save_mission_btn"):
        settings["mission"] = mission_edit
        save_settings(settings)
        st.session_state.settings = settings
        st.success("Mission saved. Refresh the page to see it update under the header.")

    # Goals management
    st.markdown("#### Goals")
    goals_df = load_csv(FILES["goals"], GOALS_COLS)
    with st.expander("➕ Add a goal"):
        g_title = st.text_input("Title", key="mg_goal_title")
        c1, c2, c3 = st.columns(3)
        with c1:
            g_cat = st.selectbox("Category", ["Health", "Career", "School", "Finance", "Personal", "Other"], key="mg_goal_cat")
        with c2:
            g_timeframe = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly", "Long-term"], key="mg_goal_timeframe")
        with c3:
            g_target = st.date_input("Target date", value=date.today(), key="mg_goal_target")
        g_notes = st.text_area("Notes", key="mg_goal_notes")

        if st.button("Add Goal", type="primary", key="mg_add_goal") and g_title:
            new = {
                "id": str(uuid.uuid4()),
                "title": g_title,
                "category": g_cat,
                "timeframe": g_timeframe,
                "target_date": g_target.isoformat(),
                "status": "Active",
                "progress": 0,
                "notes": g_notes,
            }
            goals_df = pd.concat([goals_df, pd.DataFrame([new])], ignore_index=True)
            save_csv(FILES["goals"], goals_df)
            st.success("Goal added.")

    st.markdown("### Your Goals")
    if goals_df.empty:
        st.info("No goals yet—add your first one above.")
    else:
        edited = st.data_editor(
            goals_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.TextColumn("id", disabled=True),
                "progress": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
                "status": st.column_config.SelectboxColumn(options=["Active","Paused","Completed","Dropped"]),
            },
            key="mg_goals_editor",
        )
        if st.button("Save Goals", key="mg_save_goals"):
            save_csv(FILES["goals"], edited[GOALS_COLS])
            st.success("Saved.")

# ---- FOOTER ----
st.caption("© be intentional with everything you do - life is beautiful.")
