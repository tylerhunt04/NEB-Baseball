

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
    import matplotlib.pyplot as plt

    # Ensure file paths & schemas exist (in case they weren't added yet)
    if "accounts" not in FILES:
        FILES.update({
            "accounts": os.path.join(DATA_DIR, "accounts.csv"),
            "transactions": os.path.join(DATA_DIR, "transactions.csv"),
            "paychecks": os.path.join(DATA_DIR, "paychecks.csv"),
            "budget": os.path.join(DATA_DIR, "budget.csv"),
        })
    try:
        ACCOUNT_COLS
    except NameError:
        ACCOUNT_COLS = ["id", "name", "type", "opening_balance", "notes"]
    try:
        TRANSACTION_COLS
    except NameError:
        TRANSACTION_COLS = ["id", "date", "account", "category", "description", "amount", "status", "notes"]
    try:
        PAYCHECK_COLS
    except NameError:
        PAYCHECK_COLS = ["id", "date", "net_amount", "hours", "rate", "account", "notes"]
    try:
        BUDGET_COLS
    except NameError:
        BUDGET_COLS = ["id", "category", "percent", "auto_account", "notes"]

    # Colors used in visuals (fallback palette for unknown categories)
    NAMED_COLORS = {
        "savings":   "#4CAF50",
        "investing": "#1E88E5",
        "expenses":  "#E53935",
        "wants":     "#FB8C00",
    }
    def color_for(cat: str, i: int):
        key = (cat or "").strip().lower()
        if key in NAMED_COLORS:
            return NAMED_COLORS[key]
        # fallback cycle
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        return default_cycle[i % max(1, len(default_cycle))] if default_cycle else "#9E9E9E"

    # ---------------- LOAD / SEED ----------------
    accounts     = load_csv(FILES["accounts"], ACCOUNT_COLS)
    transactions = load_csv(FILES["transactions"], TRANSACTION_COLS)
    paychecks    = load_csv(FILES["paychecks"], PAYCHECK_COLS)
    budget       = load_csv(FILES["budget"], BUDGET_COLS)
    sett         = st.session_state.settings

    # Seed accounts if empty (Checking/Savings as 'bank')
    if accounts.empty:
        accounts = pd.DataFrame([
            {"id": str(uuid.uuid4()), "name": "Checking", "type": "bank", "opening_balance": 0.0, "notes": ""},
            {"id": str(uuid.uuid4()), "name": "Savings",  "type": "bank", "opening_balance": 0.0, "notes": ""},
        ], columns=ACCOUNT_COLS)
        save_csv(FILES["accounts"], accounts)

    # Normalize any older 'cash' labels to 'bank'
    if not accounts.empty:
        accounts["type"] = accounts["type"].fillna("bank").replace({"cash": "bank"})
        save_csv(FILES["accounts"], accounts)

    # If budget empty, provide the four common categories as a starting point
    if budget.empty:
        budget = pd.DataFrame([
            {"id": str(uuid.uuid4()), "category": "Savings",   "percent": 20, "auto_account": "Savings",  "notes": "Emergency/long-term"},
            {"id": str(uuid.uuid4()), "category": "Investing", "percent": 10, "auto_account": "Savings",  "notes": "Brokerage/retirement"},
            {"id": str(uuid.uuid4()), "category": "Expenses",  "percent": 50, "auto_account": "Checking", "notes": "Bills, groceries, etc."},
            {"id": str(uuid.uuid4()), "category": "Wants",     "percent": 20, "auto_account": "Checking", "notes": "Fun, dining out"},
        ], columns=BUDGET_COLS)
        save_csv(FILES["budget"], budget)

    # ---------------- ACCOUNTS ----------------
    st.markdown("### Accounts")

    # Simple editor (no column picker, no delete UI, 'id' hidden)
    edited_accounts = st.data_editor(
        accounts[["name", "type", "opening_balance", "notes"]],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "type": st.column_config.SelectboxColumn(
                options=["bank", "cash", "credit", "loan", "investment"], default="bank"
            ),
            "opening_balance": st.column_config.NumberColumn(format="%.2f"),
        },
        key="fin_accounts_editor",
    )

    if st.button("Save Accounts", key="fin_save_accounts"):
        merged = accounts.copy()
        # Extend or overwrite by row order (simple approach)
        for i in range(len(edited_accounts)):
            if i < len(merged):
                for col in ["name", "type", "opening_balance", "notes"]:
                    merged.loc[merged.index[i], col] = edited_accounts.iloc[i][col]
            else:
                # new row added in editor
                new_row = {c: None for c in ACCOUNT_COLS}
                new_row.update({
                    "id": str(uuid.uuid4()),
                    "name": edited_accounts.iloc[i]["name"],
                    "type": edited_accounts.iloc[i]["type"],
                    "opening_balance": edited_accounts.iloc[i]["opening_balance"],
                    "notes": edited_accounts.iloc[i]["notes"],
                })
                merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)
        # drop trailing empties (no name)
        merged = merged.dropna(subset=["name"]).reset_index(drop=True)
        # Ensure IDs
        merged["id"] = merged.get("id", "").fillna("").apply(lambda x: x or str(uuid.uuid4()))
        save_csv(FILES["accounts"], merged[ACCOUNT_COLS])
        accounts = merged[ACCOUNT_COLS]
        st.success("Accounts saved.")

    # Balances from cleared transactions
    def compute_balances(acc_df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
        if acc_df.empty:
            return pd.DataFrame(columns=["account","type","opening_balance","current_balance"])
        tx = tx_df.copy()
        tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
        tx = tx[tx["status"].fillna("Cleared") == "Cleared"]
        rows = []
        for _, acc in acc_df.iterrows():
            opening = float(acc.get("opening_balance", 0) or 0)
            total_tx = float(tx[tx["account"] == acc.get("name", "")]["amount"].sum())
            rows.append({
                "account": acc.get("name", ""),
                "type": acc.get("type", ""),
                "opening_balance": opening,
                "current_balance": opening + total_tx,
            })
        return pd.DataFrame(rows)

    balances = compute_balances(accounts, transactions)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Checking", f"${balances.loc[balances['account']=='Checking','current_balance'].sum():,.2f}" if not balances.empty else "$0.00")
    with c2:
        st.metric("Savings", f"${balances.loc[balances['account']=='Savings','current_balance'].sum():,.2f}" if not balances.empty else "$0.00")
    with c3:
        st.metric("Total balance", f"${balances['current_balance'].sum():,.2f}" if not balances.empty else "$0.00")

    if not balances.empty:
        st.bar_chart(balances.set_index("account")["current_balance"])

    st.divider()

    # ---------------- TRANSACTIONS ----------------
    st.markdown("### Transactions")

    acc_names = accounts["name"].dropna().tolist()
    common_cats = [
        "Rent", "Groceries", "Gas", "Dining", "Utilities", "Phone", "Insurance",
        "Entertainment", "Gym", "Medical", "Travel", "Gifts",
        "Income: Paycheck", "Transfer", "Other"
    ]

    # Quick add
    tx_cols = st.columns(5)
    with tx_cols[0]:
        t_date = st.date_input("Date", value=date.today(), key="fin_tx_date")
    with tx_cols[1]:
        t_account = st.selectbox("Account", acc_names, index=0 if acc_names else None, key="fin_tx_account")
    with tx_cols[2]:
        t_category = st.selectbox("Category", common_cats, index=1, key="fin_tx_cat")
    with tx_cols[3]:
        t_amount = st.number_input("Amount", value=0.0, step=1.0, key="fin_tx_amt")
    with tx_cols[4]:
        t_status = st.selectbox("Status", ["Cleared", "Planned"], index=0, key="fin_tx_status")
    t_desc = st.text_input("Description", placeholder="e.g., Costco", key="fin_tx_desc")
    t_notes = st.text_input("Notes", key="fin_tx_notes")

    if st.button("Add Transaction", type="primary", key="fin_add_tx"):
        new = {
            "id": str(uuid.uuid4()),
            "date": t_date.isoformat(),
            "account": t_account or "",
            "category": t_category,
            "description": t_desc,
            "amount": float(t_amount),
            "status": t_status,
            "notes": t_notes,
        }
        transactions = pd.concat([transactions, pd.DataFrame([new])], ignore_index=True)
        save_csv(FILES["transactions"], transactions[TRANSACTION_COLS])
        st.success("Transaction added.")

    # Editable table (no delete UI; Streamlit allows adding rows)
    edited_tx = st.data_editor(
        transactions[["date","account","category","description","amount","status","notes"]],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="fin_tx_editor",
    )
    if st.button("Save Transaction Changes", key="fin_save_tx_changes"):
        merged = transactions.copy()
        # Overwrite/extend by row order
        for i in range(len(edited_tx)):
            if i < len(merged):
                for col in ["date","account","category","description","amount","status","notes"]:
                    merged.loc[merged.index[i], col] = edited_tx.iloc[i][col]
            else:
                new_row = {c: None for c in TRANSACTION_COLS}
                new_row.update({
                    "id": str(uuid.uuid4()),
                    "date": edited_tx.iloc[i]["date"],
                    "account": edited_tx.iloc[i]["account"],
                    "category": edited_tx.iloc[i]["category"],
                    "description": edited_tx.iloc[i]["description"],
                    "amount": edited_tx.iloc[i]["amount"],
                    "status": edited_tx.iloc[i]["status"],
                    "notes": edited_tx.iloc[i]["notes"],
                })
                merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)
        merged = merged.dropna(subset=["date"]).reset_index(drop=True)
        merged["id"] = merged.get("id", "").fillna("").apply(lambda x: x or str(uuid.uuid4()))
        save_csv(FILES["transactions"], merged[TRANSACTION_COLS])
        transactions = merged[TRANSACTION_COLS]
        st.success("Transactions saved.")

    st.markdown("#### Recent")
    if transactions.empty:
        st.info("No transactions yet.")
    else:
        st.dataframe(transactions.sort_values("date", ascending=False).head(50), use_container_width=True)

    st.divider()

    # ---------------- PAYCHECK PLANNER ----------------
    st.markdown("### Paycheck Planner (bi-weekly)")

    # Settings row
    acc_list = accounts["name"].dropna().tolist()
    default_deposit_idx = (
        acc_list.index(sett.get("default_deposit_account", "Checking"))
        if sett.get("default_deposit_account", "Checking") in acc_list else (0 if acc_list else None)
    )
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        last_pay_val = pd.to_datetime(sett.get("last_pay_date")).date() if sett.get("last_pay_date") else date.today()
        last_pay = st.date_input("Last payday", value=last_pay_val, key="fin_last_pay")
    with colp2:
        freq_days = st.number_input("Pay frequency (days)", min_value=7, max_value=31, value=int(sett.get("pay_frequency_days", 14)), step=1, key="fin_pay_freq")
    with colp3:
        deposit_account = st.selectbox("Deposit to account", acc_list, index=default_deposit_idx, key="fin_deposit_acct")

    if st.button("Save Pay Settings", key="fin_save_pay_settings"):
        sett["last_pay_date"] = last_pay.isoformat()
        sett["pay_frequency_days"] = int(freq_days)
        sett["default_deposit_account"] = deposit_account or "Checking"
        save_settings(sett)
        st.success("Pay settings saved.")

    next1 = last_pay + timedelta(days=int(freq_days))
    next2 = next1 + timedelta(days=int(freq_days))
    st.caption(f"Next paydays: {next1.isoformat()} and {next2.isoformat()}")

    # Minimal planner: date + net only
    colpp1, colpp2 = st.columns(2)
    with colpp1:
        p_date = st.date_input("Paycheck date", value=next1, key="fin_plan_pay_date")
    with colpp2:
        net_amount = st.number_input("Net amount ($)", min_value=0.0, value=0.0, step=10.0, key="fin_plan_net")

    # Allocation rules editor (no delete UI; 'id' hidden). Add any rows you like (title + percent).
    st.markdown("#### Allocation rules (percent of net)")
    edited_budget = st.data_editor(
        budget[["category","percent","auto_account","notes"]],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "percent": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
            "auto_account": st.column_config.SelectboxColumn(options=acc_list),
        },
        key="fin_budget_editor",
    )

    total_pct = float(edited_budget["percent"].fillna(0).sum()) if not edited_budget.empty else 0.0
    st.caption(f"Total: {total_pct:.0f}% (aim for 100%)")

    if st.button("Save Allocation Rules", key="fin_save_budget"):
        merged = edited_budget.copy()

        # Preserve IDs by category name where possible
        old_ids = {str(r["category"]).strip().lower(): r["id"] for _, r in budget.iterrows() if pd.notna(r.get("id", None))}
        if "id" not in merged.columns:
            merged["id"] = ""
        merged["id"] = merged.apply(
            lambda r: old_ids.get(str(r["category"]).strip().lower(), str(uuid.uuid4())),
            axis=1
        )
        # Clean empties
        merged = merged.dropna(subset=["category"]).reset_index(drop=True)

        save_csv(FILES["budget"], merged[BUDGET_COLS])
        budget = merged[BUDGET_COLS]
        st.success("Allocation rules saved.")

    # ---- PIE: planned split based on ALL rows you added in Allocation Rules ----
    # The pie uses your current editor rows: every row (title + percent) appears in the pie.
    st.markdown("#### Paycheck Split (Planned)")
    if net_amount > 0 and not edited_budget.empty:
        # Use percentages directly for the pie slices; label shows name + $ amount
        cats = [str(c) for c in edited_budget["category"].fillna("").tolist()]
        pcts = [float(x or 0) for x in edited_budget["percent"].fillna(0).tolist()]
        amts = [net_amount * p / 100.0 for p in pcts]

        # Filter out zero-percent categories
        nz = [(c, p, a) for c, p, a in zip(cats, pcts, amts) if p > 0]
        if nz:
            labels = [f"{c} (${a:,.0f})" for c, _, a in nz]
            values = [p for _, p, _ in nz]  # pie by percent keeps proportions clear
            colors = [color_for(c, i) for i, (c, _, _) in enumerate(nz)]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(values, labels=labels, colors=colors, startangle=90, autopct=None)  # FULL PIE (not donut)
            ax.set_title("Where this paycheck is planned to go")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Set some non-zero percentages above to see the pie chart.")
    elif net_amount <= 0:
        st.info("Enter your paycheck **Net amount ($)** to show the planned split pie.")
    else:
        st.info("Add at least one allocation row (category + percent) to show the pie chart.")

    # ---- Cash Flow (weekly bars) during this pay period, cleared only ----
    st.markdown("#### Cash Flow (Weekly, this pay period)")
    # Pay period window: from (p_date - freq_days + 1) to p_date (inclusive)
    period_start = pd.to_datetime(p_date) - pd.Timedelta(days=int(freq_days) - 1)
    period_end   = pd.to_datetime(p_date)

    txp = transactions.copy()
    if not txp.empty:
        txp["date"] = pd.to_datetime(txp["date"], errors="coerce")
        txp = txp[(txp["date"] >= period_start) & (txp["date"] <= period_end)]
        cleared_period = txp[txp["status"].fillna("Cleared") == "Cleared"]
    else:
        cleared_period = pd.DataFrame(columns=TRANSACTION_COLS)

    if not cleared_period.empty:
        weekly = cleared_period.resample("W-MON", on="date")["amount"].sum().rename("Net").to_frame()
        st.bar_chart(weekly)
    else:
        st.info("No cleared transactions in this pay period yet for the cash flow chart.")

    # ---- Log paycheck + planned allocations (create planned rows for ALL budget categories) ----
    if net_amount > 0 and st.button("Log paycheck + planned allocations", type="primary", key="fin_post_paycheck"):
        # 1) record paycheck
        p_record = {
            "id": str(uuid.uuid4()),
            "date": p_date.isoformat(),
            "net_amount": float(net_amount),
            "hours": np.nan,  # schema compatibility
            "rate": np.nan,   # schema compatibility
            "account": deposit_account or "Checking",
            "notes": "",
        }
        paychecks = pd.concat([paychecks, pd.DataFrame([p_record])], ignore_index=True)
        save_csv(FILES["paychecks"], paychecks[PAYCHECK_COLS])

        # 2) deposit transaction (cleared)
        tx_deposit = {
            "id": str(uuid.uuid4()),
            "date": p_date.isoformat(),
            "account": deposit_account or "Checking",
            "category": "Income: Paycheck",
            "description": "Bi-weekly paycheck",
            "amount": float(net_amount),
            "status": "Cleared",
            "notes": "",
        }
        transactions = pd.concat([transactions, pd.DataFrame([tx_deposit])], ignore_index=True)

        # 3) planned negative transactions for *every* allocation row
        plan_rows = []
        for _, r in edited_budget.iterrows():
            cat = str(r.get("category", "") or "").strip()
            pct = float(r.get("percent", 0) or 0)
            if pct <= 0 or not cat:
                continue
            amt = (pct / 100.0) * float(net_amount)
            target_acct = r.get("auto_account") or (deposit_account or "Checking")
            plan_rows.append({
                "id": str(uuid.uuid4()),
                "date": p_date.isoformat(),
                "account": deposit_account or "Checking",
                "category": f"Planned: {cat}",
                "description": f"Allocation plan → {target_acct}",
                "amount": -amt,
                "status": "Planned",
                "notes": r.get("notes", ""),
            })

        if plan_rows:
            transactions = pd.concat([transactions, pd.DataFrame(plan_rows)], ignore_index=True)

        save_csv(FILES["transactions"], transactions[TRANSACTION_COLS])
        st.success("Paycheck logged and allocations planned. Use Transfers and mark items **Cleared** as you move money.")





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
