
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
    "transactions": os.path.join(DATA_DIR, "transactions.csv"),
    "accounts": os.path.join(DATA_DIR, "accounts.csv"),
    "paychecks": os.path.join(DATA_DIR, "paychecks.csv"),
    "budget": os.path.join(DATA_DIR, "budget.csv"),
}

DEFAULT_SETTINGS = {
    "units": "imperial",  # or "metric"
    "default_week_start": "Monday",
    "goal_weight": None,  # number in selected units
    "pay_frequency_days": 14,  # bi-weekly
    "last_pay_date": "",      # ISO date of last paycheck
    "default_deposit_account": "",
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
TRANSACTION_COLS = [
    "id", "date", "account", "category", "description", "amount", "status", "notes"
]
ACCOUNT_COLS = [
    "id", "name", "type", "opening_balance", "notes"
]
PAYCHECK_COLS = [
    "id", "date", "net_amount", "hours", "rate", "account", "notes"
]
BUDGET_COLS = [
    "id", "category", "percent", "auto_account", "notes"
]

# =============== SESSION STATE ==================================================
if "settings" not in st.session_state:
    st.session_state.settings = load_settings()


# =============== SIDEBAR NAV ====================================================
st.set_page_config(page_title=f"{APP_NAME} — Self Management", layout="wide", initial_sidebar_state="collapsed")
# Force light-like appearance (works even if user theme is dark)
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
    [data-testid="stSidebar"] {display: none !important;} /* hide sidebar entirely */
    .stMarkdown, .stText, .stDataFrame, .stMetric { color: var(--text-color) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Tyler Hunt")

# ---- TOP NAV TABS ----
st.title("Be Intentional")
(
    tab_dashboard,
    tab_schedule,
    tab_workouts,
    tab_weighins,
    tab_journal,
    tab_habits,
    tab_finance,
    tab_data,
) = st.tabs([
    "Dashboard",
    "Schedule",
    "Workouts",
    "Weigh‑ins",
    "Journal",
    "Habits",
    "Finance",
    "Data & Settings",
])

# Quick Add lives on the Dashboard now
with tab_dashboard:
    with st.expander("➕ Quick Add", expanded=False):
        qa_choice = st.selectbox("Type", ["Task", "Weigh‑in", "Journal"], key="qa_type_top")
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
        elif qa_choice == "Weigh‑in":
            wi_date = st.date_input("Date", value=date.today(), key="qa_wi_date_top")
            wi_weight = st.number_input("Weight", min_value=0.0, step=0.1, key="qa_wi_weight_top")
            wi_bf = st.number_input("Body Fat %", min_value=0.0, max_value=100.0, step=0.1, key="qa_wi_bf_top")
            if st.button("Add Weigh‑in", use_container_width=True, key="qa_add_wi_top"):
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
                st.success("Weigh‑in logged.")
        else:
            j_date = st.date_input("Date", value=date.today(), key="qa_j_date_top")
            j_title = st.text_input("Title", key="qa_j_title_top")
            j_mood = st.slider("Mood", 1, 10, 6, key="qa_j_mood_top")
            j_tags = st.text_input("Tags (comma‑sep)", key="qa_j_tags_top")
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
    # shift so that chosen week_start is 0
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

    # Weigh‑ins stats
    wi = load_csv(FILES["weighins"], WEIGHIN_COLS)
    wi_sorted = wi.dropna(subset=["weight"]).sort_values("date")
    latest_weight = wi_sorted["weight"].iloc[-1] if len(wi_sorted) else None
    last_7 = wi_sorted.tail(7)["weight"].mean() if len(wi_sorted) >= 1 else None

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Tasks today", int(today_tasks.shape[0]))
    with c2:
        st.metric("Planned workout blocks", int(today_plan.shape[0]))
    with c3:
        st.metric("Latest weight", f"{latest_weight:.1f}" if latest_weight else "—")

    st.markdown("### Today's Schedule")
    if today_tasks.empty:
        st.info("No tasks for today yet. Use sidebar quick add or Schedule page.")
    else:
        st.dataframe(
            today_tasks[["start_time", "end_time", "title", "category", "priority", "status", "notes"]]
            .sort_values("start_time"),
            use_container_width=True,
        )

    st.markdown("### Today's Workout Plan")
    if today_plan.empty:
        st.info("No workout blocks planned for today. Add some on the Workouts page.")
    else:
        st.dataframe(today_plan[["block", "exercise", "target_sets", "target_reps", "target_load", "notes"]], use_container_width=True)

    st.markdown("### Recent Weigh‑ins")
    if wi_sorted.empty:
        st.info("Log your first weigh‑in on the Weigh‑ins page or via sidebar.")
    else:
        st.line_chart(wi_sorted.set_index("date")["weight"], height=220)

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

# ---- WEIGH‑INS ----

with tab_weighins:
    st.subheader("Weekly Weigh‑ins")

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

    if st.button("Add Weigh‑in", type="primary"):
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
        st.success("Saved weigh‑in.")

    st.markdown("### Trend")
    if wi.empty:
        st.info("Add weigh‑ins to see your trend.")
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
            j_tags = st.text_input("Tags (comma‑sep)", key="journal_tags_form")
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
        # filters
        colf1, colf2 = st.columns(2)
        with colf1:
            tag_filter = st.text_input("Filter by tag contains")
        with colf2:
            mood_min, mood_max = st.slider("Mood range", 1, 10, (1, 10))
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
        h_name = st.text_input("Habit name", placeholder="e.g., Read 20 min")
        h_weekly = st.number_input("Weekly target (times)", min_value=1, max_value=21, value=5)
        if st.button("Add Habit") and h_name:
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
        )
        if st.button("Save Habits"):
            save_csv(FILES["habits"], edited[HABIT_COLS])
            st.success("Saved.")

        st.markdown("### Weekly Review")
        view = edited.copy() if isinstance(edited, pd.DataFrame) else hb.copy()
        days = ["sun","mon","tue","wed","thu","fri","sat"]
        view["done"] = view[days].sum(axis=1)
        view["progress"] = (view["done"] / view["weekly_target"]).clip(upper=1.0)
        st.dataframe(view[["name","weekly_target","done","progress","notes"]])

# ---- FINANCE ----
with tab_finance:
    st.subheader("Finance")

    # Load datasets
    accounts = load_csv(FILES["accounts"], ACCOUNT_COLS)
    transactions = load_csv(FILES["transactions"], TRANSACTION_COLS)
    paychecks = load_csv(FILES["paychecks"], PAYCHECK_COLS)
    budget = load_csv(FILES["budget"], BUDGET_COLS)

    # Provide sensible default budget if empty
    if budget.empty:
        budget = pd.DataFrame([
            {"id": str(uuid.uuid4()), "category": "Savings", "percent": 20, "auto_account": "", "notes": "Emergency/investing"},
            {"id": str(uuid.uuid4()), "category": "Essentials", "percent": 50, "auto_account": "", "notes": "Rent, utilities, groceries"},
            {"id": str(uuid.uuid4()), "category": "Wants", "percent": 30, "auto_account": "", "notes": "Fun, eating out"},
        ], columns=BUDGET_COLS)
        save_csv(FILES["budget"], budget)

    # --- Accounts section ---
    st.markdown("### Accounts")
    st.caption("Add your accounts with opening balances. Current balance is opening balance plus cleared transactions.")
    edited_accounts = st.data_editor(
        accounts if not accounts.empty else pd.DataFrame(columns=ACCOUNT_COLS),
        num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={"id": st.column_config.TextColumn("id", disabled=True)},
        key="fin_accounts_editor",
    )
    if st.button("Save Accounts", key="fin_save_accounts"):
        if "id" in edited_accounts.columns:
            edited_accounts["id"] = edited_accounts["id"].fillna("").apply(lambda x: x if x else str(uuid.uuid4()))
        else:
            edited_accounts["id"] = [str(uuid.uuid4()) for _ in range(len(edited_accounts))]
        save_csv(FILES["accounts"], edited_accounts[ACCOUNT_COLS])
        st.success("Accounts saved.")
        accounts = edited_accounts[ACCOUNT_COLS]

    # Compute balances from cleared transactions
    def compute_balances(acc_df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
        if acc_df.empty:
            return pd.DataFrame(columns=["account","type","opening_balance","current_balance"])
        tx = tx_df.copy()
        tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
        mask = tx["status"].fillna("Cleared") != "Planned"
        tx = tx[mask]
        rows = []
        for _, acc in acc_df.iterrows():
            opening = float(acc.get("opening_balance", 0) or 0)
            total_tx = float(tx[tx["account"] == acc.get("name", "")]["amount"].sum())
            current = opening + total_tx
            rows.append({
                "account": acc.get("name", ""),
                "type": acc.get("type", ""),
                "opening_balance": opening,
                "current_balance": current,
            })
        return pd.DataFrame(rows)

    balances = compute_balances(accounts, transactions)
    if not balances.empty:
        total_cash = balances["current_balance"].sum()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total cash (cleared)", f"${total_cash:,.2f}")
        with c2:
            st.dataframe(balances, use_container_width=True)
    else:
        st.info("Add at least one account to see balances.")

    st.divider()

    # --- Transactions section ---
    st.markdown("### Transactions")
    acc_names = accounts["name"].dropna().tolist() if not accounts.empty else []
    colt1, colt2, colt3 = st.columns(3)
    with colt1:
        t_date = st.date_input("Date", value=date.today(), key="fin_tx_date")
    with colt2:
        t_account = st.selectbox("Account", acc_names if acc_names else ["(no accounts)"] , key="fin_tx_account")
    with colt3:
        t_amount = st.number_input("Amount (\u2212 expense, + income)", value=0.0, step=1.0, key="fin_tx_amount")
    t_desc = st.text_input("Description", key="fin_tx_desc")
    t_cat = st.text_input("Category", value="", key="fin_tx_cat")
    t_status = st.selectbox("Status", ["Cleared", "Planned"], index=0, key="fin_tx_status")
    t_notes = st.text_input("Notes", key="fin_tx_notes")
    if st.button("Add Transaction", type="primary", key="fin_add_tx"):
        if not acc_names:
            st.error("Please add an account first in Accounts above.")
        else:
            new = {
                "id": str(uuid.uuid4()),
                "date": t_date.isoformat(),
                "account": t_account if t_account != "(no accounts)" else "",
                "category": t_cat,
                "description": t_desc,
                "amount": float(t_amount),
                "status": t_status,
                "notes": t_notes,
            }
            transactions = pd.concat([transactions, pd.DataFrame([new])], ignore_index=True)
            save_csv(FILES["transactions"], transactions[TRANSACTION_COLS])
            st.success("Transaction added.")

    st.markdown("#### Recent")
    if transactions.empty:
        st.info("No transactions yet.")
    else:
        st.dataframe(
            transactions.sort_values("date", ascending=False).head(50),
            use_container_width=True,
        )
        # Spending this month by category (expenses only)
        txm = transactions.copy()
        txm["date"] = pd.to_datetime(txm["date"], errors="coerce")
        month_mask = txm["date"].dt.to_period("M") == pd.Timestamp(date.today()).to_period("M")
        txm = txm[month_mask]
        txm_exp = txm[txm["amount"] < 0]
        if not txm_exp.empty:
            by_cat = txm_exp.groupby("category")["amount"].sum().sort_values()
            st.bar_chart(by_cat.abs())

    st.divider()

    # --- Paycheck Planner ---
    st.markdown("### Paycheck Planner (bi‑weekly)")
    sett = st.session_state.settings

    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        last_pay = st.date_input("Last payday", value=(date.today() if not sett.get("last_pay_date") else pd.to_datetime(sett.get("last_pay_date")).date()), key="fin_last_pay")
    with colp2:
        freq_days = st.number_input("Pay frequency (days)", min_value=7, max_value=31, value=int(sett.get("pay_frequency_days", 14)), step=1, key="fin_pay_freq")
    with colp3:
        deposit_account = st.selectbox("Deposit to account", acc_names if acc_names else ["(no accounts)"], key="fin_deposit_acct")

    if st.button("Save Pay Settings", key="fin_save_pay_settings"):
        sett["last_pay_date"] = last_pay.isoformat()
        sett["pay_frequency_days"] = int(freq_days)
        sett["default_deposit_account"] = deposit_account if deposit_account != "(no accounts)" else ""
        save_settings(sett)
        st.success("Pay settings saved.")

    # Next paydays preview
    next1 = last_pay + timedelta(days=int(freq_days))
    next2 = next1 + timedelta(days=int(freq_days))
    st.caption(f"Next paydays: {next1.isoformat()} and {next2.isoformat()}")

    st.markdown("#### Plan this paycheck")
    colpp1, colpp2, colpp3 = st.columns(3)
    with colpp1:
        p_date = st.date_input("Paycheck date", value=next1, key="fin_plan_pay_date")
    with colpp2:
        hours = st.number_input("Hours", min_value=0.0, value=0.0, step=0.25, key="fin_plan_hours")
    with colpp3:
        rate = st.number_input("Rate ($/hr)", min_value=0.0, value=0.0, step=0.25, key="fin_plan_rate")
    net_amount = st.number_input("Net amount ($)", min_value=0.0, value=0.0 if (hours==0 or rate==0) else round(hours*rate*0.8,2), step=10.0, key="fin_plan_net")
    st.caption("Enter net (after taxes/withholding). If you fill Hours & Rate, we'll prefill net ~80% for convenience—adjust as needed.")

    st.markdown("#### Allocation rules (percent of net)")
    edited_budget = st.data_editor(
        budget,
        num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("id", disabled=True),
            "percent": st.column_config.NumberColumn(min_value=0, max_value=100, step=1),
        },
        key="fin_budget_editor",
    )
    total_pct = float(edited_budget["percent"].sum()) if not edited_budget.empty else 0.0
    st.caption(f"Total: {total_pct:.0f}% (should be 100%)")
    if st.button("Save Allocation Rules", key="fin_save_budget"):
        if "id" in edited_budget.columns:
            edited_budget["id"] = edited_budget["id"].fillna("").apply(lambda x: x if x else str(uuid.uuid4()))
        else:
            edited_budget["id"] = [str(uuid.uuid4()) for _ in range(len(edited_budget))]
        save_csv(FILES["budget"], edited_budget[BUDGET_COLS])
        st.success("Allocation rules saved.")
        budget = edited_budget[BUDGET_COLS]

    if net_amount > 0 and not edited_budget.empty:
        alloc_df = edited_budget.copy()
        alloc_df["amount"] = (alloc_df["percent"].astype(float) / 100.0) * float(net_amount)
        st.dataframe(alloc_df[["category","percent","amount","auto_account","notes"]], use_container_width=True)
        if abs(alloc_df["percent"].sum() - 100) > 0.01:
            st.warning("Your allocation percentages do not sum to 100%. Adjust above to reach 100%.")
        if st.button("Log paycheck + planned allocations", type="primary", key="fin_post_paycheck"):
            if not acc_names:
                st.error("Add an account in Accounts first.")
            else:
                # 1) record paycheck
                p_record = {
                    "id": str(uuid.uuid4()),
                    "date": p_date.isoformat(),
                    "net_amount": float(net_amount),
                    "hours": float(hours),
                    "rate": float(rate),
                    "account": deposit_account if deposit_account != "(no accounts)" else "",
                    "notes": "",
                }
                paychecks = pd.concat([paychecks, pd.DataFrame([p_record])], ignore_index=True)
                save_csv(FILES["paychecks"], paychecks[PAYCHECK_COLS])

                # 2) deposit transaction (cleared)
                tx_deposit = {
                    "id": str(uuid.uuid4()),
                    "date": p_date.isoformat(),
                    "account": deposit_account if deposit_account != "(no accounts)" else "",
                    "category": "Income: Paycheck",
                    "description": "Bi-weekly paycheck",
                    "amount": float(net_amount),
                    "status": "Cleared",
                    "notes": "",
                }
                transactions = pd.concat([transactions, pd.DataFrame([tx_deposit])], ignore_index=True)

                # 3) planned allocation transactions (negative amounts)
                planned_rows = []
                for _, r in alloc_df.iterrows():
                    planned_rows.append({
                        "id": str(uuid.uuid4()),
                        "date": p_date.isoformat(),
                        "account": deposit_account if deposit_account != "(no accounts)" else "",
                        "category": f"Planned: {r['category']}",
                        "description": "Allocation plan",
                        "amount": -float(r["amount"]),
                        "status": "Planned",
                        "notes": r.get("notes", ""),
                    })
                if planned_rows:
                    transactions = pd.concat([transactions, pd.DataFrame(planned_rows)], ignore_index=True)
                save_csv(FILES["transactions"], transactions[TRANSACTION_COLS])
                st.success("Paycheck logged and allocations planned. Mark items as Cleared when money actually moves.")

st.divider()

# ---- DATA & SETTINGS ----

with tab_data:
    st.subheader("Settings")
    settings = st.session_state.settings

    c1, c2, c3 = st.columns(3)
    with c1:
        units = st.selectbox("Units", ["imperial", "metric"], index=0 if settings.get("units")=="imperial" else 1)
    with c2:
        week_start = st.selectbox("Week starts", ["Monday", "Sunday"], index=0 if settings.get("default_week_start")=="Monday" else 1, key="settings_week_start")
    with c3:
        goal_weight = st.number_input("Goal weight", min_value=0.0, value=float(settings.get("goal_weight") or 0.0))
    if st.button("Save Settings", type="primary"):
        settings.update({"units": units, "default_week_start": week_start, "goal_weight": goal_weight if goal_weight>0 else None})
        save_settings(settings)
        st.session_state.settings = settings
        st.success("Settings saved.")

    st.divider()
    st.subheader("Data Export / Import")

    # Export
    if st.button("Download all data as ZIP"):
        import zipfile
        from io import BytesIO

        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # ensure files exist before zipping
            load_csv(FILES["schedule"], SCHEDULE_COLS).to_csv(FILES["schedule"], index=False)
            load_csv(FILES["workout_plan"], WORKOUT_PLAN_COLS).to_csv(FILES["workout_plan"], index=False)
            load_csv(FILES["workout_log"], WORKOUT_LOG_COLS).to_csv(FILES["workout_log"], index=False)
            load_csv(FILES["weighins"], WEIGHIN_COLS).to_csv(FILES["weighins"], index=False)
            load_csv(FILES["journal"], JOURNAL_COLS).to_csv(FILES["journal"], index=False)
            load_csv(FILES["habits"], HABIT_COLS).to_csv(FILES["habits"], index=False)
            load_csv(FILES["transactions"], TRANSACTION_COLS).to_csv(FILES["transactions"], index=False)
            load_csv(FILES["accounts"], ACCOUNT_COLS).to_csv(FILES["accounts"], index=False)
            load_csv(FILES["paychecks"], PAYCHECK_COLS).to_csv(FILES["paychecks"], index=False)
            load_csv(FILES["budget"], BUDGET_COLS).to_csv(FILES["budget"], index=False)
            for name, path in FILES.items():
                if os.path.exists(path):
                    zf.write(path, arcname=os.path.basename(path))
        st.download_button(
            label="Download lifehub_data.zip",
            data=memory_file.getvalue(),
            file_name="lifehub_data.zip",
            mime="application/zip",
        )

    # Import (replace)
    up = st.file_uploader("Upload CSV to replace a dataset (careful!)", type=["csv"], accept_multiple_files=False)
    target = st.selectbox("Target dataset", ["schedule","workout_plan","workout_log","weighins","journal","habits","transactions","accounts","paychecks","budget"], index=0)
    if st.button("Replace with uploaded CSV") and up is not None:
        try:
            df = pd.read_csv(up)
            expected = {
                "schedule": SCHEDULE_COLS,
                "workout_plan": WORKOUT_PLAN_COLS,
                "workout_log": WORKOUT_LOG_COLS,
                "weighins": WEIGHIN_COLS,
                "journal": JOURNAL_COLS,
                "habits": HABIT_COLS,
                "transactions": TRANSACTION_COLS,
                "accounts": ACCOUNT_COLS,
                "paychecks": PAYCHECK_COLS,
                "budget": BUDGET_COLS,
            }[target]
            # attempt to align columns
            for c in expected:
                if c not in df.columns:
                    df[c] = np.nan
            save_csv(FILES[target], df[expected])
            st.success(f"Replaced '{target}' dataset.")
        except Exception as e:
            st.error(f"Import failed: {e}")

st.caption("© be intentional in everything you do - life is beautiful.")
