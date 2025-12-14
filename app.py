import streamlit as st
import pandas as pd
from pathlib import Path

from solver import (
    solve_week_plan,
    GOALS_ALLOWED,
    STRENGTH_TEMPLATES,
    NO_EQUIP_OPTION
)

CSV_PATH = str(Path(__file__).resolve().parent / "megaGymDataset.csv")

st.set_page_config(page_title="Workout Optimizer", layout="wide")

st.title("Workout Optimizer")
st.caption("Made by An Pham, Andrew Nguyen, Victor Le")


@st.cache_data
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[~df["BodyPart"].isna()].copy()
    df["Rating"] = df["Rating"].fillna(df["Rating"].median())
    return df


def allowed_levels_for_ui(user_level: str) -> set:
    if user_level == "Beginner":
        return {"Beginner"}
    if user_level == "Intermediate":
        return {"Beginner", "Intermediate"}
    return {"Beginner", "Intermediate", "Expert"}


def effective_equipment_filter(user_equipment_selection: list[str] | None):
    """
    Returns:
      - None if no equipment restriction should be applied
      - list with NO_EQUIP_OPTION if body-only-only
      - list of chosen equipment (solver will add Body Only automatically)
    """
    if not user_equipment_selection:
        return None
    if NO_EQUIP_OPTION in user_equipment_selection:
        return [NO_EQUIP_OPTION]
    return user_equipment_selection


df_all = load_dataset(CSV_PATH)

with st.sidebar:
    st.header("User Inputs")

    user_level = st.selectbox("Fitness level", ["Beginner", "Intermediate", "Expert"], index=0)
    goal = st.selectbox("End goal", GOALS_ALLOWED, index=0)

    workout_days = st.slider("Workout days per week", 0, 7, 5)
    minutes = st.slider("Minutes per session", 10, 120, 60, step=5)

    allowed_lvls = allowed_levels_for_ui(user_level)

    df_eq = df_all[
        (df_all["Type"] == goal) &
        (df_all["Level"].isin(allowed_lvls)) &
        (~df_all["Equipment"].isna())
    ].copy()

    equipment_options = sorted(df_eq["Equipment"].unique().tolist())
    equipment_options = [e for e in equipment_options if e != "Body Only"]
    equipment_options = [NO_EQUIP_OPTION] + equipment_options

    equipment_ui = st.multiselect(
        "Equipment available",
        options=equipment_options,
        default=[]
    )

    if NO_EQUIP_OPTION in equipment_ui and len(equipment_ui) > 1:
        equipment_ui = [NO_EQUIP_OPTION]

    allowed_equipment = effective_equipment_filter(equipment_ui)

    st.divider()
    st.subheader("Constraints / Toggles")

    avoid_consecutive = st.toggle("Avoid same body part on consecutive days", value=True)
    allow_repeat_exercises = st.toggle("Allow same exercise on multiple days", value=False)

    st.divider()

    strength_template = None
    custom_muscles_by_day = None

    if goal == "Strength":
        strength_template = st.selectbox(
            "Strength template (required)",
            STRENGTH_TEMPLATES,
            index=STRENGTH_TEMPLATES.index("Push/Pull/Leg")
        )

        df_strength = df_all[
            (df_all["Type"] == "Strength") &
            (df_all["Level"].isin(allowed_lvls)) &
            (~df_all["BodyPart"].isna())
        ].copy()

        if allowed_equipment is not None:
            if NO_EQUIP_OPTION in allowed_equipment:
                df_strength = df_strength[df_strength["Equipment"] == "Body Only"].copy()
            else:
                equip_set = set(allowed_equipment)
                equip_set.add("Body Only")
                df_strength = df_strength[df_strength["Equipment"].isin(equip_set)].copy()

        available_muscles = sorted(df_strength["BodyPart"].unique().tolist())

        if strength_template == "Custom":
            custom_muscles_by_day = []
            for i in range(workout_days):
                picked = st.multiselect(
                    f"Workout day {i + 1} muscles",
                    options=available_muscles,
                    default=available_muscles[:1] if available_muscles else [],
                    key=f"custom_muscles_{i}"
                )
                custom_muscles_by_day.append(picked)

    generate = st.button("Generate plan", use_container_width=True)

if generate:
    if goal == "Strength" and strength_template == "Custom" and workout_days > 0:
        if not custom_muscles_by_day or any(len(x) == 0 for x in custom_muscles_by_day):
            st.error("Could not find optimized workout plan.")
            st.stop()

    try:
        df_plan, meta = solve_week_plan(
            csv_path=CSV_PATH,
            user_level=user_level,
            goal=goal,
            minutes_per_session=minutes,
            workout_days_per_week=workout_days,
            allowed_equipment=allowed_equipment,
            strength_template=strength_template,
            custom_muscles_by_workout_day=custom_muscles_by_day,
            avoid_same_bodypart_consecutive=avoid_consecutive,
            allow_repeat_exercises=allow_repeat_exercises,
        )

        st.subheader("Full week plan")
        for day, sub in df_plan.groupby("Day", sort=False):
            st.markdown(f"**{day}**")
            st.dataframe(sub.drop(columns=["Day"]), use_container_width=True, hide_index=True)

        csv_bytes = df_plan.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download plan as CSV",
            data=csv_bytes,
            file_name=f"plan_{user_level}_{goal}_{workout_days}days_{minutes}min.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception:
        st.error("Could not find optimized workout plan.")
        st.stop()
else:
    st.info("Choose your inputs on the left, then click Generate plan")
