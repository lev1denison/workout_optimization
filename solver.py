import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

GOALS_ALLOWED = ["Strength", "Cardio", "Stretching", "Plyometrics"]
LEVELS = ["Beginner", "Intermediate", "Expert"]

EQUIPMENT_ALL = [
    "Bands", "Barbell", "Kettlebells", "Dumbbell", "Body Only",
    "Cable", "Exercise Ball", "E-Z Curl Bar", "Foam Roll",
    "Machine", "Medicine Ball", "Other"
]

NO_EQUIP_OPTION = "No Equipment Available"

STRENGTH_TEMPLATES = [
    "Upper/Lower",
    "Full Body",
    "Heavy/Light/Medium",
    "Push/Pull/Leg",
    "Custom",
]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Strength template muscle groups
PUSH = {"Chest", "Shoulders", "Triceps"}
PULL = {"Lats", "Middle Back", "Lower Back", "Traps", "Biceps", "Forearms"}
LEGS = {"Quadriceps", "Hamstrings", "Glutes", "Calves", "Adductors", "Abductors"}
CORE = {"Abdominals", "Neck"}

UPPER = PUSH | PULL | CORE
LOWER = LEGS


def allowed_levels_for_user(user_level: str) -> set:
    if user_level == "Beginner":
        return {"Beginner"}
    if user_level == "Intermediate":
        return {"Beginner", "Intermediate"}
    return {"Beginner", "Intermediate", "Expert"}


def estimate_time(row) -> int:
    t = 8
    if row["Type"] == "Strength":
        if row["Level"] == "Beginner":
            t = 10
        elif row["Level"] == "Intermediate":
            t = 15
        else:
            t = 20
    elif row["Type"] in ["Plyometrics", "Cardio"]:
        t = 10
    elif row["Type"] == "Stretching":
        t = 5
    else:
        t = 10
    return int(t)


def default_exercise_bounds(goal: str, minutes: int) -> tuple[int, int]:
    minutes = max(10, int(minutes))
    if goal == "Strength":
        return max(3, minutes // 20), max(5, min(10, minutes // 8))
    if goal == "Cardio":
        return 1, max(2, min(6, minutes // 10))
    if goal == "Stretching":
        return max(3, minutes // 10), max(6, min(16, minutes // 4))
    if goal == "Plyometrics":
        return max(2, minutes // 20), max(4, min(10, minutes // 8))
    return 1, 12


def spread_workout_days(k: int) -> list[int]:
    """
    Choose k indices in [0..6] spread fairly evenly.
    Deterministic so Custom template maps consistently.
    """
    k = int(k)
    if k <= 0:
        return []
    if k >= 7:
        return list(range(7))

    step = 7 / k
    idx = sorted({int(math.floor(i * step)) for i in range(k)})

    while len(idx) < k:
        cand = max(idx) + 1
        if cand > 6:
            cand = 6
        if cand not in idx:
            idx.append(cand)
        idx = sorted(idx)
        if len(idx) == 7:
            break

    return idx[:k]


def _present_set(present_bodyparts: list[str], candidate: set[str]) -> set[str]:
    present = set(present_bodyparts)
    return {m for m in candidate if m in present}


def allowed_muscles_for_strength_template(template: str, present_bodyparts: list[str], workout_slot: int):
    """
    Return:
      - None => no restriction (Full Body, Heavy/Light/Medium)
      - set[str] => allowed body parts for that workout slot
    """
    template = template.strip()

    if template in ["Full Body", "Heavy/Light/Medium"]:
        return None

    if template == "Push/Pull/Leg":
        pattern = [PUSH, PULL, LEGS]
        return _present_set(present_bodyparts, pattern[workout_slot % 3])

    if template == "Upper/Lower":
        pattern = [UPPER, LOWER]
        return _present_set(present_bodyparts, pattern[workout_slot % 2])

    # Custom handled separately
    return None


def intensity_factor_for_hlm(workout_slot: int) -> float:
    # Heavy / Medium / Light repeating
    pattern = [1.00, 0.85, 0.70]
    return pattern[workout_slot % 3]


def solve_week_plan(
    csv_path: str,
    user_level: str,
    goal: str,
    minutes_per_session: int,
    workout_days_per_week: int,
    allowed_equipment: list[str] | None = None,

    strength_template: str | None = None,
    custom_muscles_by_workout_day: list[list[str]] | None = None,  # len=workout_days_per_week for Custom

    avoid_same_bodypart_consecutive: bool = True,
    allow_repeat_exercises: bool = False,
    seed: int | None = None,
):
    """
    Returns:
      df_plan: DataFrame rows describing each day (rest day rows included)
      meta: dict
    """
    if goal not in GOALS_ALLOWED:
        raise ValueError(f"goal must be one of {GOALS_ALLOWED}")
    if user_level not in LEVELS:
        raise ValueError(f"user_level must be one of {LEVELS}")

    workout_days_per_week = int(workout_days_per_week)
    if not (0 <= workout_days_per_week <= 7):
        raise ValueError("workout_days_per_week must be between 0 and 7")

    minutes_per_session = max(10, int(minutes_per_session))

    # If 0 workout days, return a pure rest-week without solving
    if workout_days_per_week == 0:
        rows = []
        for d in range(7):
            rows.append({
                "Day": DAY_NAMES[d],
                "WorkoutDay": False,
                "Title": "Rest / Recovery",
                "BodyPart": "",
                "Equipment": "",
                "Type": "",
                "Level": "",
                "TimeMin": 0,
                "Rating": 0.0,
            })
        df_plan = pd.DataFrame(rows)
        meta = {
            "objective": 0.0,
            "status": None,
            "goal": goal,
            "user_level": user_level,
            "minutes_per_session": minutes_per_session,
            "workout_days_per_week": workout_days_per_week,
            "workout_weekday_idx": [],
            "strength_template": strength_template if goal == "Strength" else None,
            "allow_repeat_exercises": allow_repeat_exercises,
        }
        return df_plan, meta

    if goal == "Strength":
        if strength_template is None or strength_template not in STRENGTH_TEMPLATES:
            raise ValueError(f"When goal=Strength, strength_template must be one of {STRENGTH_TEMPLATES}")

        if strength_template == "Custom":
            if custom_muscles_by_workout_day is None:
                raise ValueError("Custom template requires custom_muscles_by_workout_day.")
            if len(custom_muscles_by_workout_day) != workout_days_per_week:
                raise ValueError("custom_muscles_by_workout_day length must equal workout_days_per_week.")
            if any((m is None or len(m) == 0) for m in custom_muscles_by_workout_day):
                raise ValueError("Each custom workout day must include at least one selected muscle/body part.")

    # --- Load + clean ---
    data = pd.read_csv(csv_path)
    data = data[~data["BodyPart"].isna()].copy()
    data["Rating"] = data["Rating"].fillna(data["Rating"].median())

    # Filter by goal type
    data = data[data["Type"] == goal].copy()

    # Filter by level eligibility
    allowed_lvls = allowed_levels_for_user(user_level)
    data = data[data["Level"].isin(allowed_lvls)].copy()

    # Equipment filtering rules:
    # - If allowed_equipment is empty/None: no restriction (all equipment)
    # - If contains "No Equipment Available": restrict to Body Only only
    # - Else: selected equipment + Body Only always allowed
    if allowed_equipment is not None and len(allowed_equipment) > 0:
        if NO_EQUIP_OPTION in allowed_equipment:
            equip_set = {"Body Only"}
        else:
            equip_set = set(allowed_equipment)
            equip_set.add("Body Only")
        data = data[data["Equipment"].isin(equip_set)].copy()

    if data.empty:
        raise ValueError("No exercises match your filters (goal/level/equipment). Try relaxing filters.")

    data = data.reset_index(drop=True)
    data["TimeMin"] = data.apply(estimate_time, axis=1)

    # --- Sets ---
    E = list(data.index)
    D = list(range(7))
    M = sorted(data["BodyPart"].unique())

    rating = data["Rating"].to_dict()
    time_min = data["TimeMin"].to_dict()
    muscle_of = data["BodyPart"].to_dict()

    muscle_to_exs = {m: [] for m in M}
    for e in E:
        muscle_to_exs[muscle_of[e]].append(e)

    # Determine workout weekdays (spread)
    workout_weekday_idx = spread_workout_days(workout_days_per_week)
    workout_weekday_set = set(workout_weekday_idx)
    workout_day_to_slot = {d: i for i, d in enumerate(workout_weekday_idx)}

    # --- Model ---
    model = gp.Model("workout_week_plan")
    model.Params.OutputFlag = 0
    if seed is not None:
        model.Params.Seed = int(seed)

    x = model.addVars(E, D, vtype=GRB.BINARY, name="x")
    y = model.addVars(M, D, vtype=GRB.BINARY, name="y")

    # Objective
    model.setObjective(
        gp.quicksum(rating[e] * x[e, d] for e in E for d in D),
        GRB.MAXIMIZE
    )

    # Link y <-> x
    for m in M:
        for d in D:
            for e in muscle_to_exs[m]:
                model.addConstr(y[m, d] >= x[e, d], name=f"link_lo_{m}_{e}_{d}")
            model.addConstr(
                y[m, d] <= gp.quicksum(x[e, d] for e in muscle_to_exs[m]),
                name=f"link_hi_{m}_{d}"
            )

    # Day caps and rest days
    E_min_base, E_max_base = default_exercise_bounds(goal, minutes_per_session)

    for d in D:
        if d not in workout_weekday_set:
            model.addConstr(gp.quicksum(x[e, d] for e in E) == 0, name=f"rest_{d}")
            continue

        factor = 1.0
        if goal == "Strength" and strength_template == "Heavy/Light/Medium":
            slot = workout_day_to_slot[d]
            factor = intensity_factor_for_hlm(slot)

        T_cap = int(round(minutes_per_session * factor))
        E_min = max(1, int(round(E_min_base * factor)))
        E_max = max(E_min, int(round(E_max_base * factor)))

        model.addConstr(
            gp.quicksum(time_min[e] * x[e, d] for e in E) <= T_cap,
            name=f"time_cap_{d}"
        )
        model.addConstr(gp.quicksum(x[e, d] for e in E) >= E_min, name=f"min_ex_{d}")
        model.addConstr(gp.quicksum(x[e, d] for e in E) <= E_max, name=f"max_ex_{d}")

    # Exercise repetition toggle
    if not allow_repeat_exercises:
        for e in E:
            model.addConstr(gp.quicksum(x[e, d] for d in D) <= 1, name=f"once_{e}")

    # Avoid consecutive body part toggle
    if avoid_same_bodypart_consecutive and workout_days_per_week >= 2:
        for m in M:
            for d in range(6):
                model.addConstr(y[m, d] + y[m, d + 1] <= 1, name=f"recovery_{m}_{d}")

    # Strength templates
    if goal == "Strength" and workout_days_per_week > 0:
        for d in workout_weekday_idx:
            slot = workout_day_to_slot[d]

            if strength_template == "Custom":
                allowed = set(custom_muscles_by_workout_day[slot]).intersection(set(M))
                if not allowed:
                    raise ValueError("Custom template muscles not found after filtering. Try different filters.")

                for e in E:
                    if muscle_of[e] not in allowed:
                        model.addConstr(x[e, d] == 0, name=f"custom_block_{e}_{d}")

            else:
                allowed = allowed_muscles_for_strength_template(strength_template, M, slot)
                if allowed is not None:
                    for e in E:
                        if muscle_of[e] not in allowed:
                            model.addConstr(x[e, d] == 0, name=f"tpl_block_{e}_{d}")

    model.optimize()

    if model.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        # Raise a clean error; app.py will show a friendly message without stack trace.
        raise RuntimeError(f"No feasible solution. Status={model.status}")

    # Extract
    rows = []
    for d in D:
        if d not in workout_weekday_set:
            rows.append({
                "Day": DAY_NAMES[d],
                "WorkoutDay": False,
                "Title": "Rest / Recovery",
                "BodyPart": "",
                "Equipment": "",
                "Type": "",
                "Level": "",
                "TimeMin": 0,
                "Rating": 0.0,
            })
            continue

        for e in E:
            if x[e, d].X > 0.5:
                rows.append({
                    "Day": DAY_NAMES[d],
                    "WorkoutDay": True,
                    "Title": data.loc[e, "Title"],
                    "BodyPart": data.loc[e, "BodyPart"],
                    "Equipment": data.loc[e, "Equipment"],
                    "Type": data.loc[e, "Type"],
                    "Level": data.loc[e, "Level"],
                    "TimeMin": int(data.loc[e, "TimeMin"]),
                    "Rating": float(data.loc[e, "Rating"]),
                })

    df_plan = pd.DataFrame(rows)

    meta = {
        "objective": float(model.objVal),
        "status": int(model.status),
        "goal": goal,
        "user_level": user_level,
        "minutes_per_session": minutes_per_session,
        "workout_days_per_week": workout_days_per_week,
        "workout_weekday_idx": workout_weekday_idx,
        "strength_template": strength_template if goal == "Strength" else None,
        "allow_repeat_exercises": allow_repeat_exercises,
    }

    return df_plan, meta
