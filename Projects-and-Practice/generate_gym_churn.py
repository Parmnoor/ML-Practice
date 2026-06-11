"""
Synthetic Gym Membership Churn Dataset Generator
=================================================
Key design principle: churn is NOT random. It is driven by a latent
"engagement" factor plus contract type, tenure, and recency of last visit,
so a deep learning model has a genuine signal to learn. Demographic fields
(Name, Address, Phone_Number) are intentionally non-predictive PII-style
noise -- they exist so you can practice dropping them during cleaning.

Reproducible via SEED.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

SEED = 42
N = 50_000
TODAY = datetime(2026, 6, 10)

rng = np.random.default_rng(SEED)
fake = Faker()
Faker.seed(SEED)

# ----------------------------------------------------------------------
# 1. Latent engagement score  (the hidden driver of behavior + churn)
#    Higher engagement -> more visits, longer sessions, lower churn.
# ----------------------------------------------------------------------
engagement = rng.beta(2.2, 2.2, N)          # 0..1, centered, realistic spread

# ----------------------------------------------------------------------
# 2. Demographics
# ----------------------------------------------------------------------
age = np.clip(rng.normal(35, 12, N), 16, 80).round().astype(int)
gender = rng.choice(["Male", "Female", "Other"], N, p=[0.48, 0.48, 0.04])

# ----------------------------------------------------------------------
# 3. Membership type  (drives churn strongly: monthly churns most)
# ----------------------------------------------------------------------
membership_type = rng.choice(
    ["Monthly", "Quarterly", "Yearly"], N, p=[0.50, 0.25, 0.25]
)

# ----------------------------------------------------------------------
# 4. Tenure / Join date
#    Joined sometime in the last ~3 years.
# ----------------------------------------------------------------------
tenure_days = rng.integers(15, 365 * 3, N)
join_date = np.array([TODAY - timedelta(days=int(d)) for d in tenure_days])
tenure_months = tenure_days / 30.0

# ----------------------------------------------------------------------
# 5. Behavioral features derived from engagement (+ noise)
# ----------------------------------------------------------------------
# Visits per month: engaged members visit far more often
visits_per_month = np.clip(
    engagement * 22 + rng.normal(0, 3, N), 0, 30
).round(1)

# Session duration (minutes): mild positive link to engagement
avg_workout_duration = np.clip(
    engagement * 50 + rng.normal(45, 12, N), 10, 150
).round().astype(int)

# Favorite exercise: strength-leaning members lift more weight
exercises = ["Weightlifting", "Cardio", "Yoga", "CrossFit",
             "Swimming", "Cycling", "HIIT", "Pilates"]
favorite_exercise = rng.choice(exercises, N,
                               p=[0.22, 0.20, 0.12, 0.12, 0.08, 0.10, 0.10, 0.06])
is_strength = np.isin(favorite_exercise, ["Weightlifting", "CrossFit", "HIIT"])

# Calories burned: function of duration + intensity noise
intensity = rng.normal(7.0, 1.5, N)                       # kcal per minute
avg_calories = np.clip(
    avg_workout_duration * intensity, 80, 1400
).round().astype(int)

# Total weight lifted (kg): strength folks + more visits -> much higher
weight_per_session = np.where(is_strength,
                              rng.normal(3500, 900, N),
                              rng.normal(900, 400, N))
sessions_total = np.clip(visits_per_month * tenure_months, 1, None)
total_weight_lifted = np.clip(
    weight_per_session * sessions_total / 30.0, 0, None
).round().astype(int)

# ----------------------------------------------------------------------
# 6. Churn probability  (the learnable target signal)
# ----------------------------------------------------------------------
# Start from a log-odds baseline, then add effects.
logit = np.full(N, -1.3)

# Engagement is the dominant driver (low engagement -> high churn)
logit += (0.5 - engagement) * 6.0

# Contract type effect
contract_effect = np.select(
    [membership_type == "Monthly",
     membership_type == "Quarterly",
     membership_type == "Yearly"],
    [0.9, 0.1, -1.0]
)
logit += contract_effect

# Tenure effect: brand-new members churn more (the "honeymoon cliff")
logit += np.where(tenure_months < 3, 1.1, 0.0)
logit += np.where(tenure_months > 24, -0.6, 0.0)   # long-tenure loyalty

# "Visit cliff": very low monthly visits is a strong red flag
logit += np.where(visits_per_month < 3, 1.4, 0.0)

# Mild age effect: younger members slightly more flighty
logit += (30 - age) * 0.015

# Add noise so the boundary isn't perfectly separable
logit += rng.normal(0, 0.6, N)

churn_prob = 1 / (1 + np.exp(-logit))
churn = (rng.random(N) < churn_prob).astype(int)

# ----------------------------------------------------------------------
# 7. Last visit date  (must be CONSISTENT with churn)
#    Churned members haven't visited in a while; active members recently.
# ----------------------------------------------------------------------
days_since_visit = np.where(
    churn == 1,
    rng.integers(7, 180, N),    # churned: some visited recently (overlap with active)
    rng.integers(0, 60, N)      # active: some haven't come in a while (overlap with churned)
)
# Last visit can't predate joining
days_since_visit = np.minimum(days_since_visit, tenure_days - 1)
last_visit_date = np.array(
    [TODAY - timedelta(days=int(d)) for d in days_since_visit]
)

# ----------------------------------------------------------------------
# 8. Assemble dataframe
# ----------------------------------------------------------------------
df = pd.DataFrame({
    "Member_ID": [f"GYM{100000 + i}" for i in range(N)],
    "Name": [fake.name() for _ in range(N)],
    "Age": age,
    "Gender": gender,
    "Address": [fake.address().replace("\n", ", ") for _ in range(N)],
    "Phone_Number": [fake.phone_number() for _ in range(N)],
    "Membership_Type": membership_type,
    "Join_Date": [d.strftime("%Y-%m-%d") for d in join_date],
    "Last_Visit_Date": [d.strftime("%Y-%m-%d") for d in last_visit_date],
    "Favorite_Exercise": favorite_exercise,
    "Avg_Workout_Duration_Min": avg_workout_duration,
    "Avg_Calories_Burned": avg_calories,
    "Total_Weight_Lifted_kg": total_weight_lifted,
    "Visits_Per_Month": visits_per_month,
    "Churn": np.where(churn == 1, "Yes", "No"),
})

# Inject a little realistic messiness for cleaning practice (~1.5% missing)
for col in ["Avg_Calories_Burned", "Avg_Workout_Duration_Min", "Age"]:
    mask = rng.random(N) < 0.015
    df.loc[mask, col] = np.nan

out_path = "gym_churn_synthetic.csv"
df.to_csv(out_path, index=False)

print(f"Rows: {len(df):,}")
print(f"Churn rate: {(churn.mean()*100):.1f}%")
print(f"Columns: {list(df.columns)}")
print("\nChurn by membership type:")
print(df.groupby("Membership_Type")["Churn"].apply(lambda s: (s == "Yes").mean().round(3)))
print("\nSample rows:")
print(df.head(3).to_string())
