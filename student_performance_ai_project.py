"""
Student Performance Analytics and AI-Based Performance Prediction
Demo project for a Data Analytics & AI internship.
Requirements: pandas, matplotlib, scikit-learn
Run: python student_performance_ai_project.py
NOTE: The bundled dataset is synthetic demo data, not real student records.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "student_performance_sample_dataset.csv"
OUTPUT_DIR = BASE / "project_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
features = ["Study_Hours_Per_Day", "Attendance_Percent",
            "Assignment_Score_10", "Quiz_Score_10", "Sleep_Hours"]
target = "Final_Score_Percent"

# 1. Data quality and descriptive analytics
print("\\n=== DATA OVERVIEW ===")
print(df.head())
print("\\nShape:", df.shape)
print("\\nMissing values:\\n", df.isnull().sum())
print("\\nDescriptive statistics:\\n", df[features + [target]].describe().round(2))

# 2. Create a transparent support category (not a formal academic decision)
df["Support_Flag"] = df[target].apply(
    lambda score: "Needs support" if score < 60 else
    ("Monitor progress" if score < 75 else "On track")
)
print("\\nSupport categories:\\n", df["Support_Flag"].value_counts())

# 3. Visualizations
plt.figure(figsize=(7, 5))
plt.scatter(df["Study_Hours_Per_Day"], df[target], alpha=0.8)
plt.title("Study Time vs Final Score")
plt.xlabel("Study hours per day")
plt.ylabel("Final score (%)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "study_hours_vs_score.png", dpi=160)
plt.close()

plt.figure(figsize=(7, 5))
df["Support_Flag"].value_counts().reindex(
    ["Needs support", "Monitor progress", "On track"], fill_value=0
).plot(kind="bar")
plt.title("Students by Support Category")
plt.xlabel("Category")
plt.ylabel("Number of students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "support_categories.png", dpi=160)
plt.close()

# 4. AI/ML model: Random Forest regression
X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("\\n=== MODEL EVALUATION (held-out demo split) ===")
print(f"Mean Absolute Error: {mae:.2f} percentage points")
print(f"R-squared: {r2:.3f}")
print("\\nFeature importance:")
print(pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).round(3))

# 5. Save predictions for the held-out test records
results = X_test.copy()
results["Actual_Score"] = y_test
results["Predicted_Score"] = predictions.round(1)
results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

# 6. Example prediction; replace values with a consented, legitimate use case
example = pd.DataFrame([{
    "Study_Hours_Per_Day": 5,
    "Attendance_Percent": 75,
    "Assignment_Score_10": 7,
    "Quiz_Score_10": 6,
    "Sleep_Hours": 7
}])
print("\\nExample predicted score:", round(float(model.predict(example)[0]), 1), "%")
print("\\nOutputs saved in:", OUTPUT_DIR)
print("Reminder: predictions are estimates, not labels of a student's ability.")
