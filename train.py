from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------- Step 2: load and inspect ----------
df = pd.read_csv("data/raw/heart.csv")
print("Missing values per column:")
print(df.isna().sum(), "\n")          # 'ca' and 'thal' have a few missing values

X = df.drop(columns="target")
y = df["target"]

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


# ---------- Step 3: models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    ),
}
pipelines = {
    name: Pipeline([("prep", make_preprocessor()), ("model", m)])
    for name, m in models.items()
}

# Hold out 20% as a final test set the models never see during selection
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- Step 4a: compare models with cross-validation ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = {}
for name, pipe in pipelines.items():
    res = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=["accuracy", "recall", "roc_auc"],
    )
    scores[name] = res["test_roc_auc"].mean()
    print(f"{name:20s} acc={res['test_accuracy'].mean():.3f} "
          f"recall={res['test_recall'].mean():.3f} "
          f"auc={res['test_roc_auc'].mean():.3f}")

best_name = max(scores, key=scores.get)
print(f"\nBest model: {best_name}\n")

# ---------- Step 4b: final evaluation on the held-out test set ----------
best = pipelines[best_name].fit(X_train, y_train)
pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred, target_names=["No disease", "Disease"]))
print("ROC AUC:", round(roc_auc_score(y_test, proba), 3))
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"Confusion matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"False negatives (sick people we missed): {fn}")

# ---------- Save for the app ----------
Path("models").mkdir(exist_ok=True)
joblib.dump(best, "models/model.joblib")
print("Saved models/model.joblib")