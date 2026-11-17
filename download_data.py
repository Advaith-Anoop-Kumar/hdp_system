from pathlib import Path
from ucimlrepo import fetch_ucirepo

# id=45 is the UCI "Heart Disease" dataset (303 patients from Cleveland)
dataset = fetch_ucirepo(id=45)

df = dataset.data.features.copy()
# Original target "num" is 0-4 (0 = no disease, 1-4 = disease severity).
# We collapse it to binary: 0 = healthy, 1 = heart disease.
df["target"] = (dataset.data.targets["num"] > 0).astype(int)

Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/heart.csv", index=False)
print(df.shape)
print(df.head())