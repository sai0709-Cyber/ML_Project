import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Step 1: Load CSV
df = pd.read_csv("ml_dataset.csv")

# Step 2: Rename columns for clarity
df = df.rename(columns={
    "param1": "length",
    "param2": "width",
    "param3": "height",
    "param4": "pressure",
    "max_deformation": "tot_def_max"
})
print("✅ Renamed columns:", df.columns.tolist())

# Step 3: Fill missing width/height for spheres
df[["width", "height"]] = df[["width", "height"]].fillna(0.0)

# Step 4: Check unique shape values BEFORE encoding
if "shape" in df.columns:
    print("✅ Unique shape values:", df["shape"].unique())
    df["shape"] = df["shape"].astype("category")
else:
    print("⚠️ 'shape' column missing — check input data")
    exit()

# Step 5: Encode shape as one-hot
df = pd.get_dummies(df, columns=["shape"], drop_first=False)
print("✅ Columns after one-hot encoding:", df.columns.tolist())

# Step 6: Check for shape columns
if "shape_sphere" not in df.columns:
    df["shape_sphere"] = 0  # assume all are 'rect'

# Step 7: Prepare features and target
X = df[["length", "width", "height", "pressure", "shape_sphere"]]
y = df["tot_def_max"]

# Step 8: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained.")

# Step 10: Save model
joblib.dump(model, "beam_model.pkl")
print("✅ Model saved as beam_model.pkl")
