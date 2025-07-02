import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- File paths ---
csv_path = "ml_dataset.csv"
model_path = "beam_model.pkl"

# --- Load dataset ---
df = pd.read_csv(csv_path)

# --- Rename columns to match training format ---
df = df.rename(columns={
    "param1": "length",
    "param2": "width",
    "param3": "height",
    "param4": "pressure",
    "tot_def_max":"max_deformation"


    
})

# --- Fill missing width/height values (for spheres) ---
df[["width", "height"]] = df[["width", "height"]].fillna(0.0)

# --- One-hot encode shape column ---
if "shape" in df.columns:
    df = pd.get_dummies(df, columns=["shape"], drop_first=False)
else:
    raise ValueError("🛑 'shape' column missing in dataset")

# --- Ensure both shape_sphere and shape_rect exist ---
for col in ["shape_sphere"]:
    if col not in df.columns:
        df[col] = 0

# --- Ensure feature order matches training ---
expected_features = ["length", "width", "height", "pressure", "shape_sphere"]
missing = [col for col in expected_features if col not in df.columns]
for col in missing:
    df[col] = 0  # Add missing with 0s

print("🧾 Available columns:", df.columns.tolist())


df = df[expected_features]
X = df
y = df["max_deformation"]

# --- Load trained model ---
model = joblib.load(model_path)

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Predict ---
y_pred = model.predict(X_test)

# --- Evaluation ---
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"🔹 R² Score           : {r2:.4f}")
print(f"🔹 Mean Squared Error : {mse:.4e}")

# --- Scatter plot ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Deformation")
plt.ylabel("Predicted Deformation")
plt.title("Predicted vs Actual Deformation")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Histogram of predictions ---
plt.hist(y_test, bins=30, alpha=0.7, label="Actual")
plt.hist(y_pred, bins=30, alpha=0.7, label="Predicted")
plt.title("Distribution: Actual vs Predicted")
plt.xlabel("Deformation")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
