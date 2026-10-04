"""
AgriSense - model training script
Run: python train_models.py

This trains the Random Forest pest-risk model from the supplied demo CSV.
The CSV is synthetic/demo data, not field-validated agricultural data.
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "AgriSense_demo_dataset.csv")

features = [
    "Crop", "Temperature_C", "Humidity_percent", "Rainfall_mm",
    "Soil_Type", "Soil_pH", "Nitrogen_mgkg", "Phosphorus_mgkg",
    "Potassium_mgkg", "Water_Availability"
]
target = "Pest_Risk"

X, y = df[features], df[target]
categorical = ["Crop", "Soil_Type", "Water_Availability"]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\\n=== AGRISENSE RANDOM FOREST ===")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Accuracy         : {accuracy_score(y_test, pred)*100:.2f}%")
print(f"Precision        : {precision_score(y_test, pred, average='weighted')*100:.2f}%")
print(f"Recall           : {recall_score(y_test, pred, average='weighted')*100:.2f}%")
print(f"F1 Score         : {f1_score(y_test, pred, average='weighted')*100:.2f}%")
print("\\nClassification Report:\\n", classification_report(y_test, pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, pred))

joblib.dump(model, ROOT / "models" / "agrisense_pest_risk_model.pkl")
print("\\nModel saved to models/agrisense_pest_risk_model.pkl")
