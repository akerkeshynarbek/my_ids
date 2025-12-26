import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

#loading the data(i'm using the Friday PortScan file)
print("Loading PortScan dataset...")
data_path = "data/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
df = pd.read_csv(data_path)

#cleaning data
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# feature selection part
# for a basic research project, chose 4 features
features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']
X = df[features]
y = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1) # Binary Classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL 1: BASELINE (Decision Tree)
print("Training Model 1: Baseline Decision Tree...")
baseline = DecisionTreeClassifier(max_depth=3) # simple and shallow
baseline.fit(X_train, y_train)
base_preds = baseline.predict(X_test)

#MODEL 2: OPTIMIZED (random Forest)
print("Training Model 2: Optimized Random Forest...")
optimized = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
optimized.fit(X_train, y_train)
opt_preds = optimized.predict(X_test)

# evlauation and comparison part
print("\n" + "="*20 + " RESEARCH RESULTS " + "="*20)
print(f"BASELINE Accuracy: {accuracy_score(y_test, base_preds):.4f}")
print(f"OPTIMIZED Accuracy: {accuracy_score(y_test, opt_preds):.4f}")

print("\n--- Baseline Classification Report ---")
print(classification_report(y_test, base_preds))

print("\n--- Optimized Classification Report ---")
print(classification_report(y_test, opt_preds))

# Saving the best model
joblib.dump(optimized, 'optimized_ids_model.pkl')
print("\nSuccess: Optimized model saved as 'optimized_ids_model.pkl'")
