# 🔹 Step 1: Load data
import pandas as pd

print("STARTING SCRIPT")

df = pd.read_csv("C:/Users/Sakshi/fraud-detection-project/data/creditcard.csv")

print("Dataset loaded!")
print(df.shape)
print(df["Class"].value_counts())


# 🔹 Step 2: Handle imbalance
from sklearn.utils import resample

fraud = df[df.Class == 1]
normal = df[df.Class == 0]

normal_sample = resample(normal,
                         replace=False,
                         n_samples=len(fraud)*2,
                         random_state=42)

df_balanced = pd.concat([fraud, normal_sample])

print("\nBalanced dataset:")
print(df_balanced["Class"].value_counts())


# 🔹 Step 3: Split data
from sklearn.model_selection import train_test_split

X = df_balanced.drop("Class", axis=1)
y = df_balanced["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData split done!")


# 🔹 Step 4: Train model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Model trained!")


# 🔹 Step 5: Evaluate
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))


# 🔹 Step 6: Save model
import joblib

joblib.dump(model, "model/fraud_model.pkl")

print("Model saved!")
# Save a sample transaction
df.sample(1).drop("Class", axis=1).to_csv("model/sample_input.csv", index=False)
