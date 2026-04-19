import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("--- Starting Training Script (Bulletproof Version) ---")

filename = 'data.csv'
if not os.path.exists(filename):
    print(f"❌ ERROR: {filename} NOT FOUND!")
else:
    # 1. Load Data
    df = pd.read_csv(filename)
    print(f"✅ Data loaded. Rows: {len(df)}")

    # 2. Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # 3. Automatic Encoding (The Fix)
    print("⏳ Converting all text to numbers using One-Hot Encoding...")
    # This automatically finds all text columns and converts them to 0s and 1s
    X = df.drop('Churn', axis=1)
    X = pd.get_dummies(X) 
    
    # Convert 'Yes'/'No' in Churn to 1/0
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 4. Training
    print("⏳ Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Saving
    joblib.dump(model, 'churn_model.pkl')
    # Save the column names so the App knows the exact order
    joblib.dump(X.columns.tolist(), 'features.pkl')

    print(f"🚀 SUCCESS! Model trained with {len(X.columns)} features.")