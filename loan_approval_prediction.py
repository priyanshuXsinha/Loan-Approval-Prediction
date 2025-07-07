# Loan Approval Prediction System (Starter Code)

# Step 1: Load and preprocess dummy loan data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Sample data
sample_data = {
    'Gender': ['Male', 'Female'],
    'Married': ['Yes', 'No'],
    'Education': ['Graduate', 'Not Graduate'],
    'Self_Employed': ['No', 'Yes'],
    'ApplicantIncome': [5000, 3000],
    'LoanAmount': [130, 70],
    'Loan_Amount_Term': [360, 360],
    'Credit_History': [1.0, 0.0],
    'Property_Area': ['Urban', 'Rural'],
    'Loan_Status': ['Y', 'N']
}
df = pd.DataFrame(sample_data)

# Preprocess
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
    df[col] = le.fit_transform(df[col])

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model
pickle.dump(model, open("loan_model.pkl", "wb"))
