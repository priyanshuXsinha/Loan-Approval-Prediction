import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Expanded sample data
sample_data = {
    'Gender': ['Male','Female','Male','Male','Female','Female','Male','Male','Female','Female'],
    'Married': ['Yes','No','Yes','No','Yes','No','Yes','Yes','No','Yes'],
    'Education': ['Graduate','Graduate','Not Graduate','Graduate','Not Graduate','Graduate','Graduate','Not Graduate','Graduate','Graduate'],
    'Self_Employed': ['No','Yes','No','Yes','No','Yes','No','No','Yes','No'],
    'ApplicantIncome': [5000,3000,4000,2500,6000,3500,4200,3900,2800,4500],
    'LoanAmount': [130,70,110,80,150,90,120,100,85,140],
    'Loan_Amount_Term': [360,360,360,360,360,360,360,360,360,360],
    'Credit_History': [1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0],
    'Property_Area': ['Urban','Rural','Urban','Semiurban','Urban','Rural','Semiurban','Urban','Rural','Semiurban'],
    'Loan_Status': ['Y','N','Y','N','Y','Y','N','Y','N','Y']
}
df = pd.DataFrame(sample_data)

# Encode categorical variables
le = LabelEncoder()
for col in ['Gender','Married','Education','Self_Employed','Property_Area']:
    df[col] = le.fit_transform(df[col])

# Map target
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Split features & target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# Save model
pickle.dump(model, open("loan_model.pkl", "wb"))


import pickle
import pandas as pd

# Load saved model
model = pickle.load(open("loan_model.pkl", "rb"))

# User input function
def predict_loan(gender, married, education, self_employed, income, loan_amt, term, credit_hist, property_area):
    # Encode values same as training
    mapping = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 0, 'Not Graduate': 1},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    }

    data = pd.DataFrame([{
        'Gender': mapping['Gender'][gender],
        'Married': mapping['Married'][married],
        'Education': mapping['Education'][education],
        'Self_Employed': mapping['Self_Employed'][self_employed],
        'ApplicantIncome': income,
        'LoanAmount': loan_amt,
        'Loan_Amount_Term': term,
        'Credit_History': credit_hist,
        'Property_Area': mapping['Property_Area'][property_area]
    }])

    pred = model.predict(data)[0]
    return "Approved" if pred == 1 else "Rejected"

# Example run
print(predict_loan(
    gender="Male",
    married="Yes",
    education="Graduate",
    self_employed="No",
    income=5000,
    loan_amt=120,
    term=360,
    credit_hist=1.0,
    property_area="Urban"
))
