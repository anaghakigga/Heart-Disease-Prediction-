import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load Dataset
heart_data = pd.read_csv('/content/heart.csv')

# Dataset info
print("Dataset shape:", heart_data.shape)
print("\nTarget counts:\n", heart_data['target'].value_counts())
print("\nMean values per class:\n", heart_data.groupby('target').mean())

# Split features & labels
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print("\nTrain/Test Split Shapes:")
print("X:", X.shape, "X_train:", X_train.shape, "X_test:", X_test.shape)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("\nAccuracy score of training data:", training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy score of test data:", test_data_accuracy)

# Confusion Matrix & Report
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, X_test_prediction))
print("\nClassification Report:\n", classification_report(Y_test, X_test_prediction))

# Function for Predictions
def predict_heart_disease(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_as_numpy_array)
    prediction = model.predict(std_data)
    return "The Person has Heart Disease" if prediction[0] == 1 else "The Person does not have Heart Disease"

# Example test input
sample_input = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
print("\nPrediction for sample input:", predict_heart_disease(sample_input))

# Save Model and Scaler
with open('heart_disease_model.sav', 'wb') as f:
    pickle.dump(model, f)

with open('heart_disease_scaler.sav', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel and Scaler saved.")
