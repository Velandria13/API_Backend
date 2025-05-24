import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# Ensure the output directory exists
os.makedirs("trained_data", exist_ok=True)

# Load dataset
df = pd.read_csv('csv/Student_performance_data _.csv')

# Simulate or define required columns
df['PassingScore'] = 2.0
np.random.seed(42)
df['HoursSleepDaily'] = np.random.normal(loc=7, scale=1.5, size=len(df)).clip(3, 10)

def attendance_level(absences):
    if absences <= 3:
        return 'Excellent'
    elif absences <= 10:
        return 'Good'
    elif absences <= 20:
        return 'Fair'
    else:
        return 'Poor'

df['AttendanceLevel'] = df['Absences'].apply(attendance_level)

# Target: Pass if GPA >= PassingScore
df['PassFail'] = df.apply(lambda row: 'Pass' if row['GPA'] >= row['PassingScore'] else 'Fail', axis=1)

# Select inputs and output
X_raw = df[['StudyTimeWeekly', 'HoursSleepDaily', 'AttendanceLevel', 'PassingScore']]
y_raw = df['PassFail']

# Encode categorical input and output
X = pd.get_dummies(X_raw)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Save encoders
joblib.dump(label_encoder, 'trained_data/passfail_encoder.pkl')
joblib.dump(X.columns.tolist(), 'trained_data/passfail_features.pkl')

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'trained_data/model_passfail.pkl')

print("✅ Training complete. Model saved to 'trained_data/'")