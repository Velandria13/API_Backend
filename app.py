from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load('trained_data/model_passfail.pkl')
features = joblib.load('trained_data/passfail_features.pkl')
label_encoder = joblib.load('trained_data/passfail_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Derive attendance level
    absences = data.get('Absences', 0)
    if absences <= 3:
        attendance_level = 'AttendanceLevel_Excellent'
    elif absences <= 10:
        attendance_level = 'AttendanceLevel_Good'
    elif absences <= 20:
        attendance_level = 'AttendanceLevel_Fair'
    else:
        attendance_level = 'AttendanceLevel_Poor'

    # Construct input DataFrame
    input_dict = {
        'StudyTimeWeekly': data['StudyTimeWeekly'],
        'HoursSleepDaily': data['HoursSleepDaily'],
        'PassingScore': data['PassingScore'],
        attendance_level: 1
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=features, fill_value=0)

    pred_encoded = model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return jsonify({"Prediction": pred_label})

if __name__ == '__main__':
    app.run(debug=True)