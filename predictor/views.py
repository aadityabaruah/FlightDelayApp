from django.shortcuts import render
import joblib
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

MODEL_PATH = 'C:/Users/Aaditya/PycharmProjects/FlightDelayApp/'

# Load all the trained models and the scaler
lin_reg = joblib.load(os.path.join(MODEL_PATH, 'lin_reg_model.pkl'))
log_reg = joblib.load(os.path.join(MODEL_PATH, 'log_reg_model.pkl'))
knn = joblib.load(os.path.join(MODEL_PATH, 'knn_model.pkl'))
decision_tree = joblib.load(os.path.join(MODEL_PATH, 'decision_tree_model.pkl'))
random_forest = joblib.load(os.path.join(MODEL_PATH, 'random_forest_model.pkl'))
ann = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'ann_model_sample_tf'))
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))


def get_prediction_from_models(data):
    # Standardize the input data
    data_scaled = scaler.transform(data)

    # Predict using the trained models
    predictions = {
        'Linear Regression (Delay Score)': lin_reg.predict(data_scaled)[0],
        'Logistic Regression (Probability of Delay)': log_reg.predict_proba(data_scaled)[:, 1][0],
        'KNN (Probability of Delay)': knn.predict_proba(data_scaled)[:, 1][0],
        'Decision Tree (Probability of Delay)': decision_tree.predict_proba(data_scaled)[:, 1][0],
        'Random Forest (Probability of Delay)': random_forest.predict_proba(data_scaled)[:, 1][0],
        'ANN (Probability of Delay)': ann.predict(data_scaled)[0][0]
    }

    return predictions


def predict(request):
    if request.method == 'POST':
        # Extract form data
        airline = int(request.POST['airline'])
        airport = int(request.POST['airport'])
        date_str = request.POST['date']
        departure_time = request.POST['departure_time']
        arrival_time = request.POST['arrival_time']
        flight_duration = float(request.POST['flight_duration'])
        flight_distance = float(request.POST['flight_distance'])

        # Convert date to day and month
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        day = date_obj.day

        # Convert times to minutes since midnight
        departure_hour, departure_minute = map(int, departure_time.split(':'))
        arrival_hour, arrival_minute = map(int, arrival_time.split(':'))
        scheduled_departure = departure_hour * 60 + departure_minute
        scheduled_arrival = arrival_hour * 60 + arrival_minute

        # Create a dataframe for prediction
        data = {
            'DAY_OF_WEEK': [day],
            'AIRLINE': [airline],
            'FLIGHT_NUMBER': [0],  # Placeholder, as it's not in the form
            'ORIGIN_AIRPORT': [airport],
            'DESTINATION_AIRPORT': [airport],  # Placeholder
            'SCHEDULED_DEPARTURE': [scheduled_departure],
            'SCHEDULED_TIME': [flight_duration],
            'SCHEDULED_ARRIVAL': [scheduled_arrival]
        }

        input_data = pd.DataFrame(data)
        predictions = get_prediction_from_models(input_data)

        return render(request, 'result.html', {'predictions': predictions})

    # For GET requests
    # Assuming you have predefined lists of airlines and airports. Adjust accordingly.
    airlines = list(range(1, 15))  # Placeholder
    airports = list(range(1, 300))  # Placeholder

    return render(request, 'predict.html', {'airlines': airlines, 'airports': airports})
