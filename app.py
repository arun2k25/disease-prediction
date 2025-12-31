
import numpy as np
import pickle
from flask import Flask, request, jsonify

# Load models and symptoms
# Ensure these files are in the same directory as app.py or provide full paths
try:
    dt_model = pickle.load(open("dt_model.pkl", "rb"))
    rf_model = pickle.load(open("rf_model.pkl", "rb"))
    nb_model = pickle.load(open("nb_model.pkl", "rb"))
    symptoms = pickle.load(open("symptoms.pkl", "rb"))
except FileNotFoundError:
    print("Error: Model files not found. Please ensure 'dt_model.pkl', 'rf_model.pkl', 'nb_model.pkl', and 'symptoms.pkl' are in the correct directory.")
    # Exit or handle error appropriately for production
    exit()

# Define the predict_disease function (as refined in the previous step)
def predict_disease(selected_symptoms):
    input_vector = np.zeros(len(symptoms))

    for symptom in selected_symptoms:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            input_vector[index] = 1

    input_vector = input_vector.reshape(1, -1)

    pred_dt = dt_model.predict(input_vector)[0]
    pred_rf = rf_model.predict(input_vector)[0]
    pred_nb = nb_model.predict(input_vector)[0]

    # Consensus logic
    if pred_dt == pred_rf or pred_dt == pred_nb:
        final_prediction = pred_dt
    elif pred_rf == pred_nb:
        final_prediction = pred_rf
    else:
        final_prediction = pred_rf  # fallback

    return {
        'decision_tree_prediction': pred_dt,
        'random_forest_prediction': pred_rf,
        'naive_bayes_prediction': pred_nb,
        'final_prediction': final_prediction
    }

# Initialize the Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) # force=True to handle requests without Content-Type header

    if not data or 'symptoms' not in data:
        return jsonify({"error": "Invalid request format. 'symptoms' field is required."}), 400

    selected_symptoms = data['symptoms']

    if not isinstance(selected_symptoms, list):
        return jsonify({"error": "'symptoms' must be a list of strings."}), 400

    # Get predictions
    predictions = predict_disease(selected_symptoms)

    return jsonify(predictions)

# Standard Python entry point to run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
