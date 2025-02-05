from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained RandomForest model
MODEL_PATH = "stress_model.pkl"  # Replace with the correct path to your model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Ensure the loaded model has a predict method
if not hasattr(model, "predict"):
    raise ValueError("Loaded object is not a trained RandomForestClassifier!")

@app.route("/")
def home():
    return render_template("index.html")  # Render index.html

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the Content-Type is application/json
        if request.content_type != 'application/json':
            return jsonify({"error": "Request must be in JSON format"}), 415

        # Get the JSON data from the request
        data = request.get_json()  # Flask's built-in method to parse JSON data

        # Extract the features from the JSON payload
        features = data.get("features")

        if not features or len(features) != 8:
            return jsonify({"error": "Invalid input. Please provide exactly 8 features."}), 400

        # Convert features to a numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
