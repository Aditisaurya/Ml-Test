from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained RandomForest model
MODEL_PATH = "stress_model.pkl"
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Ensure the loaded model has a predict method
if not hasattr(model, "predict"):
    raise ValueError("Loaded object is not a trained RandomForestClassifier!")

# Serve the HTML file
@app.route("/")
def home():
    return render_template("index.html")  # Load index.html from the templates folder

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = data.get("features")

        if not features or len(features) != 8:
            return jsonify({"error": "Invalid input. Please provide exactly 8 features."}), 400

        features_array = np.array(features).reshape(1, -1)

        prediction = model.predict(features_array)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
