from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and polynomial transformer
with open("polynomial_model.pkl", "rb") as f:
    poly_model = pickle.load(f)

with open("poly_transformer.pkl", "rb") as f:
    poly = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if not data or "value" not in data:
        return jsonify({"error": "Missing 'value' in request body"}), 400

    values = data["value"]

    try:
        # Make sure input is a list of numbers
        if not isinstance(values, list) or not all(isinstance(x, (int, float)) for x in values):
            return jsonify({"error": "'value' must be a list of numbers"}), 400

        # Convert to 2D array
        values_array = np.array(values).reshape(1, -1)

        # Apply polynomial transformation
        values_transformed = poly.transform(values_array)

        # Predict
        prediction = poly_model.predict(values_transformed)

        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
