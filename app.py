from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)

# Load the model

model = joblib.load("random_forest_best_model.pkl")
# with open("random_forest_best_model.pkl", "rb") as f:
#     model = pickle.load(f)

@app.route("/")
def form():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:

        features = [
            data["price"],
            data["downpmt"],
            data["monthdue"],
            data["payment_left"],
            data["monthly_payment"],
            data["pmttype"],
            data["credit_score"],
            data["age"],
            data["gender"]
        ]

        prediction = model.predict([features])

        result = "Accept" if prediction[0] == 0 else "Reject"
        return jsonify({"prediction": result})

    except KeyError as e:
        return jsonify({"error": f"Missing parameter {e.args[0]}"})

if __name__ == "__main__":
    app.run(debug=True)
