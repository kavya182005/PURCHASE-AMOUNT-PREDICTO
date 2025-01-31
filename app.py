from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained ML model, scaler, and label encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load dataset to extract column names
file_path = "shopping_trends.csv"
df = pd.read_csv(file_path)

# Extract feature columns (excluding the target variable)
feature_columns = [col for col in df.columns if col != "Purchase Amount (USD)"]

@app.route("/")
def home():
    return render_template("index.html", feature_columns=feature_columns)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract user input from form
        form_values = dict(request.form)

        # Convert categorical inputs using label encoders (if applicable)
        for col in label_encoders:
            if col in form_values:
                if form_values[col] in label_encoders[col].classes_:
                    form_values[col] = label_encoders[col].transform([form_values[col]])[0]
                else:
                    # If unseen label, assign a default category (e.g., first known category)
                    form_values[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]

        # Convert form data into a numerical array
        features = [float(form_values[col]) for col in feature_columns]
        input_array = np.array(features).reshape(1, -1)

        # Apply feature scaling
        input_array = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_array)

        return render_template("index.html", feature_columns=feature_columns, prediction_text=f'Predicted Purchase Amount: ${prediction[0]:.2f}')

    except Exception as e:
        return render_template("index.html", feature_columns=feature_columns, prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
