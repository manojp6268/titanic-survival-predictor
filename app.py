import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model
with open("models/titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


# Define a route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    pclass     = int(request.form["pclass"])
    sex        = int(request.form["sex"])          # 0=male, 1=female
    age        = float(request.form["age"])
    fare       = float(request.form["fare"])
    family_size = int(request.form["family_size"])
    is_child   = 1 if age < 12 else 0
    has_cabin  = int(request.form["has_cabin"])    # 0 or 1
    title      = request.form["title"]

    # Log transform fare — same as notebook
    fare_log = np.log1p(fare)

    # Title one-hot encoding
    # Training created: Title_Miss, Title_Mr, Title_Mrs, Title_Rare
    # drop_first=True dropped the first category alphabetically
    # Check your feature_names to confirm which was dropped

    title_miss  = 1 if title == "Miss"   else 0
    title_mr    = 1 if title == "Mr"     else 0
    title_mrs   = 1 if title == "Mrs"    else 0
    title_rare  = 1 if title == "Rare"   else 0

    # Build raw feature dictionary
    raw_features = {
        "Pclass"     : pclass,
        "Sex"        : sex,
        "Age"        : age,
        "FamilySize" : family_size,
        "IsChild"    : is_child,
        "Fare_log"   : fare_log,
        "HasCabin"   : has_cabin,
        "Title_Miss" : title_miss,
        "Title_Mr"   : title_mr,
        "Title_Mrs"  : title_mrs,
    }

    input_df = pd.DataFrame([raw_features])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale features
    scale_cols = ["Age", "Fare_log", "FamilySize"]
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]*100  # Probability of survival in percentage

    # Prepare result
    result = {
        "prediction": "Survived" if prediction == 1 else "Did not survive",
        "probability": f"{probability:.2f}%",
        "survived" : bool(prediction),  # True if survived, False otherwise
        "pclass"   : pclass,
        "sex"      : "Female" if    sex == 1 else "Male",
        "age"      : age,
        "title"    : title,
    }

    return render_template("results.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)