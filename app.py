from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
clf = joblib.load("classifier_model.pkl")
reg = joblib.load("regression_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


ge2024_df = pd.read_csv("GE_2024_Results new.csv") 

# Extract state list and mapping to constituencies
states = sorted(ge2024_df['State'].dropna().unique().tolist())
state_constituency_map = {
    state: sorted(ge2024_df[ge2024_df['State'] == state]['Constituency'].dropna().unique().tolist())
    for state in states
}

# All parties
all_parties = sorted(label_encoder.classes_.tolist())

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", states=states, parties=all_parties)

@app.route("/get_constituencies", methods=["POST"])
def get_constituencies():
    selected_state = request.json["state"]
    constituencies = state_constituency_map.get(selected_state, [])
    return jsonify(constituencies)

@app.route("/predict", methods=["POST"])
def predict():
    party_name = request.form["party"]
    vote_margin = float(request.form["vote_margin"])
    selected_state = request.form["state"]
    selected_constituency = request.form["constituency"]

    prediction = None
    vote_share = None

    if party_name not in all_parties:
        prediction = "Invalid Party"
    else:
        party_encoded = label_encoder.transform([party_name])[0]
        input_data = pd.DataFrame([[party_encoded, vote_margin]], columns=["Party_Label", "Vote_Margin"])

        win_pred = clf.predict(input_data)[0]
        vote_pred = reg.predict(input_data)[0]

        prediction = "WIN" if win_pred == 1 else "LOSE"
        vote_share = round(vote_pred, 2)

    return render_template(
        "index.html",
        parties=all_parties,
        states=states,
        selected_party=party_name,
        selected_state=selected_state,
        selected_constituency=selected_constituency,
        vote_margin=vote_margin,
        prediction=prediction,
        vote_share=vote_share
    )

if __name__ == "__main__":
    app.run(debug=True)
