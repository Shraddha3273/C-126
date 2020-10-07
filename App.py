from flask import Flask, jsonify, request
from Recognize_Data import get_prediction

app = Flask(__name__)

@app.route("/add-data", methods=["POST"])

def predict_data() : 
    image = request.files.get("digit")
    prediction = get_prediction(image)
    return jsonify({
        "Prediction" : prediction
    }), 200
    
if(__name__ == "__main__") : 
    app.run(debug=True)