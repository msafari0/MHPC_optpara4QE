from flask import Flask, request, jsonify, render_template
import os
import sys
sys.path.append("./src")
import pandas as pd
import ann_model as ann
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
#CORS(app)
CORS(app, resources={r"/*": {"origins": "https://maxpredict.cineca.it"}})

#model = joblib.load('model.pkl')  # Load weights if necessary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
    
        # Get model name from URL query parameter
        model_name = request.args.get('model', 'Leonardo_Booster')

        # Build the model path
        model_path = f"models/model_{model_name}.keras"
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model '{model_name}' not found"}), 400

        # Load the selected model dynamically
        model = ann.TimePerCall.load(model_path)

        # Load and convert input
        data = request.get_json()
        test_input = pd.DataFrame([data])
        
        print("Received Data:\n", test_input)  # Debugging
         
        numeric_input = test_input.select_dtypes(include=[np.number])

        prediction_normed = model.predict_normed(numeric_input)
       
        output = float(prediction_normed[0][0])
        return jsonify({
            "prediction_time": round(output, 10)
        })
        #return jsonify({"prediction": round(output, 10)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)

