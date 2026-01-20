from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'model', 'wine_cultivar_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'model', 'wine_scaler.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get 6 inputs from form
        feature_values = [float(x) for x in request.form.values()]
        final_features = scaler.transform([np.array(feature_values)])
        prediction = model.predict(final_features)
        
        # Wine dataset has 3 classes: 0, 1, 2
        cultivars = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
        result = cultivars.get(prediction[0], "Unknown")
        
        return render_template('index.html', prediction_text=f'Predicted Origin: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)