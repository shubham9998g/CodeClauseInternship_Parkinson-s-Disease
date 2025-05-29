from flask import Flask, render_template, request
import numpy as np
import pickle
import os
app = Flask(__name__)
model=pickle.load(open('parkinson_model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('body.html')

@app.route('/submit', methods=['POST'])
def submit():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction=model.predict(features)

    return render_template('body.html', prediction_text='Positive' if prediction[0] == 1 else 'Negative')
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
