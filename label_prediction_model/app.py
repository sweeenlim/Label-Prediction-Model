import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
scaler= pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('cover.html') #return this html file to the client aka my browser

@app.route('/predict',methods=['POST'])
def predict():
    X_test = vectorizer.transform([x for x in request.form.values()]) #returns a generator 
    X_test= scaler.transform(X_test.toarray())
    prediction = model.predict(X_test)
    result = dict(result=prediction[0])
    return jsonify(result)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(port=8000,debug=True)