import pickle
from flask import Flask, jsonify, render_template, request

app = Flask('tasta3-predict')

# with open('models/regressio_logistica.pck', 'rb') as f:
#     dv, model = pickle.load(f)

# def predict_single(flor, dv, model):
#     x = dv.transform([flor])
#     y_pred = model.predict_proba(x)[:, 1]
#     return y_pred[0]

# Índex de la Web
@app.route('/')
def home():
    return render_template('index.html')

# Predicció Web
@app.route('/predict', methods=['POST'])
def predict():
    # Get input value from the form
    flor = [float(request.form['amplada_value']),float(request.form['longitud_value'])]
        
    #prediction = model.predict([flor])
    x = dv.transform([flor])
    y_pred = model.predict_proba(x)[:, 1]

    return render_template('result.html', amplada_value=request.form['amplada_value'], longitud_value=request.form['longitud_value'], prediction=prediction[0])

# Directament amb CURL - Regression
@app.route('/regresion', methods=['POST'])
def regresion():
    
    with open('models/regressio_logistica.pck', 'rb') as f:
        dv, model = pickle.load(f)

    flor = request.get_json()
    x = dv.transform([flor])
    prediction = model.predict_proba(x)[:, 1]

    result = {
        'prediction': float(prediction)
    }

    return jsonify(result)

# Directament amb CURL - Decision Tree
@app.route('/decision', methods=['POST'])
def decision():
    
    with open('models/decision_tree.pck', 'rb') as f:
        dv, model = pickle.load(f)

    flor = request.get_json()
    x = dv.transform([flor])
    prediction = model.predict_proba(x)[:, 1]

    result = {
        'prediction': float(prediction)
    }

    return jsonify(result)

# Directament amb CURL - KNN
@app.route('/knn', methods=['POST'])
def knn():
    
    with open('models/knn.pck', 'rb') as f:
        dv, model = pickle.load(f)

    flor = request.get_json()
    x = dv.transform([flor])
    prediction = model.predict_proba(x)[:, 1]

    result = {
        'prediction': float(prediction)
    }

    return jsonify(result)

# Directament amb CURL - SVM
@app.route('/svm', methods=['POST'])
def svm():
    
    with open('models/svm.pck', 'rb') as f:
        dv, model = pickle.load(f)

    flor = request.get_json()
    x = dv.transform([flor])
    prediction = model.predict_proba(x)[:, 1]

    result = {
        'prediction': float(prediction)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)