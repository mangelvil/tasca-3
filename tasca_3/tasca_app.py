import pickle
from flask import Flask, jsonify, render_template, request

app = Flask('tasta3-predict')

# with open('models/regressio_logistica.pck', 'rb') as f:
#     dv, model = pickle.load(f)

# def predict_single(flor, dv, model):
#     x = dv.transform([flor])
#     y_pred = model.predict_proba(x)[:, 1]
#     return y_pred[0]


def predice_modelo(flor, modelo):
    if modelo == "regresion":
        file = "models/regressio_logistica.pck"
        
    elif modelo == "decision":
        file = "models/decision_tree.pck"
            
    elif modelo == "knn":
        file = "models/knn.pck"
                
    elif modelo == "scm":
        file = "models/scm.pck"
        
    else:
        return false
    
    with open(file, 'rb') as f: dv, model = pickle.load(f)
    
    x = dv.transform([flor])
    y_pred = model.predict_proba(x)[:, 1]
    return y_pred[0]
    
    

# Índex de la Web
@app.route('/')
def home():
    return render_template('index.html')

# Predicció Web
@app.route('/predict', methods=['POST'])
def predict():
    # Get input value from the form
    flor = {'petal amplada': float(request.form['amplada_value']), 'petal longitud': float(request.form['longitud_value'])}
    # flor = [float(request.form['amplada_value']),float(request.form['longitud_value'])]
    modelo = request.form['modelo']
        
    prediction = predice_modelo(flor ,modelo)

    return render_template('result.html', amplada_value=request.form['amplada_value'], longitud_value=request.form['longitud_value'], modelo=modelo, prediction=prediction)

# Directament amb CURL - Regression
@app.route('/regresion', methods=['POST'])
def regresion():
    
    prediction = predice_modelo(request.get_json() ,'regresion')
    
    print(type(request.get_json()))
    print(request.get_json())

    result = {
        'prediction': float(prediction)
    }

    return jsonify(result)

# Directament amb CURL - Decision Tree
@app.route('/decision', methods=['POST'])
def decision():
    
    prediction = predice_modelo(request.get_json() ,'decision')

    result = {
        'prediction': float(prediction)
    }

    return jsonify(result)

# Directament amb CURL - KNN
@app.route('/knn', methods=['POST'])
def knn():
    
    prediction = predice_modelo(request.get_json() ,'knn')

    result = {
        'prediction': float(prediction)
    }

    return jsonify(result)

# Directament amb CURL - SVM
@app.route('/svm', methods=['POST'])
def svm():
    
    prediction = predice_modelo(request.get_json() ,'svm')

    result = {
        'prediction': float(prediction)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)