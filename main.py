from flask import Flask, request, jsonify, render_template, url_for
import sys
import load_nn_v2 as nn
from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route('/predict',methods=['GET', 'POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    f = request.files['file']
    filename = f.filename
    filePath = "/tmp/" + secure_filename(filename)
    f.save(filePath)      
    
    predictions = nn.predict(filename)
    return jsonify(predictions)
    
@app.route('/prediction',methods=['GET', 'POST'])
def predict():
    '''
    For Testing
    '''
    f = request.files['file']
    filename = f.filename
    filePath = "/tmp/" + secure_filename(filename)
    f.save(filePath)    
    
    predictions = nn.predict(filename)
    return render_template('home.html',chat_in=predictions)

if __name__ == '__main__':
    app.run(debug=True)