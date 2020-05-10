from flask import Flask, request, jsonify, render_template, url_for
import sys
import load_nn_v2 as nn

app = Flask(__name__)

@app.route('/predict',methods=['GET', 'POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    f = request.files['file']
    filename = f.filename
    
    # Let assume right now that the file name is given
    #filename = 'sample.wav'
    predictions = nn.predict(filename)
    return jsonify(predictions)
    #return render_template('home.html',chat_in=predictions)
    
@app.route('/prediction',methods=['GET', 'POST'])
def predict():
    '''
    For Testing
    '''
    
    f = request.files['file']
    filename = f.filename
    
    predictions = nn.predict(filename)
    return render_template('home.html',chat_in=predictions)

if __name__ == '__main__':
    app.run(debug=True)