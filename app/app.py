from flask import Flask, render_template, request

app = Flask(__name__)
@app.route('/',methods=['GET'])
def webpage():
    return render_template("index.html")

@app.route('/',methods=['POST'])
def predict():
    fileInput = request.files['fileInput']
    return render_template('index.html')

app.run(debug=True)