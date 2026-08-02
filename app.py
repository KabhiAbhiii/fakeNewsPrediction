from flask import Flask,jsonify,request,render_template
import joblib
import string
import re

app=Flask(__name__)

model=joblib.load("model/random_forest.pkl")
vect=joblib.load("model/vectorizer.pkl")
@app.route('/')
def home():
    return render_template("index.html")

def wordproc(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]','',text)
    text = re.sub(r"\W"," ",text)
    text = re.sub(r'https?://\S+|www\.\S+','',text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub(r'\n','',text)
    text = re.sub(r'\w*\d\w*','',text)
    return text


@app.route('/predict', methods=['POST'])
def predict():
    text=request.form['text']
    text=wordproc(text)
    textv= vect.transform([text])
    prediction=model.predict(textv)
    if prediction[0]==1:
        prediction_text='The news is REAL'
    else:
       prediction_text='The news is FAKE'
    return render_template("index.html",prediction_text=prediction_text)  

if __name__=="__main__":
    app.run(debug=True)          