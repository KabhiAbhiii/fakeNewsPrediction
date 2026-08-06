import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import string
import re

df=pd.read_csv('dataset/news.csv')
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

df=df.fillna("")
x=df["text"]
y=df["label"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer(stop_words="english",max_features=10000,ngram_range=(1,2))
xv_train=vect.fit_transform(x_train)
xv_test=vect.transform(x_test)

model=RandomForestClassifier(n_estimators=100,max_depth=40,min_samples_split=5,min_samples_leaf=2,random_state=42,n_jobs=-1)
model.fit(xv_train,y_train)
accuracy=model.score(xv_test,y_test)
print(f"Model accuracy: {accuracy}")

joblib.dump(model,'model/random_forest.pkl')
joblib.dump(vect,"model/vectorizer.pkl")
print("model saved as random_forest.pkl")