from flask import Flask, jsonify, render_template
from flask import request
from flask_cors import CORS, cross_origin

from threading import Thread

import pandas

from preprocess import prepare_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from myModels import my_naive_bayes,my_random_forest,my_log_reg,my_stacking_concatenated



#Get data
train_data = pandas.read_csv("data/train.txt", sep=";", names=["Description","Emotion"])
test_data = pandas.read_csv("data/test.txt", sep=";", names=["Description","Emotion"])
meta_data = pandas.read_csv("data/val.txt", sep=";", names=["Description","Emotion"])

allData = [train_data,test_data]
df = pandas.concat(allData)


#Preprocess text
cv = CountVectorizer()

X = cv.fit_transform(prepare_data(df))
y = df['Emotion'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

print("Start Training")

#Base Models

#Naive Bayes Model
nb = my_naive_bayes()
nb.fit(X_train,y_train)
#Random Forest
rf = my_random_forest()
rf.fit(X_train,y_train)
#Logistic Regression
lr = my_log_reg()
lr.fit(X_train,y_train)

base_models = [nb,rf,lr]


#Meta Model
#Stacking where X_meta is concatenated with base_predictions: 90.55% Accuracy
stacking_model_concat = my_stacking_concatenated(base_models)
stacking_model_concat.train_and_score(X_test, y_test)



print("End Training")

app = Flask(
  __name__,
	template_folder='templates',
  static_url_path="/static"
)
CORS(app, support_credentials=True)

@app.route("/")
@cross_origin(supports_credentials=True)
def home():
  return render_template(
		'index.html')

@app.route("/api")
@cross_origin(supports_credentials=True)
def find_emotion():
    word = request.args.get('string')
    return jsonify({
      "emotion":stacking_model_concat.predictInput(word, cv)[0]
      })

def run():
  app.run(host='0.0.0.0',port=7000)

t = Thread(target = run)
t.start()
