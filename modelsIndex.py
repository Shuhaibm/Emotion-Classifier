import pandas
import numpy as np

from preprocess import prepare_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from myModels import my_naive_bayes,my_random_forest,my_log_reg,my_lin_reg,my_return_the_mode, my_ensemble,my_stacking,my_stacking_concatenated

#Get data
train_data = pandas.read_csv("data/train.txt", sep=";", names=["Description","Emotion"])
test_data = pandas.read_csv("data/test.txt", sep=";", names=["Description","Emotion"])
meta_data = pandas.read_csv("data/val.txt", sep=";", names=["Description","Emotion"])

allData = [train_data,test_data]
df = pandas.concat(allData)


#Preprocess text
emotion_to_label = {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4,'surprise': 5}
cv = CountVectorizer()

X = prepare_data(df)
X = cv.fit_transform(X)
y = df['Emotion'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

X_meta = prepare_data(meta_data)
X_meta = cv.transform(X_meta)
y_meta = meta_data['Emotion'].values



#Base Models

#Naive Bayes Model
nb = my_naive_bayes()
nb.fit(X_train,y_train)
print("Naive Bayes Model, Accuracy Score: " + str(nb.model.score(X_test,y_test)))


#Random Forest
rf = my_random_forest()
rf.fit(X_train,y_train)
print("Random Forest Model, Accuracy Score: " + str(rf.model.score(X_test,y_test)))


#Logistic Regression
lr = my_log_reg()
lr.fit(X_train,y_train)
print("Logistic Regression Model, Accuracy Score: " + str(lr.model.score(X_test,y_test)))




#Meta Models

base_models = [nb,rf,lr]


#Stacking
stacking_model = my_stacking(base_models)
stacking_model.train_and_score(X_meta, y_meta)


#Stacking where X_meta is concatenated with base_predictions
stacking_model_concat = my_stacking_concatenated(base_models)
stacking_model_concat.train_and_score(X_meta, y_meta)


#Return the Mode
base_models = [nb,rf,lr]
    
rtm = my_return_the_mode(base_models)
rtm.train_and_score(X_test, y_test)


#Ensemble - taking avg of probabilities

ens = my_ensemble(base_models)
ens.train_and_score(X_meta,y_meta)

