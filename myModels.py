from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np

class my_model:
    def __init__(self):
        pass
    
    def fit(self,X_train,y_train):
        self.model.fit(X_train,y_train)
      
    def predictInput(self,inputString, cv):
        sample_input = [inputString]
        vect = cv.transform(sample_input).toarray()
        return self.model.predict(vect)



class my_naive_bayes(my_model):
    def __init__(self):
        self.model = MultinomialNB()


class my_random_forest(my_model):
    def __init__(self):
        #grid = {'n_estimators':[10,20,40]}
        #random_forest = GridSearchCV(RandomForestClassifier(),grid) --> Dif btw 10 and 20 is 10%, dif btw 20 and 40 is 0.5%
        self.model = RandomForestClassifier(n_estimators=20)


class my_log_reg(my_model):
    def __init__(self):
        self.model = LogisticRegression(max_iter=170)
        

class my_lin_reg(my_model):
    def __init__(self):
        self.model = LinearRegression()
        
        
class my_return_the_mode:
    def __init__(self,base_models):
        self.base_models = base_models
    def train_and_score(self,X_test,y_test):
        base_predictions = []

        for y,example in enumerate(X_test):
            predictions=[]
            for x,myModel in enumerate(self.base_models):
                prediction = myModel.model.predict(example)
                predictions.append(prediction[0])
            
            mode = max(set(predictions),key=predictions.count)
                
            base_predictions.append(mode)
        
        total = 0
        correct = 0
        for i,prediction in enumerate(base_predictions):
            if prediction == y_test[i]:
                total +=1
                correct+=1
            else:
                total+=1
            
        print("Meta Model (Return the Mode), Accuracy Score: " + str(correct/total))
    def predictInput(self,inputString, cv):
        predictions=[]
        for x,myModel in enumerate(self.base_models):
            prediction = myModel.predictInput(inputString,cv)
            predictions.append(prediction[0])
            
        return max(set(predictions),key=predictions.count)
    
class my_stacking:
    def __init__(self,base_models):
        self.base_models = base_models
        self.model=my_random_forest()
    def train_and_score(self,X_meta,y_meta):
        emotion_to_label = {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4,'surprise': 5}
        
        base_predictions = np.zeros((len(y_meta),len(self.base_models)))
        
        for y,example in enumerate(X_meta):
            for x,myModel in enumerate(self.base_models):
                prediction = myModel.model.predict(example)
                base_predictions[y][x] = emotion_to_label[prediction[0]]
          
        self.model.fit(base_predictions, y_meta)
        print("Meta Model (Random Forest), Accuracy Score: " + str(self.model.model.score(base_predictions, y_meta)))

    def predictInput(self,inputString,cv):
        emotion_to_label = {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4,'surprise': 5}
        
        sample_input = [inputString]
        vect = np.array(cv.transform(sample_input).toarray())
        vect = []

        for x,myModel in enumerate(self.base_models):
            prediction = myModel.predictInput(inputString,cv)
            vect.append(emotion_to_label[prediction[0]])

        return self.model.model.predict(np.array(vect).reshape(1,-1))

 

class my_stacking_concatenated:
    def __init__(self,base_models):
        self.base_models = base_models
        self.model=my_random_forest()
    def train_and_score(self,X_meta,y_meta):
        emotion_to_label = {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4,'surprise': 5}
        
        base_predictions = np.array(X_meta)
        base_predictions = np.zeros((X_meta.shape[0], X_meta.shape[1]+3))
                
        for y,example in enumerate(X_meta):
            for x,myModel in enumerate(self.base_models):
                prediction = myModel.model.predict(example)
                base_predictions[y][X_meta.shape[1]+x] = emotion_to_label[prediction[0]]
        
        self.model.fit(base_predictions, y_meta)
        
        print("Meta Model with base_predictions concatenated with X_meta (Random Forest), Accuracy Score: " + str(self.model.model.score(base_predictions, y_meta)))

    def predictInput(self,inputString,cv):
        emotion_to_label = {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4,'surprise': 5}
        
        sample_input = [inputString]
        vect = np.array(cv.transform(sample_input).toarray())
        vect = np.zeros((1, vect.shape[1]+3))

        for x,myModel in enumerate(self.base_models):
            prediction = myModel.predictInput(inputString,cv)
            vect[0][vect.shape[1]-3+x] = emotion_to_label[prediction[0]]

        return self.model.model.predict(vect)





class my_ensemble: # - taking avg of probabilities
    def __init__(self,base_models):
        self.base_models = base_models
    def train_and_score(self,X_test,y_test):
        
        base_predictions = []
        
        classes = self.base_models[0].model.classes_
        
        for y,example in enumerate(X_test):
            probs = np.zeros((1,6))
            for x,myModel in enumerate(self.base_models):
                probs += myModel.model.predict_proba(example)
                    
            probs_avg = probs/len(self.base_models)   

            base_predictions.append(classes[np.argmax(probs_avg)])

      
        
        
        total = 0
        correct = 0
        for i,prediction in enumerate(base_predictions):
            if prediction == y_test[i]:
                total +=1
                correct+=1
            else:
                total+=1
            
        print("Meta Model (Ensemble), Accuracy Score: " + str(correct/total))

    def predictInput(self,inputString,cv):
        inputVal= cv.transform([inputString]).toarray()
        probs = np.zeros((1,6))
        for x,myModel in enumerate(self.base_models):
            probs += myModel.model.predict_proba(inputVal)

        classes = self.base_models[0].model.classes_
        probs_avg = probs/len(self.base_models)   

        return classes[np.argmax(probs_avg)]


    
    