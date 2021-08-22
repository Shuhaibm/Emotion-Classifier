import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords

import re
import numpy as np


def prepare_data(data):
    stop_words = set(stopwords.words('english'))
    tokenized_data = []
    mapping = {}
    for text in data["Description"]:

        text = re.sub(r'\W+', ' ', text).lower()
        tokenized_text = word_tokenize(text)

        text=[]
        for token in tokenized_text:
            if token not in stop_words:
                text.append(token)
                mapping[token] = len(mapping)-1

        text=" ".join(text)
        tokenized_data.append(text)
        
    
    return tokenized_data



def one_hot_encoding_version1(tokenized_data,mapping):    
    
    one_hot_encoding = np.zeros((len(tokenized_data),len(mapping)))
    for i,sentence in enumerate(tokenized_data):
        for word in sentence:
            one_hot_encoding[i,mapping[word]] = 1
    

    return one_hot_encoding

#def one_hot_encoding_version2(tokenized_data):
#    one_hot_encoding = [one_hot(input_text=sentence, n=1000) for sentence in tokenized_data]
#    final_data = pad_sequences(sequences=one_hot_encoding,
#                              maxlen=150,
#                              padding="pre")
#    return final_data
