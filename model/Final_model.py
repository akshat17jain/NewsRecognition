import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from sklearn.ensemble import RandomForestClassifier

loaded_vector = joblib.load('tfidf.pkl')
filename = 'finalized_model.sav'
loaded_RFC = joblib.load(filename)



def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "True News"
 

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = loaded_vector.transform(new_x_test)
    pred_RFC = loaded_RFC.predict(new_xv_test)

    return print("\n\nRFC Prediction: {}".format(output_lable(pred_RFC[0])))

news = str('')
manual_testing(news)