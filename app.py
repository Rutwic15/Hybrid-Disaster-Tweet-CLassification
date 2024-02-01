from flask import Flask, render_template, request
import pickle
import tensorflow as tf
from keras.models import load_model
import numpy as np
import nltk

import string
from sklearn.feature_extraction.text import TfidfVectorizer

model = load_model('conv1d.h5')

app = Flask(__name__)



@app.route('/')
def man():
    # return "hello"
    return render_template('home.html')
def preprocess(tweet):
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    nltk_stopwords+=string.punctuation
    stopwords=[]
    for x in nltk_stopwords:
        if x not in ['#','@']:
            stopwords.append(x)
    def message_cleaning(message):
        Test_punc_removed= [char for char in message if char not in string.punctuation]
        Test_punc_removed_join=''.join(Test_punc_removed)
        Test_punc_removed_join_clean=[word for word in Test_punc_removed_join.split() if word.lower() not in stopwords ]
        return Test_punc_removed_join_clean
    tweet=message_cleaning(tweet)
    vectorizer = TfidfVectorizer()
    tweet = vectorizer.fit_transform(tweet)
    tweet = tweet.toarray()
    tweet = np.pad(tweet, (0, 5637-len(tweet)),mode='constant')
    tweet=tf.expand_dims(tweet,axis=-1)
    return tweet

@app.route('/predict', methods=['POST'])
def home():
    tweet = request.form['a']
    tweet=preprocess(tweet)
    # print(tweet)
    pred = model.predict(tweet)
    return render_template('after.html', data=pred[0][0])


if __name__ == "__main__":
    app.run(debug=True)















