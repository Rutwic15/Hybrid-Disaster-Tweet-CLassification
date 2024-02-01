import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from scipy.sparse import csr_matrix
from tensorflow.keras.layers import Dropout
from keras.layers import Flatten
import keras
from keras.models import load_model
data=pd.read_csv(r'california_wildfires.csv')
nltk_stopwords = nltk.corpus.stopwords.words("english")
additional_words = ['“','”','...','``',"''",'’',"httpst",'amp',"theto","}",
                    "{","rt",".codgy3azxwOY","http:","https:","RT" ,"rt"]
nltk_stopwords+=string.punctuation
stopwords=[]
for x in nltk_stopwords:
    if x not in ['#','@']:
        stopwords.append(x)
def message_cleaning(message):
    Test_punc_removed=[char for char in message if char not in string.punctuation]
    Test_punc_removed_join=''.join(Test_punc_removed)
    Test_punc_removed_join_clean=[word for word in Test_punc_removed_join.split() if word.lower() not in stopwords ]
    return Test_punc_removed_join_clean
text=data['Tweet'].apply(message_cleaning)
data['new_Tweet']=text
data['new_Tweet']=data['new_Tweet'].apply(lambda x:[i.lower() for i in x])
# Get Labeled tweets vocabulary
# put all tokens of each tweet in one set
total_vocabulary = set(word for text in data.new_Tweet for word in text)

len(total_vocabulary)
print('There are {} unique tokens in the dataset. '. format (

len(total_vocabulary)) )

data['new_Tweet']=data['new_Tweet'].apply(lambda x:' '.join(x))

X = data['new_Tweet']
y = data['Label']

X = data['new_Tweet'].apply(str).tolist()
x_train,x_test,y_train,y_test=train_test_split(data['new_Tweet'],y,test_size=0.2,random_state=42)

vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

max_len = x_test_tfidf.shape[1]


# Convert the sparse matrix into a dense NumPy array
x_train_tfidf = x_train_tfidf.toarray()
x_test_tfidf=x_test_tfidf.toarray()


model=Sequential()
model.add(Embedding(len(total_vocabulary),100,input_length=max_len))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=2, padding='valid',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=2, padding='valid',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Fit the model to the training data
model.fit(tf.expand_dims(x_train_tfidf,axis=-1), y_train, epochs=5 ,batch_size=30,validation_data=(tf.expand_dims(x_test_tfidf,axis=-1),y_test))


model.save('conv1d.h5')

# from sklearn.metrics import classification_report
# y_pred=model.predict(x_test_tfidf)
# y_pred=[round(float(i)) for i in y_pred]
# print(classification_report(y_test,y_pred))