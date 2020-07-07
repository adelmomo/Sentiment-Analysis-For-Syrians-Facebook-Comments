from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
#read the dataset
corpus=pd.read_csv('../ProjectFiles/Sentiment.csv')
classes=4 #number of classes (positive,negative,...etc)
vocab=[]
Y=[]
max_len=-1
for sample in corpus.values:
    item=(sample[0],sample[1])
    text=item[1]
    label=item[0]
    text=nltk.word_tokenize(text)
    max_len=max(max_len,len(text))
    for word in text:
        vocab.append(word)
    if label=='__label__Negative':
        Y.append([0])
    elif label=='__label__Positive':
        Y.append([1])
    elif label=='__label__Sarcasm':
        Y.append([2])
    else:
        Y.append([3])
#tokenizing the sentences
vocab=set(vocab)
vocab_length=len(vocab)+1
oov_tok='<OOV>'
tokenizer=Tokenizer(num_words=vocab_length,oov_token=oov_tok)
tokenizer.fit_on_texts([sent[1] for sent in corpus.values])
word_index=tokenizer.word_index
X=tokenizer.texts_to_sequences([sent[1] for sent in corpus.values])
#embedded_sentences = [tf.keras.preprocessing.text.one_hot(sent[1], vocab_length) for sent in corpus.values]
BATCH_SIZE=16
EPOCHS=50
#X is the sentences
X=tf.keras.preprocessing.sequence.pad_sequences(np.asarray(X),maxlen=max_len,padding='post')
#Y is the label for X (positive,negative...etc)
Y=tf.keras.utils.to_categorical(Y,classes)
#split the data into 70% training data and 30% test data
trainx,testx,trainy,testy=train_test_split(X,Y,test_size=.3,random_state=200)
output_dim=32
#define the model
def get_model():
    input_layer=tf.keras.Input((None,))
    embedding=tf.keras.layers.Embedding(vocab_length,output_dim)(input_layer)
    features=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,return_sequences=True))(embedding)
    features=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100))(features)
    y_hat=tf.keras.layers.Dense(1024)(features)
    y_hat=tf.nn.relu(y_hat)
    y_hat=tf.keras.layers.Dense(classes)(y_hat)
    y_hat=tf.nn.softmax(y_hat)
    model=tf.keras.Model(input_layer,y_hat)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    return model
model=get_model()
checkpoint = tf.keras.callbacks.ModelCheckpoint('model', save_best_only=True, verbose=2, monitor='val_accuracy',save_weights_only=True)
#training the model
model.fit(trainx,trainy,validation_data=(testx,testy),epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=2,callbacks=[checkpoint])

