import pickle
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
dt=pd.read_csv('./ProjectFiles/Sentiment.csv')
dt=dt.values
vocab=[]
max_width=-1
for sample in dt:
    txt=sample[1]
    txt=nltk.word_tokenize(txt)
    max_width=max(max_width,len(txt))
    for word in txt:
        vocab.append(word)
classes=4
out_dim=64
vocab_len=len(set(vocab))+1
pickle_in = open("./ProjectFiles/Dictionary.pkl","rb")
dictionary = pickle.load(pickle_in)
pickle_in = open("./ProjectFiles/tokenizer.pkl","rb")
tokenizer = pickle.load(pickle_in)
class Embedding_Score_Network(tf.keras.Model):
    def __init__(self,out_dim,classes,vocab_len):
        super(Embedding_Score_Network,self).__init__()
        self.embedding=tf.keras.layers.Embedding(vocab_len,out_dim)
        self.lstm1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,return_sequences=True))
        self.lstm2=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100))
        self.dense1=tf.keras.layers.Dense(512)
        self.dense2=tf.keras.layers.Dense(512)
        self.dense3=tf.keras.layers.Dense(classes)
    def call(self,X):
        x1=self.embedding(X[1])
        x1=self.lstm1(x1)
        x1=self.lstm2(x1)
        x=tf.keras.layers.Concatenate()([x1,X[0]])
        y_hat=self.dense2(x)
        y_hat=tf.nn.relu(y_hat)
        y_hat=self.dense3(y_hat)
        y_hat=tf.nn.softmax(y_hat)
        return y_hat
def get_Embedding_model():
    input_layer=tf.keras.Input((None,))
    embedding=tf.keras.layers.Embedding(vocab_len,out_dim)(input_layer)
    features=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,return_sequences=True))(embedding)
    features=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100))(features)
    y_hat=tf.keras.layers.Dense(1024)(features)
    y_hat=tf.nn.relu(y_hat)
    y_hat=tf.keras.layers.Dense(classes)(y_hat)
    y_hat=tf.nn.softmax(y_hat)
    model=tf.keras.Model(input_layer,y_hat)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    return model
embedding_score=Embedding_Score_Network(out_dim,classes,vocab_len)
embedding_score([np.zeros((1,max_width),dtype='float32'),np.zeros((1,vocab_len),dtype='int32')])
embedding_score.load_weights('Embedding&Score/model')
embedding=get_Embedding_model()
embedding.load_weights('Embedding/model')
def process(txt,threshold=0.1):
    text=np.array(txt)
    i=0
    while i < len(text):
        if(text[i]==0.0) and i>0 and i<len(text):
            flag=0
            for j in range(i+1,len(text)):
                if text[j]>0 and text[i-1]>0:
                    text[i:j]=threshold
                    i=j
                    flag=1
                    break
                elif text[j]>0 and text[i-1]<0:
                    text[i:j]=0.0
                    i=j
                    flag=1
                    break
                elif text[j]<0 and text[i-1]>0:
                    text[i:j]=0.0
                    i=j
                    flag=1
                    break
                elif text[j]<0 and text[i-1]<0:
                    text[i:j]=-threshold
                    i=j
                    flag=1
                    break
            if flag==0:
                if text[i-1]>0:
                    text[i:]=threshold
                elif text[i-1]<0:
                    text[i:]=-threshold
                i+=1
        elif i==0 and text[i]==0.0:
            flag=0
            for j in range(i + 1, len(text)):
                if text[j]>0:
                    text[i:j]=threshold
                    i=j
                    flag=1
                    break
                elif text[j]<0:
                    text[i:j]=-threshold
                    i=j
                    flag=1
                    break
            if flag==0:
                i+=1
        else:
            i+=1
    return np.pad(text, pad_width=[0, max_width - len(text)])

def get_score(text):
    score=[]
    for word in text:
        if dictionary.get(word) is None:
            score.append(0.0)
        else:
            score.append(dictionary.get(word))
    score=np.array(score)
    return score
def predict(text):
    txt=tokenizer.texts_to_sequences([text])
    txt=np.array(txt)
    words=nltk.word_tokenize(text)
    score=get_score(words)
    score=process(score)
    score=np.expand_dims(score,axis=0).astype('float32')
    res1=embedding_score([score,txt])
    res2=embedding(txt)
    return res1,res2
while True:
    print('Text: ')
    text=input()
    res1,res2=predict(text)
    print('Embedding&Score: ')
    res1=res1.numpy()
    print('Positive: ',res1[0][1])
    print('Negative: ',res1[0][0])
    print('Sarcasm: ',res1[0][2])
    print('Neutral: ',res1[0][3])
    if res1.argmax(axis=1)==0:
        print('Class: Negative')
    elif res1.argmax(axis=1)==1:
        print('Class: Positive')
    elif res1.argmax(axis=1)==2:
        print('Class: Sarcasm')
    else:
        print('Class: Neutral')

    print('Embedding: ')
    res2=res2.numpy()
    print('Positive: ',res2[0][1])
    print('Negative: ',res2[0][0])
    print('Sarcasm: ',res2[0][2])
    print('Neutral: ',res2[0][3])
    if res2.argmax(axis=1)==0:
        print('Class: Negative')
    elif res2.argmax(axis=1)==1:
        print('Class: Positive')
    elif res2.argmax(axis=1)==2:
        print('Class: Sarcasm')
    else:
        print('Class: Neutral')
