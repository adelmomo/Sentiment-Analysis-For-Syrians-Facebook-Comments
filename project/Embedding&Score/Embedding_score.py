import pickle
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
data=pd.read_csv('../ProjectFiles/Sentiment.csv')
Y=[]
max_width=-1
classes=4
stopwords=stopwords.words('arabic')
pickle_in = open("../ProjectFiles/Dictionary.pkl","rb")
dictionary = pickle.load(pickle_in)
word_index=1
vocab=[]
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

for sample in data.values:
    words = nltk.word_tokenize(sample[1])
    max_width=max(max_width,len(words))
    if sample[0]=='__label__Negative':
        Y.append([0]) 
    elif sample[0]=='__label__Positive':
        Y.append([1])
    elif sample[0]=='__label__Sarcasm':
        Y.append([2])
    else:
        Y.append([3])

Y=np.array(Y)
X1=np.empty((0,max_width))
for sample in data.values:
    words = nltk.word_tokenize(sample[1])
    for word in words:
        vocab.append(word)
    x=get_score(words)
    x=process(x)
    X1=np.append(X1,x.reshape(-1,max_width),axis=0)

#define hyperparameters
vocab=set(vocab)
oov_tok='<OOV>'
vocab_len=len(vocab)+1
tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=vocab_len,oov_token=oov_tok)
tokenizer.fit_on_texts([sent[1] for sent in data.values])
dic=tokenizer.word_index
dic_file=open('../ProjectFiles/tokenizer.pkl','wb')
pickle.dump(tokenizer,dic_file)
X2=tokenizer.texts_to_sequences([sent[1] for sent in data.values])
X2=tf.keras.preprocessing.sequence.pad_sequences(np.array(X2),maxlen=max_width,padding='post')
Y=tf.keras.utils.to_categorical(Y,classes)
rnd=np.random.permutation(len(data.values))
X1=X1[rnd]
X2=X2[rnd]
Y=Y[rnd]
train_split=0.3
BATCH_SIZE=32
#load first input data
testx1=X1[:int(len(data.values)*train_split)]
trainx1=X1[int(len(data.values)*train_split):]

#load second input data
testx2=X2[:int(len(data.values)*train_split)]
trainx2=X2[int(len(data.values)*train_split):]

#load target data
trainy=Y[int(len(data.values)*train_split):]
testy=Y[:int(len(data.values)*train_split)]
#define model
output_dim=64
class Network(tf.keras.Model):
    def __init__(self):
        super(Network,self).__init__()
        self.embedding=tf.keras.layers.Embedding(vocab_len,output_dim)
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

EPOCHS=50
opt=tf.keras.optimizers.Adam()
model=Network()
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
checkpoint = tf.keras.callbacks.ModelCheckpoint('model', save_best_only=True, verbose=2, monitor='val_accuracy',save_weights_only=True)
model.fit([trainx1,trainx2],trainy,validation_data=([testx1,testx2],testy),epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=2,callbacks=[checkpoint])











    
