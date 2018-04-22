# author - yjzhou
# Mar 26 2017
import numpy as np
import pandas as pd
import re

from bs4 import BeautifulSoup

import sys
import os
import csv

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model


MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = string.decode('utf-8')
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

data_train = pd.read_csv('Sentiment Analysis Dataset.csv',sep='\t')
print (data_train.shape)

texts = []
labels = []


for idx in range(data_train.SentimentText.shape[0]):
    text = BeautifulSoup(data_train.SentimentText[idx], "lxml")
    texts.append(clean_str(text.get_text().encode('ascii','ignore')))
    labels.append(data_train.Sentiment[idx])
#print(texts)

#从语料库中获取句子和标签


embeddings_index = {}
f = open(os.path.abspath('glove.6B.100d.txt'),encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', len(texts))
print('Shape of label tensor:', len(labels))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#测试集数据读取 & 转成词向量
data_test = pd.read_csv('SA-test.csv')
print(data_test.shape)
test_texts=[]
for idx in range(data_test.SentimentText.shape[0]):
    test_text = BeautifulSoup(data_test.SentimentText[idx], "lxml")
    test_texts.append(clean_str(test_text.get_text().encode('ascii','ignore')))
    
test_tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
test_tokenizer.fit_on_texts(test_texts)
test_sequences = test_tokenizer.texts_to_sequences(test_texts)
test_word_index = test_tokenizer.word_index
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(test_data)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print ('Length of embedding_matrix:', embedding_matrix.shape[0])
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            mask_zero=False,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
#词向量可训练

print('Traing and validation set number of positive and negative reviews')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))



sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_gru = Bidirectional(LSTM(100, return_sequences=False))(embedded_sequences)
dropout_1=Dropout(0.5)(l_gru)
dense_1 = Dense(100,activation='tanh')(dropout_1)
dropout_2=Dropout(0.5)(dense_1)
dense_2 = Dense(2, activation='softmax')(dropout_2)

model = Model(sequence_input, dense_2)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=32)


csvfile = open('output.csv', 'w') 
writer = csv.writer(csvfile)
writer.writerow(['statement','label'])

a=np.array(model.predict(test_data, batch_size=32, verbose=1))
output=[]
for i in range(len(test_texts)):
    output_a=[test_texts[i],np.argmax(a,axis=1)[i]]
    output.append(output_a)
    print(test_texts[i])
    print(np.argmax(a, axis=1)[i])
print(output)
model.save('Twitter-LSTM-Model.h5')

 
writer.writerows(output)
csvfile.close()

