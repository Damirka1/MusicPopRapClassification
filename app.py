import numpy as np
import pandas as pd

df_train = pd.read_csv('train.csv', sep = ',')

print(df_train.head(10))

target_map = {1:'pop', 0:'rap'}
df_train['genre'] = df_train['class'].map(target_map)

df_train['lyric'] = df_train['lyric'].str.replace(',','')

train_text = df_train.lyric
train_label = df_train['class']


from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Embedding
from keras.layers import LSTM
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

token = Tokenizer(num_words=4000)
token.fit_on_texts(train_text)
x_train_seq = token.texts_to_sequences(train_text)
x_train = pad_sequences(x_train_seq, maxlen = 400)
x_train = np.array(x_train)
y_train = np.array(train_label).reshape(-1,1)

model = Sequential()
model.add(Embedding(output_dim=32,input_dim = 4000,input_length=400))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))

print(model.summary())

model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])

train_history = model.fit(x_train,y_train,batch_size=400,epochs=10,verbose=2,
                         validation_split=0.2)


token = Tokenizer(num_words=4000)
text = ['As he came into the window', 'All my people from the front to the back nod, back nod']

token.fit_on_texts(text)
x_train_seq = token.texts_to_sequences(text)
x_train = pad_sequences(x_train_seq,maxlen = 400)
x_train = np.array(x_train)

res = model.predict(x_train)

print(res)

for i in range(len(res)):
    if(res[i] > 0.5):
        print("Pop", text[i], sep=" : ")
    else:
        print("Rap", text[i], sep=" : ")