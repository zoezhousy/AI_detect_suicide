# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df = pd.read_csv('https://raw.githubusercontent.com/soumyajit4419/AI_For_Social_Good/master/Dataset/mergedData.csv?token=AK7VCIERPG353P22MNQU4KDAJIQRQ')

# %%
df.head()

# %%
text = df['text']
label = df['label']

# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

# %%
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# %%
sequence = tokenizer.texts_to_sequences(text)
padded_sequence = pad_sequences(sequence,padding='post')

# %%
x_train, x_test, y_train, y_test = train_test_split(padded_sequence,label,test_size=0.3,shuffle=True,random_state = 42)

# %%
# Used a api of globe for faster access
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove*.zip

# %%
embeddings_index = {};
with open('/content/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size,100));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;

# %%
model = Sequential()

model.add(Embedding(vocab_size,100,weights=[embeddings_matrix],trainable=False))
model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# %%
model.summary()

# %%
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# %%
history = model.fit(x_train,y_train,batch_size=32,epochs=10,validation_data=(x_test,y_test))

# %%
fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(18,5))
ax1.plot(history.history['accuracy'],label='train_accuracy')
ax1.plot(history.history['val_accuracy'],label='test_accuracy')
ax1.legend()
ax2.plot(history.history['loss'],label='train_loss')
ax2.plot(history.history['val_loss'],label='test_loss')
ax2.legend()
plt.show()


# %%
model.evaluate(x_test,y_test)

# %%
model.save('./lstm.h5')

# %%
pred = model.predict(x_test)

# %%
pred = pred>0.5

# %%
print(classification_report(pred,y_test))


