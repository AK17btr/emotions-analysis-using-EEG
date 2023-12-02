#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.io import loadmat
import os
from pywt import wavedec
from scipy import signal
from scipy.stats import entropy
from scipy.fft import fft, ifft
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as K
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_validate
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional, BatchNormalization, PReLU, ReLU, Reshape
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt;
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from tensorflow import keras
from tensorflow.keras.layers import Conv1D,Conv2D,Add
from tensorflow.keras.layers import MaxPool1D, MaxPooling2D
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("C:\\Users\\sendm\\Downloads\\archive (2)\\emotions.csv")
data.head()


# In[3]:


sample = data.loc[0, 'fft_0_b':'fft_749_b']


# In[4]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))
plt.plot(range(len(sample)),sample)
plt.title("Features fft_0_b through fft_749_b",weight='bold',color='black')
plt.show()


# In[5]:


fft_data = data.loc[:,'fft_0_b':'fft_749_b']
fft_data


# In[6]:


data.columns.unique()


# In[7]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['label']=le.fit_transform(data['label'])


# In[8]:


y = data.pop('label')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,random_state=48)
X_train = np.array(X_train).reshape((X_train.shape[0],X_train.shape[1],1))
X_test = np.array(X_test).reshape((X_test.shape[0],X_test.shape[1],1))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# In[9]:


from keras.layers import Flatten
from keras.layers import Dense
inputs = tf.keras.Input(shape=(X_train.shape[1],1))
gru = tf.keras.layers.GRU(512, return_sequences=True)(inputs)
flat = Flatten()(gru)
outputs = Dense(3, activation='softmax')(flat)
model = tf.keras.Model(inputs, outputs)
model.summary()


# In[10]:


import pydot
graph = pydot.Dot()


# In[11]:


def train_model(model, x_train, y_train, x_test, y_test, save_to, epoch = 2):
     
        opt_adam = keras.optimizers.Adam(learning_rate=0.001)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        mc = ModelCheckpoint(save_to + '_best_model.h5', monitor='val_accuracy', mode ='max', verbose=1, save_best_only=True)

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))
        model.compile(optimizer=opt_adam, loss=['categorical_crossentropy'],metrics=['accuracy'])
 
        history = model.fit(x_train, y_train, batch_size=32, epochs = epoch, validation_data = (x_test, y_test), callbacks = [es, mc, lr_schedule])
    
        saved_model = load_model(save_to + '_best_model.h5') 

        return model, history


# In[12]:


model,history = train_model(model, X_train, y_train, X_test, y_test, save_to= './', epoch = 3) 


# In[13]:


plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy']) 

plt.title('model accuracy') #
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[14]:


model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))


# In[15]:


y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(X_test))))
y_test = y_test.idxmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[16]:


print("Classification Report:\n----------------------\n", clr)


# In[ ]:




