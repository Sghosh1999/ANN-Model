# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 02:02:56 2019

@author: Sayantan Ghosh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values


Y = dataset.iloc[:,13].values

# Encoding the data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features =[1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]  #Avoiding the dummy variable trap

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the Input Layer and first Hidden Layer
classifier.add(Dense(units = 6,kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
  
#Adding the 2nd Hidden Layer
classifier.add(Dense(units = 6,kernel_initializer = 'uniform', activation = 'relu'))

#Adding the output Layer
classifier.add(Dense(units = 1,kernel_initializer = 'uniform', activation = 'sigmoid'))


#Compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training Set(Stochastic Gradient descent)
classifier.fit(X_train,Y_train,batch_size = 10,epochs = 100)

#Making the prediction and evaluating the model

Y_pred = classifier.predict(X_test)
Y_pred =(Y_pred>0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
print("Accuracy:", accuracy*100)



