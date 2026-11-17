import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from  sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU
from keras.layers import Dropout

result = pd.read_csv("Churn_Modelling.csv")
#print(result.columns)
#print(result.info())
#print(result.head())
#print(result.dtypes)

L = LabelEncoder()
result["Geography"] = L.fit_transform(result["Geography"])
result["Gender"] = L.fit_transform(result["Gender"])
print(result)
print(result.head())
print(result.info())
print(result.dtypes)
print(result.shape)
result = result.drop(["RowNumber","CustomerId","Surname"],axis = 1)
print(result)
print(result.shape)

y = result.pop("Exited")
x = result
print(x.shape)
print(y.shape)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2, random_state= 0)
s = StandardScaler()
xtrain = s.fit_transform(xtrain)
xtest = s.fit_transform(xtest)

classifier = Sequential()
classifier.add(Dense(units = 6,kernel_initializer = "he_uniform", activation = "relu", input_dim = 10))
classifier.add(Dense(units = 6,kernel_initializer = "he_uniform", activation = "relu"))
classifier.add(Dense(units = 1,kernel_initializer = "glorot_uniform", activation = "sigmoid"))

classifier.compile(optimizer = "Adamax", loss = "binary_crossentropy", metrics = ["accuracy"])
model_history = classifier.fit(xtrain,ytrain,batch_size = 10, epochs = 100)
classifier.summary()
yprediction = classifier.predict(xtest)
print(yprediction)