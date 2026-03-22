import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import statistics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn import preprocessing



# Loading data
data = pd.read_csv(r'C:\Users\Omar\Downloads\results22.csv')
datax = data.iloc[:,:3]
print(datax)

datax = preprocessing.StandardScaler().fit_transform(datax)

print(datax)

# Create feature and target arrays
X = datax
print("feature arrays:")
print(X)
y = data.iloc[:,-1]
print("Target array:") 
print(y.head) 


    
#Split into training and test set

X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size = 0.2, random_state=42)


print("Training data:" )
print(X_train.shape) 
print("Test data:" )
print(X_test.shape)

# training the model on training set

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print("Predicted SF for test data:")
pred = gnb.predict(X_test) 
print("Predicted SF for test data:",pred) 


# Calculate the accuracy of the model

print("Gaussian Naive Bayes model accuracy(in %):",gnb.score(X_test,y_test )*100)


ten = pd.read_csv(r'C:\Users\Omar\Downloads\50k50.csv')
# Predict on dataset which model has not seen before
print("Predicted SF for test data 10:")



X_testt=ten.iloc[:,:3]
X_testt=preprocessing.StandardScaler().fit_transform(X_testt)
y_testt=ten.iloc[:,-1]

print("NEW X test")
print(X_testt) 
print("NEW y test")
print(y_testt)

pred = gnb.predict(X_testt) 
print("Predicted SF for test data:",pred)

# Calculate the accuracy of the model
print("Gaussian Naive Bayes model accuracy(in %):",gnb.score(X_testt,y_testt )*100)

df = pd.read_csv(r'C:\Users\Omar\Downloads\50000.csv')

df["SFF"]=pred
df.to_csv(r'C:\Users\Omar\Downloads\SF-GNB50000-norm.csv', index=False)

  




