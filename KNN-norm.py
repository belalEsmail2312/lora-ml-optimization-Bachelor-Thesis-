import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import numpy as np
import statistics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

test_accuracy = [99.9,94,92.8,92.17,90.9,90.1,90.5,91.2,94.3,96,97.2,98.8,99.3,99.7,99.5,99.1,98.8,98.4,96.9,95]
n=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    
plt.plot(n,test_accuracy, label = 'Testing dataset Accuracy')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

print( "14 neighbors has the highest accuracy")
print("----------------------------------------------------")

# training the model on training set
knn = KNeighborsClassifier(n_neighbors=14)
knn.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print("Predicted SF for test data:")

pred=knn.predict(X_test) 
print(knn.predict(X_test))

# Calculate the accuracy of the model 
accuracypercent=(knn.score(X_test, y_test))*100
print("percentage of accuracy:",accuracypercent,"%")


ten = pd.read_csv(r'C:\Users\Omar\Downloads\results10000.csv')
# Predict on dataset which model has not seen before
print("Predicted SF for test data 10:")



X_testt=ten.iloc[:,:3]
X_testt=preprocessing.StandardScaler().fit_transform(X_testt)
y_testt=ten.iloc[:,-1]

print("NEW X test")
print(X_testt)
print("NEW y test")
print(y_testt)

pred=knn.predict(X_testt) 
print(knn.predict(X_test))

# Calculate the accuracy of the model
accuracypercent=(knn.score(X_testt, y_testt))*100
print("percentage of accuracy:",accuracypercent,"%")

df = pd.read_csv(r'C:\Users\Omar\Downloads\results10k.csv')

df["SFF"]=pred
#df.to_csv(r'C:\Users\Omar\Downloads\SF-K-NN50000-norm.csv', index=False)
def SF_assignment(args):
    SF7=[]
    SF8=[]
    SF9=[]
    SF10=[]
    SF11=[]
    SF12=[]
    for i in range(len(args)):
        if args[i]==7:
            SF7.append(args[i])
        elif args[i]==8:
            SF8.append(args[i])
        elif args[i]==9: 
            SF9.append(args[i])
        elif args[i]==10:
            SF10.append(args[i])
        elif args[i]==11:
            SF11.append(args[i])
        elif args[i]==12:
            SF12.append(args[i])
    return SF7,SF8,SF9,SF10,SF11,SF12
              
SF7,SF8,SF9,SF10,SF11,SF12 = SF_assignment(pred)

h=[len(SF7),len(SF8),len(SF9),len(SF10),len(SF11),len(SF12)]
print(len(SF12))
bin_edges=[7,8,9,10,11,12]
plt.bar(bin_edges,h)
plt.xlabel("SF")
plt.ylabel("Number of nodes")
plt.title("SF assignment")
plt.show() 




dff = pd.read_csv(r'C:\Users\Omar\Downloads\copyy.csv')

x=dff["SFF"]
y=dff["SF"]

same=[]
diff=[]

for i in range(len(x)):
    if x[i]==y[i]:
        same.append(1)
    else:
        diff.append(1)
        
print("same")
print(len(same)) 
print("diff") 
print(len(diff))    
  
    
        



