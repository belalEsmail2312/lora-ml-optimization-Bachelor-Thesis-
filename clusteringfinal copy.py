# %%
from pyexpat import model

import clf as clf
import matplotlib
import numpy
import numpy as np
from numpy import sqrt
import random
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error
matplotlib.use('TkAgg')
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.cluster as cluster
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

def k_clustering_method(kmeans, df_80):
    y = kmeans.fit_predict(df_80[['Packets Sent', 'SF','TP','TotalEnergyConsumed']])
    centroids = kmeans.cluster_centers_
    df_80['Cluster'] = y
    df_80['centroid_Packets Sent'] = df_80['Cluster'].map(lambda x: centroids[x][0])
    df_80['centroid_TP'] = df_80['Cluster'].map(lambda x: centroids[x][1])
    df_80['centroid_SF'] = df_80['Cluster'].map(lambda x: centroids[x][2])
    df_80['centroid_TotalEnergyConsumed'] = df_80['Cluster'].map(lambda x: centroids[x][3])
    df_80['Packets Sent_dist'] = (df_80['Packets Sent'] - df_80['centroid_Packets Sent']) ** 2
    df_80['TP_dist'] = (df_80['TP'] - df_80['centroid_TP']) ** 2
    df_80['SF_dist'] = (df_80['SF'] - df_80['centroid_SF']) ** 2
    df_80['TotalEnergyConsumed_dist'] = (df_80['TotalEnergyConsumed'] - df_80['centroid_TotalEnergyConsumed']) ** 2
    df_80['total_distance'] = sqrt(
    df_80['Packets Sent_dist'] + df_80['TP_dist'] + df_80['SF_dist'] + df_80['TotalEnergyConsumed_dist'])
    df_80['total_distance_avg'] = numpy.average(df_80['total_distance'])
    df_80 = df_80.reset_index(level=0)
    df_80.rename(columns={df_80.columns[0]: "ID"}, inplace=True)
    df_80 = df_80.sort_values(by=['ID'])
    return df_80

def remove_columns(df_20_6):
    df_20_6 = df_20_6.drop('total_distance_avg', 1)
    df_20_6 = df_20_6.drop('total_distance', 1)
    df_20_6 = df_20_6.drop('TotalEnergyConsumed_dist', 1)
    df_20_6 = df_20_6.drop('SF_dist', 1)
    df_20_6 = df_20_6.drop('TP_dist', 1)
    df_20_6 = df_20_6.drop('centroid_TotalEnergyConsumed', 1)
    df_20_6 = df_20_6.drop('centroid_SF', 1)
    df_20_6 = df_20_6.drop('centroid_TP', 1)
    df_20_6 = df_20_6.drop('centroid_Packets Sent', 1)
    df_20_6 = df_20_6.drop('Packets Sent_dist', 1)
    return df_20_6




X = pd.read_csv("final.csv")

df = pd.DataFrame(X, columns=['Packets Sent', 'SF','TP','TotalEnergyConsumed','Module','X','Y'])
result=[]
x1= df['Packets Sent']
y1= df['SF']
z1=df['TP']
w1=df['TotalEnergyConsumed']
a1=df['Module']
b1=df['X']
c1=df['Y']
df['Module'] = X['Module'].values
df['X']=X['X'].values
df['Y']=X['Y'].values
x1_train,x1_test,y1_train,y1_test,z1_train,z1_test,w1_train,w1_test,a1_train,a1_test,b1_train,b1_test,c1_train,c1_test=train_test_split(x1,y1,z1,w1,a1,b1,c1,test_size=0.2)

#print(type(x1_test))
df_80=pd.DataFrame(columns=['Packets Sent', 'SF', 'TP','TotalEnergyConsumed','Module','X','Y'])
df_80['Module']=a1_train
df_80['X']=b1_train
df_80['Y']=c1_train
test_array1 = x1_train
normalized_x1_test = (test_array1 - min(test_array1)) / (max(test_array1) - min(test_array1))
#print(normalized_test_array1)
df_80['Packets Sent'] = normalized_x1_test

test_array2 = y1_train
#normalized_y1_test = (test_array2 - min(test_array2)) / (max(test_array2) - min(test_array2))
#print(normalized_test_array2)
df_80['SF'] = test_array2

test_array3 = z1_train
normalized_z1_test = (test_array3 - min(test_array3)) / (max(test_array3) - min(test_array3))
#print(normalized_test_array3)
df_80['TP'] = normalized_z1_test

test_array4 = w1_train
normalized_w1_test = (test_array4 - min(test_array4)) / (max(test_array4) - min(test_array4))
#print(normalized_test_array2)
df_80['TotalEnergyConsumed'] = normalized_w1_test


print(df_80)

kmeans_6 = KMeans(n_clusters=6)
for i in range(1,7):
    kmeans = KMeans(n_clusters=i)

    y = kmeans.fit_predict(df_80[['Packets Sent', 'SF', 'TP','TotalEnergyConsumed']])
    centroids = kmeans.cluster_centers_
    df_80['Cluster'] = y
    df_80['centroid_Packets Sent'] = df_80['Cluster'].map(lambda x: centroids[x][0])
    df_80['centroid_TP'] = df_80['Cluster'].map(lambda x: centroids[x][1])
    df_80['centroid_SF'] = df_80['Cluster'].map(lambda x: centroids[x][2])
    df_80['centroid_TotalEnergyConsumed'] = df_80['Cluster'].map(lambda x: centroids[x][3])
    df_80['Packets Sent_dist'] = (df_80['Packets Sent'] - df_80['centroid_Packets Sent']) ** 2
    df_80['TP_dist'] = (df_80['TP'] - df_80['centroid_TP'])**2
    df_80['SF_dist'] = (df_80['SF'] - df_80['centroid_SF']) ** 2
    df_80['TotalEnergyConsumed_dist'] = (df_80['TotalEnergyConsumed'] - df_80['centroid_TotalEnergyConsumed']) ** 2
    df_80['total_distance'] = sqrt(df_80['Packets Sent_dist']+df_80['TP_dist']+df_80['SF_dist']+df_80['TotalEnergyConsumed_dist'])
    df_80['total_distance_avg']=numpy.average(df_80['total_distance'])
    result.append(numpy.average(df_80['total_distance']))
    #print(df_80.head())
    #print(centroids)
print(result)
min_cluster=np.argmin(result)+1
print('Number Of Clusters =',min_cluster)





kmeans_6 = KMeans(n_clusters=6)
kmeans = KMeans(n_clusters=min_cluster)

res_min_cluster = k_clustering_method(kmeans, df_80)
res_6 = k_clustering_method(kmeans_6,df_80)

true_values = res_6['total_distance_avg']
predictions = res_min_cluster['total_distance_avg']

N = true_values.shape[0]
accuracy = (true_values == predictions).sum() / N

print('accuracy1',accuracy)

res_6=remove_columns(res_6)
res_6.to_csv('data 80.csv')
#print('res_min_cluster',res_min_cluster)



#print('res_6',res_6)
#accuracy1=mean_squared_error(res_6['total_distance_avg'],res_min_cluster['total_distance_avg'])




#result.append(numpy.average(df_80['total_distance']))
#print(df_80.head())
#print(centroids)

#acc=accuracy_score(min_cluster,kmeans)
#print('accuracy is=',acc)



# %%
df_20=pd.DataFrame(columns=['Packets Sent', 'SF', 'TP','TotalEnergyConsumed','Module','X','Y'])
df_20['Packets Sent'] = x1_test
df_20['SF'] = y1_test
df_20['TP'] = z1_test
df_20['TotalEnergyConsumed'] = w1_test
df_20['Module']=a1_test
df_20['X']=b1_test
df_20['Y']=c1_test
#df_2020=df_20
#df_2020.to_csv('df_2020.csv')
#### NORMALIZATION

test_array1 = x1_test
normalized_x1_test = (test_array1 - min(test_array1)) / (max(test_array1) - min(test_array1))
#print(normalized_test_array1)
df_20['Packets Sent'] = normalized_x1_test

test_array2 = y1_test
#normalized_y1_test = (test_array2 - min(test_array2)) / (max(test_array2) - min(test_array2))
#print(normalized_test_array2)
df_20['SF'] = test_array2

test_array3 = z1_test
normalized_z1_test = (test_array3 - min(test_array3)) / (max(test_array3) - min(test_array3))
#print(normalized_test_array3)
df_20['TP'] = normalized_z1_test

test_array4 = w1_test
normalized_w1_test = (test_array4 - min(test_array4)) / (max(test_array4) - min(test_array4))
#print(normalized_test_array3)
df_20['TotalEnergyConsumed'] = normalized_w1_test



########## ASSSIGN FOR TEST BASED ON CENTROIDS

#print(df_20.head())
#print(df_20.shape)

df_20_min_cluster = k_clustering_method(kmeans, df_20)
df_20_6 = k_clustering_method(kmeans_6, df_20)
df_20_6=remove_columns(df_20_6)
df_20_6.to_csv('data 20.csv')
print(df_20_6)
#res = kmeans.predict(df_20)
#df_20['Cluster'] = res + 1
#print(df_20.head(10))


# Merging 80% with 20%
frames = [res_6,df_20_6]
all_cluster_6 =pd.concat(frames)
all_cluster_6 = all_cluster_6.sort_values(by=['ID'])

frames = [res_min_cluster, df_20_min_cluster]
all_min_cluster =pd.concat(frames)
all_min_cluster= all_min_cluster.sort_values(by=['ID'])
#all_cluster_6.to_csv('all_cluster_6.csv')


X['Clustering_6']=all_cluster_6['Cluster'].values
X.to_csv('data 100.csv')


plt.plot(list(range(1,7)),result)
plt.xlabel('Number of clusters')
plt.ylabel('Total distance average')
plt.title('K-Means')
plt.show()

data_reduced=df_80[['Packets Sent', 'SF','TP','TotalEnergyConsumed','Cluster']]
color = ['black', 'green', 'blue', 'yellow', 'orange','red','pink','cyan','magenta','white','lime']
#for k in range(min_cluster):
for k in range(6):
    data = data_reduced[data_reduced['Cluster'] == k + 1]
    plt.scatter(data['Packets Sent'], data['SF'],data['TP'],c=color[k])
plt.show()


a = df_20['SF']

# Creating histogram
fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(a, bins=np.linspace(0, 1.0, num=6))
plt.xlabel('Normalized SF')
plt.ylabel('Number Of Users')

# Show plot
plt.show()



#print('y1 test',y1_test)




