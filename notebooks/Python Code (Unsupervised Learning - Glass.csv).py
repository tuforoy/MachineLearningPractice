# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:49:20 2023

@author: lauka
"""
#%% import package

from sklearn.metrics import silhouette_score
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as shc
from sklearn.mixture import GaussianMixture 
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

#%% import and visualize data

data = pd.read_csv('C:/data/Glass.csv')
data = data.iloc[:,1:]
print(data.iloc[:,0:5].describe())
print(data.iloc[:,5:9].describe())
scaler = StandardScaler()
plt.boxplot(data.iloc[:,:-1])
plt.show()
plt.boxplot(data)
plt.show()
pd.plotting.scatter_matrix(data.iloc[:,:-1])

#%% Silhouette Plot and Elbow Method (k-means)

K = range(2,10); silhouette_avg = []
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data.iloc[:,:-1])
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(data, cluster_labels));
plt.figure();
plt.plot(K,silhouette_avg,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette score For Optimal k')

K = range(1,10); Sum_of_squared_distances = []
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data.iloc[:,:-1])
    Sum_of_squared_distances.append(kmeans.inertia_)

plt.figure();
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Sum of squared distances/Inertia')
plt.title('Elbow Method For Optimal k')

#%%  AIC BIC for GMM Optimal K

np.random.seed(4640)

K = range(1,10); AIC = []; BIC=[]
for a in K :
    gmm = GaussianMixture(n_components=a).fit(data.iloc[:,:-1]) 
    AIC.append(gmm.aic(data.iloc[:,:-1]))
    BIC.append(gmm.bic(data.iloc[:,:-1]))
plt.figure();
a=plt.plot(K,AIC,'rx-',label='AIC')
b=plt.plot(K,BIC,'bx-',label='BIC')
plt.legend(loc=1)
plt.xlabel('Values of K')
plt.ylabel('Sum of squared distances/Inertia')
plt.title('AIC and BIC for Optimal K (GMM)')

#%% Hierarchical (Dendrogram)

plt.figure()
dend= shc.dendrogram(shc.linkage (data.iloc[:,:-1],method='single'))
plt.title('Single link')

plt.figure()
dend= shc.dendrogram(shc.linkage (data.iloc[:,:-1],method='complete'))
plt.title('Complete link')

plt.figure()
dend= shc.dendrogram(shc.linkage (data.iloc[:,:-1],method='average'))
plt.title('Average link')

#%% kmeans (k=4)

kmeans4 = KMeans(n_clusters=4).fit(data.iloc[:,:-1]) 
centroids = kmeans4.cluster_centers_ 
labels = kmeans4.labels_ 
label_encoder = LabelEncoder()
print("Confusion Matrix with K-means (k=4)","\n",
      confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels))
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print('Score=',acc)

plt.figure();
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'.'); 
plt.plot(data.iloc[labels==2,0],data.iloc[labels==2,1],'k.');  
plt.plot(data.iloc[labels==3,0],data.iloc[labels==3,1],'y.'); 
plt.xlabel("Ri")
plt.ylabel("Na")
plt.title("Sactter plot of Ri and Na with 4 Clusters by K-means Clustering")
plt.plot(centroids[:,0],centroids[:,1],'ro') 


#%% GMM (k=5)

gmm = GaussianMixture(n_components=5,covariance_type="tied").fit(data.iloc[:,:-1]) 
centroids = gmm.means_
labels = gmm.predict(data.iloc[:,:-1])
print("Confusion Matrix with GMM (5 Clusters)","\n",
      confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels))
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print('Score=',acc)

plt.figure(); 
plt.plot(data.iloc[labels==0,2],data.iloc[labels==0,3],'g.');  
plt.plot(data.iloc[labels==1,2],data.iloc[labels==1,3],'.');  
plt.plot(data.iloc[labels==2,2],data.iloc[labels==2,3],'k.');  
plt.plot(data.iloc[labels==3,2],data.iloc[labels==3,3],'y.');
plt.plot(data.iloc[labels==4,2],data.iloc[labels==4,3],'c.');
plt.xlabel("Mg")
plt.ylabel("Al")
plt.title("Sactter plot of Mg and Al with 5 Clusters by GMM")
plt.plot(centroids[:,2],centroids[:,3],'ro') 

plt.show()

#%% GMM (k=9)

gmm = GaussianMixture(n_components=9).fit(data.iloc[:,:-1]) 
centroids = gmm.means_
labels = gmm.predict(data.iloc[:,:-1])
print("Confusion Matrix with GMM (9 Clusters)","\n",
      confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels))
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print('Score=',acc)

plt.figure(); 
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g.');  
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'g+'); 
plt.plot(data.iloc[labels==2,0],data.iloc[labels==2,1],'k.');  
plt.plot(data.iloc[labels==3,0],data.iloc[labels==3,1],'y.'); 
plt.plot(data.iloc[labels==4,0],data.iloc[labels==4,1],'c.'); 
plt.plot(data.iloc[labels==5,0],data.iloc[labels==5,1],'m.');
plt.plot(data.iloc[labels==6,0],data.iloc[labels==6,1],'cx');
plt.plot(data.iloc[labels==7,0],data.iloc[labels==7,1],'b.');
plt.plot(data.iloc[labels==8,0],data.iloc[labels==8,1],'b+');
plt.xlabel("Ri")
plt.ylabel("Na")
plt.title("Sactter plot of Ri and Na with 9 Clusters by GMM")
plt.plot(centroids[:,0],centroids[:,1],'ro') 
plt.show()

#%% Hierarchical - Single (k=2)

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single').fit(data.iloc[:,:-1])
labels = cluster.labels_
label_encoder = LabelEncoder()
print("Confusion Matrix with Hierarchical (Single) (2 Clusters)","\n",
      confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels))
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print('Score=',acc)

plt.figure(); 
plt.plot(data.iloc[labels==0,2],data.iloc[labels==0,3],'y.');  
plt.plot(data.iloc[labels==1,2],data.iloc[labels==1,3],'k.');  
plt.xlabel("Mg")
plt.ylabel("Al")
plt.title("Sactter plot of Mg and Al with Hierarchical (Single) (2 Clusters)")

#%% Hierarchical - Complete (k=4)

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete').fit(data.iloc[:,:-1]) 
labels = cluster.labels_
label_encoder = LabelEncoder()
print("Confusion Matrix with Hierarchical (complete) (4 Clusters)","\n",
      confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels))
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print('Score=',acc)

plt.figure(); 
plt.plot(data.iloc[labels==0,3],data.iloc[labels==0,5],'y.');  
plt.plot(data.iloc[labels==1,3],data.iloc[labels==1,5],'m.');  
plt.plot(data.iloc[labels==2,3],data.iloc[labels==2,5],'k.');  
plt.plot(data.iloc[labels==3,3],data.iloc[labels==3,5],'g.'); 
plt.xlabel("Al")
plt.ylabel("k")
plt.title("Sactter plot of Al and k with Hierarchical (complete) (4 Clusters)")

#%% Hierarchical - Average (k=4)

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average').fit(data.iloc[:,:-1]) 
labels = cluster.labels_
label_encoder = LabelEncoder()
print("Confusion Matrix with Hierarchical (average) (4 Clusters)","\n",
      confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels))
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print('Score=',acc)

plt.figure(); 
plt.plot(data.iloc[labels==0,0],data.iloc[labels==0,1],'g+');  
plt.plot(data.iloc[labels==1,0],data.iloc[labels==1,1],'r.'); 
plt.plot(data.iloc[labels==2,0],data.iloc[labels==2,1],'m+');  
plt.plot(data.iloc[labels==3,0],data.iloc[labels==3,1],'k.'); 

plt.xlabel("Ri")
plt.ylabel("Na")
plt.title("Sactter plot of ri and Na with Hierarchical (average) (4 Clusters)")


#%% 3d scatter plot

import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data.iloc[:,0], data.iloc[:,1], data.iloc[:,2], c=data.iloc[:,0]);

ax.set_xlabel('ri')
ax.set_ylabel('na')
ax.set_zlabel('mg');

#%% 3d scatter plot (kmeans k=4)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data.iloc[labels==0,0], data.iloc[labels==0,1], data.iloc[labels==0,2],color='blue',marker='+');
ax.scatter3D(data.iloc[labels==1,0], data.iloc[labels==1,1], data.iloc[labels==1,2],color='grey',marker='+');
ax.scatter3D(data.iloc[labels==2,0], data.iloc[labels==2,1], data.iloc[labels==2,2],color='purple',marker='+');
ax.scatter3D(data.iloc[labels==3,0], data.iloc[labels==3,1], data.iloc[labels==3,2],color='green',marker='+');
ax.scatter3D(centroids[:,0],centroids[:,1],centroids[:,2],color='red',marker='o',s=30) 

ax.set_xlabel('ri')
ax.set_ylabel('na')
ax.set_zlabel('mg');
#%%  ensanble 1 (kmeans(4)+hir_com(4))

np.random.seed(4640)
no_samples = data.shape[0]
no_estimators = 50
vote = 1/no_estimators
co_association = np.zeros((no_samples, no_samples))

for est in range(no_estimators):
    kmeans = KMeans(n_clusters=4,init='random',n_init=1).fit(data.iloc[:,:-1])
    labels = kmeans.labels_
    res = [[int(i == j) for i in labels] for j in labels]
    res = np.array(res)
    res = res * vote
    co_association = co_association + res

distance_matrix = 1-co_association
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean',linkage='complete',compute_distances=True).fit(distance_matrix) 
labels = cluster.labels_
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(results)
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)

acc=0

#%%  ensanble 2 (kmeans(4)+hir_average(4))

np.random.seed(4640)
no_samples = data.shape[0]
no_estimators = 50
vote = 1/no_estimators
co_association = np.zeros((no_samples, no_samples))

for est in range(no_estimators):
    kmeans = KMeans(n_clusters=4,init='random',n_init=1).fit(data.iloc[:,:-1])
    labels = kmeans.labels_
    res = [[int(i == j) for i in labels] for j in labels]
    res = np.array(res)
    res = res * vote
    co_association = co_association + res

distance_matrix = 1-co_association
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean',linkage='average',compute_distances=True).fit(distance_matrix) 
labels = cluster.labels_
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(results)
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)

#%%  ensanble 3 (gmm(9) +hir_average(4))

np.random.seed(4640)
no_samples = data.shape[0]
no_estimators = 50
vote = 1/no_estimators
co_association = np.zeros((no_samples, no_samples))

for est in range(no_estimators):
    gmm = GaussianMixture(n_components=9).fit(data.iloc[:,:-1])
    labels = gmm.predict(data.iloc[:,:-1])
    res = [[int(i == j) for i in labels] for j in labels]
    res = np.array(res)
    res = res * vote
    co_association = co_association + res

distance_matrix = 1-co_association
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean',linkage='average',compute_distances=True).fit(distance_matrix) 
labels = cluster.labels_
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(results)
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)





#%% ensemble 4 (k-means(4) + gmm(9) + hir_complete(4))


np.random.seed(4640)
no_samples = data.shape[0]
no_estimators = 50
vote = 1/no_estimators
co_association = np.zeros((no_samples, no_samples))

for est in range(no_estimators):
    kmeans = KMeans(n_clusters=4,init='random',n_init=1).fit(data.iloc[:,:-1])
    labels = kmeans.labels_
    gmm = GaussianMixture(n_components=5).fit(data.iloc[:,:-1])
    glabels = gmm.predict(data.iloc[:,:-1])
    res = [[int(i == j) for i in labels] for j in labels]
    res = np.array(res)
    gres = [[int(i == j) for i in glabels] for j in glabels]
    gres = np.array(gres)
    res = res * vote
    gres = gres * vote
    co_association = co_association + res + gres
distance_matrix = 2-co_association
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean',linkage='complete',compute_distances=True).fit(distance_matrix) 
labels = cluster.labels_
label_encoder = LabelEncoder()
results = confusion_matrix(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(results)
acc =  metrics.completeness_score(label_encoder.fit_transform(data.iloc[:,-1]),labels)
print(acc)




