# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 07:55:26 2020

@author: Venkata Sainath
"""

from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
import matplotlib.pyplot as plt





X = np.load("Embeddings.npy")
#print(X)



distortions = [] 
inertias = [] 
mapping1 = {} 
mapping2 = {} 
K = range(1,25) 
  
for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k).fit(X) 
    kmeanModel.fit(X)     
      
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
  
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X.shape[0] 
    mapping2[k] = kmeanModel.inertia_ 


for key,val in mapping1.items(): 
    print(str(key)+' : '+str(val)) 
    
    
plt.plot(K, inertias, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('inertia') 
plt.title('The Elbow Method using inertia') 
plt.savefig("elbow_inertia.png")
plt.show() 


