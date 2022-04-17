#!/usr/bin/env python
# coding: utf-8

# # KNeighbour Classifier - Dataset - FRUITS.text

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics


# In[2]:


df=pd.read_table("fruits.txt")


# In[3]:


df.head()


# In[4]:


X=df[["mass","width","height","color_score"]]
y=df["fruit_label"]


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=0,test_size=0.25)


# In[7]:


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()


# In[8]:


Xtrain=sc_x.fit_transform(Xtrain)
Xtest=sc_x.transform(Xtest)


# In[9]:


import math
math.ceil(math.sqrt(len(Xtrain)))


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[11]:


knn.fit(Xtrain,ytrain)
ypred=knn.predict(Xtest)


# In[12]:


metrics.confusion_matrix(ytest,ypred)


# In[13]:


error = []
accuracy = []
# Calculating error for K values between 1 and 40
for i in range(1,40,2):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(Xtrain, ytrain)
 pred_i = knn.predict(Xtest)
 error.append(np.mean(pred_i != ytest))
 accuracy.append(metrics.accuracy_score(ytest, pred_i))


# In[14]:


plt.figure(figsize=(12, 6))
plt.plot(range(1,40,2), error, color='red', linestyle='dashed', marker='o',
 markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[15]:


knn=KNeighborsClassifier(n_neighbors=7)


# In[16]:


knn.fit(Xtrain,ytrain)
ypred=knn.predict(Xtest)
#metrics.confusion_matrix(ytest,ypred)
metrics.accuracy_score(ytest,ypred)


# In[ ]:




