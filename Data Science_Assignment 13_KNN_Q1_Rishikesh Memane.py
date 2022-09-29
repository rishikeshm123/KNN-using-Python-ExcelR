#!/usr/bin/env python
# coding: utf-8

# # KNN

# Prepare a model for glass classification using KNN
# 
# Data Description: <br>
# RI : refractive index<br>
# Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)<br>
# Mg: Magnesium<br>
# AI: Aluminum<br>
# Si: Silicon<br>
# K:Potassium<br>
# Ca: Calcium<br>
# Ba: Barium<br>
# Fe: Iron<br>
# 
# Type: Type of glass: (class attribute)<br>
# 1 -- building_windows_float_processed<br>
#  2 --building_windows_non_float_processed<br>
#  3 --vehicle_windows_float_processed<br>
#  4 --vehicle_windows_non_float_processed (none in this database)<br>
#  5 --containers<br>
#  6 --tableware<br>
#  7 --headlamps<br>
# 
# 
# 
# 
# 
# 

# # EDA

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


glass = pd.read_csv('glass.csv')
glass.head()


# In[5]:


glass[glass.duplicated()]


# In[6]:


glass.drop_duplicates()


# In[7]:


glass.info()


# In[8]:


glass.isna().sum()


# In[9]:


glass.describe()


# # Visualization 

# In[10]:


sns.pairplot(glass, hue=  "Type")


# In[11]:


categ = ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]
for col in categ:
    plt.figure(figsize=(10,5))
    sns.boxplot(glass[col])
    plt.title(col)


# In[12]:


sns.countplot(glass["Type"])
plt.title("Type")
glass["Type"].value_counts()


# # PPScore

# In[13]:


import ppscore as pps
pps.matrix(glass)


# # Feature Engineering

# In[14]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
X = glass.drop(['Type'], axis = 1)
X.head()


# In[15]:


Y = glass["Type"]
Y


# In[16]:


Feature_model = LogisticRegression()
fit = RFE(Feature_model,3).fit(X, Y)


# In[17]:


# Feature Ranking:
fit.ranking_


# In[18]:


X.head()


# In[19]:


X_feat = X.drop(["RI","Al"], axis=1)
X_feat.head()


# In[20]:


X1 = X_feat
Y1 = Y


# # Building KNN Model

# ### 1. Grid search  for algorithm tuning

# In[37]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[33]:


n_neighbors = np.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)
model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X1, Y1)


# In[34]:


print(grid.best_score_)
print(grid.best_params_)


# ### 2. Model Building

# #### a. Evaluating using Cross Validation

# In[35]:


kfold = KFold(n_splits = 10)


# In[38]:


model = KNeighborsClassifier(n_neighbors = 1)
results = cross_val_score(model, X1, Y1, cv=kfold)


# In[39]:


print("Model Accuracy:-",round(results.mean(),3)*100,"%")


# #### b. Evaluating Using Leave one ut cross validation as dataset has comparatively less entries

# In[41]:


from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
loocv = LeaveOneOut()
model_knn = KNeighborsClassifier(n_neighbors = 1)
results_knn = cross_val_score(model_knn, X, Y, cv = loocv)
results_knn


# In[42]:


res1 = results_knn.mean()*100.0
res1


# In[43]:


print("Model Score by Leave One Out Cross Validation :- ",np.round(res1,3),"%")


# ### As expected LooCV prvides better results and accuracy.

# In[ ]:





# In[ ]:





# In[ ]:




