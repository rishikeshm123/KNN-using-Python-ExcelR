#!/usr/bin/env python
# coding: utf-8

# # KNN

# Implement a KNN model to classify the animals in to categories.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# # EDA

# In[2]:


zoo = pd.read_csv("Zoo.csv")
zoo


# In[3]:


zoo.info()


# In[4]:


zoo.isnull().sum()


# In[5]:


zoo[zoo.duplicated()] #No Duplicated Rows


# In[6]:


zoo.describe()


# In[7]:


sns.pairplot(zoo)


# ## ppscore

# In[8]:


import ppscore as pps
pps.matrix(zoo)


# In[9]:


zoo.head()


# In[10]:


categ = ["hair","feathers","eggs","milk","airborne","aquatic","predator","toothed","backbone","breathes","venomous","fins","legs","tail","domestic","catsize"]
for col in categ:
    plt.figure(figsize=(10,4))
    sns.countplot(zoo[col])
    plt.title(col)


# In[11]:


sns.countplot(zoo["type"])
plt.title("type")
zoo["type"].value_counts()


# In[12]:


zoo1 = zoo.copy()
zoo1.head()


# In[13]:


from sklearn.preprocessing import LabelEncoder
zoo1["animal name"] = LabelEncoder().fit_transform(zoo1["animal name"])
zoo1


# # FEATURE ENGINEERING:-
# ## METHOD:- Feature Extraction with Recursive Feature Elimination.(RFE)

# In[14]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
X = zoo1.drop(["type"],axis = 1)
X.head()


# In[15]:


Y = zoo["type"]
Y


# In[16]:


Feature_model = LogisticRegression()
fit = RFE(Feature_model,10).fit(X, Y)


# In[17]:


# Feature Ranking:
fit.ranking_


# In[18]:


X.head(3)


# # Selecting Only Top 7 Features.

# In[19]:


X_feat = X.drop(["animal name","domestic"], axis=1)
X_feat.head()


# In[20]:


X1 = X_feat
Y1 = Y


# # Building Model.
# ### 1) Grid Search for Algorithm Tuning.

# In[21]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[22]:


n_neighbors = np.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)
model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X1, Y1)


# ### Visualizing the CV results.

# In[23]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 41
k_range = range(1, 40)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X1, Y1, cv = 5)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[25]:


print(grid.best_score_)
print(grid.best_params_)


# #### As we can see by using {'n_neighbors': 1}, we get 0.96 accuracy
# #### so, we use it for model building in knn algorithm.

# ### 2) Model Building 

# In[26]:


kfold = KFold(n_splits = 5)


# In[27]:


model = KNeighborsClassifier(n_neighbors = 1)
results = cross_val_score(model, X1, Y1, cv=kfold)


# In[28]:


print("Model Accuracy:-",round(results.mean(),3)*100,"%")


# In[ ]:




