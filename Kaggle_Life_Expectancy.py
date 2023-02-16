#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# The file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly in "names"
df = pd.read_csv(
    "C:\\Users\\prasa\\Downloads\\Kaggle Files\\Kaggle CSV Files\\Life_expectancy_dataset.csv", encoding='ISO-8859-1' )
# IPython.display allows nice output formatting within the Jupyter notebook
display(df.head())


# In[49]:


#RUN ONLY ONCE:

X = df
y = df.pop('Continent')


# In[50]:


print(X.shape)
print(y.shape)

print(y.value_counts())


# In[51]:


for i in range(len(y)):
    if y[i] == 'Europe':
        y[i] = 0
    elif y[i] =='Asia':
        y[i] = 1
    elif y[i] =='North America':
        y[i] = 2
    elif y[i] =='Africa':
        y[i] = 3
    elif y[i] =='Oceania':
        y[i] = 4
    elif y[i] =='South America':
        y[i] = 5
    else:
        pass


# In[93]:


# Poping out the country column as it is a string and can later be linked
# with its respective Ranks. 
X = df.drop(['Country','Rank','Overall Life'],axis=1)
y = y.astype(int)


# In[94]:


print(X)


# #### Using the Linear models for multiclass classification:

# In[95]:


print(X.columns)


# In[96]:


#X = X.head(10)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.20, random_state=42)  


# In[97]:


from sklearn.preprocessing import StandardScaler

#Using the standard scaler to scale the features for preprocessing:
#When using the standard scaler you do not need to manually scale the model.
scaler = StandardScaler()
scale = scaler.fit(X_train)
X_train = scale.transform(X_train)
X_test = scale.transform(X_test)


# In[98]:


#Logistic Regression model:
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[99]:


from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)
score


# In[100]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)


# In[101]:


y_train.unique()


# In[102]:


# Getting dataframe labels

from sklearn.utils.multiclass import unique_labels

print(unique_labels(y_test))

#This shows the y_train.unique(), but in accending order.


# In[103]:


#Now, showing the confusion matrix with its proper labels in table format:

def plot(y_true, y_pred):
    labels = unique_labels(y_test)
    columns = [f'Predicted {label}' for label in labels]
    index = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_true,y_pred), columns=columns, index=index)

    return table


# In[104]:


plot(y_test, y_pred)


# In[105]:


#Now for the heat map for the confusion matrix:

import seaborn as sns

def plot_heat(y_true, y_pred):
    labels = unique_labels(y_test)
    columns = [f'Predicted {label}' for label in labels]
    index = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_true,y_pred), columns=columns, index=index)

    return sns.heatmap(table, annot=True, fmt='d', cmap='viridis')


# In[106]:


plot_heat(y_test, y_pred)


# #### Analyzing Confusion matrix

# In[107]:


#Using confusion matrix to calculate the precision, recall, accuracy score and f1-score.
#Calculate TP, TN, FP, FN.

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[ ]:





# 
