#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("C:/Users/arund/OneDrive/Documents/GRIP/data.csv")


# In[3]:


data.head()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores',style= 'o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[6]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[7]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# In[8]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[9]:


regressor.coef_


# In[10]:


regressor.intercept_


# In[11]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[12]:


y_pred = regressor.predict(X_test)


# In[13]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[14]:


pred= regressor.predict([[9.25]])
print("No of Hours = 9.25")
print("Predicted Score = {}".format(pred[0]))


# In[ ]:




