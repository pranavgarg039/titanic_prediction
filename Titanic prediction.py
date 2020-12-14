#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('titanictrain.csv')


# In[3]:


df.describe()


# In[4]:


Y = df.Survived


# In[5]:


df.columns


# In[6]:


df.head(10)


# In[7]:


df.Name = df.Name.str.split()


# In[8]:


df.Name.dtypes


# In[9]:


df.shape


# In[10]:


for i in range(df.shape[0]) :
    df.Name[i] = df.Name[i][1]
    


# In[11]:


df.Name


# In[12]:


df.Name.unique()


# In[13]:


df.Name.value_counts()


# In[14]:


df.Name.replace(['Dr.','Rev.','y','Planke,','Impe,','Gordon,','Major.','Mlle.','Col.','Walle,','Don.','Cruyssen,','Melkebeke,','Shawah,','Velde,','Ms.','Pelsmaeker,','Mme.','the','Billiard,','Steen,','Carlo,','der','Capt.','Jonkheer.','Messemaeker,','Mulder,'],'0', inplace = True)


# In[15]:


df.Name.replace(['Mr.','Miss.','Mrs.','Master.'],['1','2','3','4'], inplace = True)


# In[16]:


df.Name.astype('int64')


# In[17]:


df[['Survived', 'Name']].groupby('Name').mean()


# In[18]:


df.Embarked.replace(np.nan, 'S', inplace = True)


# In[19]:


df.Embarked.replace(['S','C','Q'],['0','1','2'], inplace = True)
df.Embarked.astype('int64')


# In[20]:


df[['Survived','Embarked']].groupby('Embarked').mean()


# In[21]:


df.Sex.replace(['male','female'],['0','1'], inplace = True)
df.Sex.astype('int64')


# In[22]:


df[['Survived','Sex']].groupby('Sex').mean()


# In[23]:


df[['Survived','Pclass']].groupby('Pclass').mean()


# In[24]:


df.SibSp.value_counts()


# In[25]:


df[['Survived','SibSp']].groupby('SibSp').mean()


# In[26]:


df[['Survived','Parch']].groupby('Parch').mean()


# In[27]:


fare0 = df.loc[df.Survived ==0, :]
fare0 = fare0.Fare
fare1 = df.loc[df.Survived == 1, :]
fare1 = fare1.Fare


# In[28]:


plt.subplot(121)
plt.hist(fare0, label = 'Dead', bins = 20)
plt.xlabel('fare')
plt.xticks([0,40,80,120,160,200,240,280,320,360,400,440,480,520])

plt.legend()

plt.subplot(122)
plt.hist(fare1, label = 'Alive', bins = 20)
plt.xlabel('fare')
plt.yticks([0,50,100,150,200,250,300])
plt.xticks([0,40,80,120,160,200,240,280,320,360,400,440,480,520])
plt.legend()
plt.tight_layout()

plt.show()


# In[29]:


df.columns


# In[30]:


for i in range(df.shape[0]) :
    df.Fare[i] = int(df.Fare[i])


# In[31]:


df.Fare.max()


# In[32]:


df.Fare.replace(np.arange(0,20), 0, inplace = True)


# In[33]:


df.Fare.replace(np.arange(20,40), 1, inplace = True)
df.Fare.replace(np.arange(40,513), 2, inplace = True)


# In[34]:


df.Fare.value_counts()


# In[35]:


df.Fare.astype('int64')


# In[36]:


df.describe()


# In[37]:


df.drop(columns = ['Cabin','Ticket','Age','Survived'],inplace = True)


# In[38]:


train_x, test_x, train_y, test_y = tts(df,Y)


# In[39]:


rfc = RandomForestClassifier(criterion = 'entropy', max_depth = 5, max_leaf_nodes = None, min_impurity_decrease = 0.005, warm_start = True)
rfc.fit(train_x, train_y)
yhat = rfc.predict(test_x)


# In[40]:


confusion_matrix(yhat, test_y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




