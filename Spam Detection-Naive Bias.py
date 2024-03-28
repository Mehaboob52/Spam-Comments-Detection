#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set()


# In[2]:


files=glob.glob("D:\ml datasets\youtube-dataset\\*.csv")


# In[3]:


files


# In[4]:


all_df = []
for i in files:
    all_df.append(pd.read_csv(i).drop(['COMMENT_ID','AUTHOR','DATE'],axis=1))


# In[5]:


all_df[0]


# In[6]:


df=pd.concat(all_df,axis=0,ignore_index=True)


# In[7]:


df


# In[8]:


df.isnull().sum()


# In[9]:


df['CLASS'].value_counts()


# # Message Sample 1

# In[10]:


message_sample=['This is a Dog']

vectorizer_sample=CountVectorizer()
vectorizer_sample.fit(message_sample)
vectorizer_sample.transform(message_sample).toarray()


# In[11]:


vectorizer_sample.get_feature_names_out()


# In[12]:


vectorizer_sample.transform(["This is a cat"]).toarray()


# # Message Sample 2

# In[13]:


message_sample2=['This is a dog and that is a dog','This is a cat']
vectorizer_sample=CountVectorizer()
vectorizer_sample.fit(message_sample2)
vectorizer_sample.transform(message_sample2).toarray()


# In[14]:


vectorizer_sample.get_feature_names_out()


# In[15]:


vectorizer_sample.transform(['Those are birds']).toarray()


# In[16]:


vectorizer_sample.get_feature_names_out()


# # Pre Processing

# In[17]:


x=df['CONTENT']
y=df['CLASS']


# In[18]:


# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=365,stratify=y) 
#stratify divided equally classes in Y both train and test data


# In[19]:


y_train.value_counts(normalize=True)


# In[20]:


y


# # Tokenizing the Youtube Comments or Converting into Numerical Data

# In[21]:


vectorizer= CountVectorizer()


# In[22]:


x_train_transf=vectorizer.fit_transform(x_train)
x_test_transf=vectorizer.transform(x_test)


# In[23]:


x_train_transf.toarray()


# In[24]:


x_train_transf.shape


# In[25]:


x_test_transf.shape


# In[26]:


model=MultinomialNB()


# In[27]:


model.fit(x_train_transf,y_train)


# In[28]:


model.get_params()


# # Performing the evaluation on the dataset

# In[29]:


pred=model.predict(x_test_transf)


# In[30]:


cm = confusion_matrix(y_test, pred)


# In[31]:


cm


# In[32]:


print(classification_report(y_test,pred,target_names=['Ham','Spam']))


# In[33]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred)


# In[34]:


accuracy


# In[35]:


spam_proba = model.predict_proba(x_test_transf).round(3)[:,1];

df_scatter = pd.DataFrame()

df_scatter['True class'] = y_test
df_scatter['Predicted class'] = pred
df_scatter['Predicted probability (spam)'] = spam_proba

df_scatter = df_scatter.reset_index(drop = True)

palette_0 = sns.color_palette(['#000000'])
palette_1 = sns.color_palette(['#FF0000'])

df_scatter_0 = df_scatter[df_scatter['True class'] == 0].reset_index(drop = True)
df_scatter_1 = df_scatter[df_scatter['True class'] == 1].reset_index(drop = True)

sns.set()

fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12,5))
fig.tight_layout(pad = 3)

sns.scatterplot(x = 'Predicted probability (spam)', 
                y = np.zeros(df_scatter_0.shape[0]), 
                data = df_scatter_0,
                hue = 'True class', 
                s = 50,
                markers = ['o'],
                palette = palette_0,
                style = 'True class',
                legend = False, 
                ax = ax1).set(yticklabels=[])

ax1.set_title('Probability distribution of comments belonging to the true \'ham\' class')
ax1.vlines(0.5, -1, 1, linestyles = 'dashed', colors = 'red');


sns.scatterplot(x = 'Predicted probability (spam)', 
                y = np.zeros(df_scatter_1.shape[0]), 
                hue = 'True class', 
                data = df_scatter_1,
                s = 50,
                palette = palette_1,
                markers = ['X'],
                style = 'True class',
                legend = False, 
                ax = ax2).set(yticklabels=[])

ax2.set_title('Probability distribution of comments belonging to the true \'spam\' class')

ax2.vlines(0.5, -1, 1, linestyles = 'dashed', colors = 'red');


# In[62]:


predict_data=vectorizer.transform(['Good content'])


# In[63]:


result=model.predict(predict_data)


# In[64]:


if result[0]==1:
    print("Spam message")
else:
    print("Not a Spam Message. Its Normal")


# In[ ]:




