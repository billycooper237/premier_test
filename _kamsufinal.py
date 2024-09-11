#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector,make_column_transformer
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression


# <h1>analyse de forme</h1>

# importaion de notre dataset nous allons presenter notre dataset en partie question pour nous de voir quelques premieres lignes

# In[2]:


data_train = pd.read_csv("C:\\Users\\GENERAL STORES\\Desktop\\laboratoire\\train.csv")
data_test = pd.read_csv("C:\\Users\\GENERAL STORES\\Desktop\\laboratoire\\test.csv")
data_train.head()


# nous allons a present prenseter la taille de notre dataset question de voir son ettendu 

# In[5]:


data_train.shape


# ainsi nous pouvons constater que notre dataset possede 1640 ligne et 81 colones

# In[3]:


data_train.dtypes.value_counts()


# apres avoir presenter la taille de notre data , il est important de connaitre de type de nos variables dans notre data nous remarquons qu'elle possede 43 variables de types objets ,35 variables de type entier et en fin 3 variables de type reel

# In[3]:


pd.set_option('display.max_rows', 10)

data_train


# In[ ]:





# In[5]:


sns.heatmap(data_train.isna(),cbar= False)


# sur ce schema il est question de presenter notre dataset sous forme de graphe pour mieux voir notre data.ansi,nous pouvons observer deux zones sur notre graphe une zone de noir qui reprente toutes les variables ne contenant pas de valeurs manquantes,et une wone blanche qui represente des variables pour les quelles les valeurs manquantes son tres import.il est question pour de les eliminer pour une prediction plus sofistique

# In[4]:


pd.set_option('display.max_rows',81)
(data_train.isna().sum()/data_train.shape[0]).sort_values(ascending = True)


# ici, nous presentons toutes les variables suivies du poucentage en valeurs manquantes.

# Elimination de toutes les valeurs manquantes

# In[7]:


data_train = data_train[data_train.columns[data_train.isna().sum()/data_train.shape[0]<0.75]]
data_train.head()


# In[8]:


sns.heatmap(data_train.isna())


# nous remarquons juste sur ce heatmap que la zone blanche a considerablement diminuer cela signifi que nos valeurs manquantes ont bien et belle ete diminue.

# nous allons a present analyser la target

# In[12]:


#analyse de la target
pd.set_option('display.max_rows',77)
(data_train['SalePrice'].value_counts()).sort_values(ascending=True)


# In[10]:


data_train.info()


# In[11]:


data_train.describe()


# In[12]:


data_train.describe(include='O')


# <h1>visualisation des variables en fonction de la target</h1>

# In[18]:


X = data_train.drop(['SalePrice'],axis=1)
y = data_train[['SalePrice']]


# In[19]:


sns.catplot(x='MSSubClass',y='MSZoning',data=data_train,hue='SalePrice')


# sur ce graphe,nous representons deux variables en fonction de notre target a savoir MISwoning et MSSubClass:nous remarquons que le prix des logement peuvent varier selon le niveau de la sous classe et la zone geographique .entre une sous classe de 50 a 75,et une situation de zone industrielle a moyenne densite nous observons que les prix varient entre 150k et 450k

# In[20]:


sns.catplot(x='PoolArea',y='LotArea',data=data_train,hue='SalePrice')


# sur ce graphe nous observons que pour une piscine de superficie 555 m*m et un terrain entre 0 et 50K lr prix est tres eleve 
# par contre pour une maison sans piscine et une superficie de terrain inferieur a 25K ,le prix est bat

# In[21]:


data_train2 =data_train[['GrLivArea','HouseStyle','Neighborhood','BldgType','BsmtExposure','GarageType','Fence','MoSold','Fireplaces','SalePrice']]
sns.pairplot(data_train2,hue='SalePrice')


# ces graphes mettent en relation quelques variables en fonction de la target .nous observons que les prix peuvent changer en fonction des relations des variables comme vue sur les graphes precedent

# In[13]:


X = data_train.drop(['SalePrice'],axis=1)
y = data_train[['SalePrice']]


# In[14]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[16]:


numerical_feacture = make_column_selector(dtype_include=np.number)
categorial_feacture = make_column_selector(dtype_exclude=np.number)


# In[21]:


pipeline_numerical = make_pipeline(SimpleImputer(),StandardScaler())
pipeline_categorial = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore'))

application = make_column_transformer((pipeline_numerical,numerical_feacture),(pipeline_categorial,categorial_feacture))
application


#  ecrivons un algorithme qui va choisir le meilleur model parmis 3 choisi pour ce problemme de regression

# In[52]:


modeles = {
    'SVR':SVR(C=100000),
    'LinearRegression':LinearRegression(),
    'RandomForestRegressor':RandomForestRegressor()
    
}
for nom,model in modeles.items():
    modele = make_pipeline(application,model)
    
    modele.fit(X_train,y_train)
    a = modele.score(X_train,y_train)
    print(f"{nom}:{a}")


# In[42]:


model = make_pipeline(application,RandomForestRegressor())
model


# In[43]:


model.fit(X_train,y_train)


# In[45]:


model.score(X_train,y_train)


# In[47]:


model.score(X_test,y_test)


# In[48]:


model.predict(X_test)


# In[ ]:





# 
# 
# 
# 
# 
