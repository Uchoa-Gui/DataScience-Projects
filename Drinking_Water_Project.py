#!/usr/bin/env python
# coding: utf-8

# In[150]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter


# In[48]:


dados = pd.read_csv('water_potability.csv')


# In[49]:


dados.head()


# In[50]:


dados.shape


# ### Data Visualization

# In[51]:


fig = px.histogram(dados, x = 'Solids', template='plotly_white',
            color = 'Potability', histfunc = 'count', marginal = 'box', opacity=0.9)

fig.update_layout(font_family = 'monospace', bargap = 0.4, 
                  title = dict(text="TDS - Total Dissolved Solids",x= 0.48, y = 0.95),
                  xaxis_title_text='Solids')


# In[52]:


fig = px.histogram(dados, x = 'Sulfate', template='plotly_white',
            color = 'Potability', histfunc = 'count', marginal = 'box', opacity=0.9)

fig.update_layout(font_family = 'monospace', bargap = 0.4,
                  title = dict(text="Sulfate Distribution",x= 0.54, y = 0.95),
                  xaxis_title_text='Sulfate (mg/L)')

fig.add_vline(x = 250, line_width = 0.8, line_dash = 'dot')
fig.add_annotation(x = 210, y = 90, text = '<250 mg/L is considered <br>safe to drink', showarrow  = False)
fig.show()


# In[53]:


fig = px.histogram(dados, x = 'Hardness', template='plotly_white',
            color = 'Potability', histfunc = 'count', marginal = 'box', opacity=0.9)

fig.update_layout(font_family = 'monospace', bargap = 0.4,
                  title = dict(text="Water Hardness",x= 0.5, y = 0.95),
                  xaxis_title_text='Hardness (mg/L)')

fig.add_vline(x = 75, line_dash = 'dot', line_width = 0.8)
fig.add_annotation(text = '<75mg/L is generally  <br> considered soft', x = 50, y = 150,showarrow=False)
fig.add_vline(x = 150, line_dash = 'dot', line_width = 0.8)
fig.add_annotation(text = '75<Hardness<150(mg/L) is <br> moderately hard', x = 120, y = 150,showarrow=False)
fig.add_vline(x = 300, line_dash = 'dot', line_width = 0.8)
fig.add_annotation(text = '151<Hardness<300(mg/L) is <br> hard', x = 260, y = 150,showarrow=False)
fig.add_annotation(text = '>300mg/L is very hard', x = 350, y = 150,showarrow=False)


# In[54]:


fig = px.histogram(dados, x = 'Turbidity', template='plotly_white',
            color = 'Potability', histfunc = 'count', marginal = 'box', opacity=0.9)

fig.update_layout(font_family = 'monospace', bargap = 0.4,
                  title = dict(text="Water Turbidity",x= 0.45, y = 0.95),
                  xaxis_title_text='Turbidity (NTU)')

fig.add_vline(x = 5, line_width = 0.8, line_dash = 'dot')
fig.add_annotation(text = '<5 NTU is safe <br> to drink', x=5.5, y=120,showarrow = False)


# In[55]:


fig = px.histogram(dados, x = 'ph', template='plotly_white',
            color = 'Potability', histfunc = 'count', marginal = 'box', opacity=0.9)

fig.update_layout(font_family = 'monospace', bargap = 0.4,
                  title = dict(text="Water pH",x= 0.49, y = 0.95),
                  xaxis_title_text='pH')

fig.add_vline(x = 7, line_width = 2.0, line_dash = 'dot')
fig.add_annotation(text = 'pH=7 is<br> neutral', x = 6.0, y = 160, showarrow=False)
fig.add_annotation(text = 'pH<7 is<br> acid', x = 3.5, y = 100, showarrow=False)
fig.add_annotation(text = 'pH>7 is<br> alkaline', x = 10, y = 100, showarrow=False)


# In[56]:


fig = px.histogram(dados, x = 'Conductivity', template='plotly_white',
            color = 'Potability', histfunc = 'count', marginal = 'box', opacity=0.9)

fig.update_layout(font_family = 'monospace', bargap = 0.3,
                  title = dict(text="Water Conductivity",x= 0.45, y = 0.95),
                  xaxis_title_text='Conductivity (µS/cm)')

fig.add_vline(x = 400, line_width = 0.8, line_dash = 'dot')
fig.add_annotation(text='<400 µS/cm is considered<br> safe to drink', x=330, y = 160,showarrow=False)


# In[57]:


fig = px.histogram(dados, x = 'Chloramines', template='plotly_white',
            color = 'Potability', histfunc = 'count', marginal = 'box', opacity=0.9)

fig.update_layout(font_family = 'monospace', bargap = 0.4,
                  title = dict(text="Water Chloramines",x= 0.5, y = 0.95),
                 xaxis_title_text='Chloramines (ppm)')

fig.add_vline(x = 4, line_width = 0.8, line_dash = 'dot')
fig.add_annotation(text='Chloramine levels up to 4 mg/L<br> are considered safe<br> in drinking water', x = 2, y = 120,
                  showarrow=False)


# In[58]:


fig = px.histogram(dados, x = 'Organic_carbon', template='plotly_white',
            color = 'Potability', histfunc = 'count', marginal = 'box', opacity=0.9)

fig.update_layout(font_family = 'monospace', bargap = 0.4,
                  title = dict(text="Organic Carbon Distribution",x= 0.5, y = 0.95),
                 xaxis_title_text='Organic Carbon (mg/L)')

fig.add_vline(x=10, line_width = 0.8, line_dash = 'dot')
fig.add_annotation(text='Typical TOC values<br> are up to 10mg/L',x=7, y = 120,showarrow=False)


# In[59]:


fig = px.histogram(dados, x = 'Trihalomethanes', template='plotly_white',
            color = 'Potability', histfunc = 'count', marginal = 'box', opacity=0.9)

fig.update_layout(font_family = 'monospace', bargap = 0.4,
                  title = dict(text="Trihalomethanes Distribution",x= 0.5, y = 0.95),
                 xaxis_title_text='Trihalomethanes (µg/L)')

fig.add_vline(x=80,line_width=0.8,line_dash = 'dot')
fig.add_annotation(text='THM levels up to 80µg/L <br> are considered safe<br> in drinking water', x=95,y=120,
                  showarrow=False)


# In[203]:


#Visualizing the correlation between the features:
sns.heatmap(dados.corr(), vmin = -0.5, cmap = 'Blues');


# From the heatmap we can see that there is no correlation between the features in this dataset.

# ### Data treatment

# In[60]:


dados.isnull().sum()


# In[61]:


#Treating the Null values by replacing them with their mean
dados['ph'] = dados['ph'].fillna(dados['ph'].mean())
dados['Sulfate'] = dados['Sulfate'].fillna(dados['Sulfate'].mean())
dados['Trihalomethanes'] = dados['Trihalomethanes'].fillna(dados['Trihalomethanes'].mean())


# In[62]:


dados.isnull().sum()


# In[63]:


dados.describe()


# In[64]:


#Splitting the data in predictive (x) and classicative (y)
x_water = dados.iloc[:, 0:9].values
y_water = dados.iloc[:, 9].values
x_water.shape, y_water.shape


# In[65]:


#Normalizing the values so that the algorithm doesn't priorize higher values over smaller ones.
from sklearn.preprocessing import StandardScaler


# In[66]:


scaler_water = StandardScaler()
x_water = scaler_water.fit_transform(x_water)


# In[67]:


from sklearn.model_selection import train_test_split
x_water_training, x_water_test, y_water_training, y_water_testing = train_test_split(x_water, y_water, test_size = 0.20,
                                                                                    random_state = 0)


# In[68]:


x_water_training.shape, y_water_training.shape


# In[168]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV   #CV = Cross Validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# In[169]:


#Parameter Tuning:
lr = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
rn = MLPClassifier()


# In[171]:


par_knn = {'n_neighbors': np.arange(1,40), 'p':[1,2]}
grid_knn = GridSearchCV(knn, param_grid=par_knn, cv = 5)

par_dt = {'criterion': ['gini', 'entropy'], 'max_depth':np.arange(1,40),
         'min_samples_leaf':[1,2,4,5,10,20,30,40,80,100]}
grid_dt = GridSearchCV(dt, param_grid=par_dt, cv=5)

par_rf = {'n_estimators': [100,200,350,500], 'min_samples_leaf':[2,10,30]}
grid_rf = GridSearchCV(rf, param_grid=par_rf, cv=5)

par_lr = {'tol': [0.0001, 0.00001, 0.000001],
             'C': [1.0, 1.5, 2],
             'solver': ['lbfgs', 'sag', 'saga']}
grid_lr = GridSearchCV(lr, param_grid=par_lr, cv=5)

par_xgb = {'n_estimators': [50,100,250,400,600,800,1000],
              'learning_rate': [0.2,0.5,0.8,1]}
rs_xgb = RandomizedSearchCV(xgb, param_distributions=par_xgb, cv=5)

par_rn = {'max_iter': [500, 1000, 1500], 'tol': [0.0001, 0.00001],
         'solver':['adam', 'lbfgs', 'sgd']}
grid_rn = GridSearchCV(rn, param_grid=par_nn, cv=5)


# In[172]:


grid_knn.fit(x_water_training, y_water_training)
print('Melhores parâmetros kNN', grid_knn.best_params_)


# In[173]:


grid_dt.fit(x_water_training, y_water_training)
print('Melhores parâmetros Árvores de Decisão', grid_dt.best_params_)


# In[174]:


grid_lr.fit(x_water_training, y_water_training)
print('Melhores parâmetros Regressão Logística', grid_lr.best_params_)


# In[176]:


rs_xgb.fit(x_water_training, y_water_training)
print('Melhores parâmetros XGB', rs_xgb.best_params_)


# In[177]:


grid_rf.fit(x_water_training, y_water_training)
print('Melhores Parâmetros Random Forest', grid_rf.best_params_)


# In[178]:


grid_nn.fit(x_water_training,y_water_training)
print('Melhores parametros rede neural simples', grid_nn.best_params_)


# In[179]:


classifiers = [('kNN', knn), ('Árvore Decisão', dt), ('RandomForest', rf), ('Logistic Regression', lr),
              ('XGBoost', xgb), ('Rede Neural Simples', rn)]


# In[227]:


knn = KNeighborsClassifier(n_neighbors=22, p=2)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_leaf=4, random_state=0)
rf = RandomForestClassifier(min_samples_leaf=2, n_estimators=500)
lr = LogisticRegression(C=1.0, solver='lbfgs', tol=0.0001, random_state=0)
xgb = XGBClassifier(n_estimators = 50, learning_rate = 0.2, random_state=0)
rn = MLPClassifier(activation='tanh', max_iter=1500, solver='sgd', tol=0.00001, random_state=0)


# In[231]:


from sklearn.metrics import accuracy_score, classification_report
for classifier_name, classifier in classifiers:
    classifier.fit(x_water_training, y_water_training)
    pred = classifier.predict(x_water_test)
    acc = accuracy_score(y_water_testing, pred)
    
    
    print('{:s} : {:.3f}'.format(classifier_name, acc))


# ### As we can see, RandomForest got the highest accuracy over the other models.

# ### Analizing the Confusion Matrix

# In[229]:


rf.fit(x_water_training, y_water_training)
pred_rf = rf.predict(x_water_test)
from sklearn.metrics import classification_report
print(classification_report(pred_rf, y_water_testing))


# In[230]:


from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(rf)
cm.fit(x_water_training, y_water_training)
cm.score(x_water_test, y_water_testing)


# ### Percebe-se que o algoritmo consegue identificar com uma precisão de 70% os dados, e consegue classificar com 92% de precisão se a água é potavel (0) e 32% de precisão para classificar a água como não potável (1).

# Esse algoritmo utilizado tem uma boa usabilidade para
