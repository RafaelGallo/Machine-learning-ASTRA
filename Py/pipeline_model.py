#!/usr/bin/env python
# coding: utf-8

# # Modelo - Pipeline classificação 
# - Marketing campaign

# In[1]:


from platform import python_version

print('Versão python neste Jupyter Notebook:', python_version())


# In[2]:


import sklearn
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib as m
import warnings

import missingno as msno
import plotly.graph_objects as go
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer

warnings.filterwarnings("ignore")


# In[3]:


get_ipython().run_line_magic('reload_ext', 'watermark')

get_ipython().run_line_magic('watermark', '-a "Rafael Gallo" --iversions')


# In[5]:


# Configuração para os gráficos largura e layout dos graficos

plt.rcParams["figure.figsize"] = (18, 5)

plt.style.use('fivethirtyeight')
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

m.rcParams['axes.labelsize'] = 25
m.rcParams['xtick.labelsize'] = 25
m.rcParams['ytick.labelsize'] = 25
m.rcParams['text.color'] = 'k'


# # Base dados

# In[6]:


# Carregando base dados
data = pd.read_csv("marketing_campaign.csv", sep='\t')
data


# In[7]:


# Visualizando os 5 primeiros dados
data.head()


# In[8]:


# Visualizando os 5 cinco últimos dados
data.tail()


# In[9]:


# Visualizando os infos dados
data.info()


# In[10]:


# Tipos de dados
data.dtypes


# In[11]:


# Visualizando linhas colunas
data.shape


# In[12]:


# Total de colunas e linhas - data_train

print("Números de linhas: {}" .format(data.shape[0]))
print("Números de colunas: {}" .format(data.shape[1]))


# In[13]:


# Exibindo valores ausentes e valores únicos

print("\nMissing values :  ", data.isnull().sum().values.sum())
print("\nUnique values :  \n",data.nunique())


# In[14]:


# Verificando dados duplicados
data.duplicated()


# In[15]:


# Visualizando dados nulos
data.isnull().sum()


# # Estatística descritiva

# In[16]:


# Exibindo estatísticas descritivas visualizar alguns detalhes estatísticos básicos como percentil, média, padrão, etc. 
# De um quadro de dados ou uma série de valores numéricos.

data.describe().T


# In[17]:


# Matriz correlação de pares de colunas, excluindo NA / valores nulos.

corr = data.corr()
corr


# In[18]:


plt.figure(figsize=(18.2, 8))

ax = sns.distplot(data['Year_Birth']);
plt.title("Distribuição normal", fontsize=20)
plt.axvline(data['Year_Birth'].mean(), color='k')
plt.axvline(data['Year_Birth'].median(), color='r')
plt.axvline(data['Year_Birth'].mode()[0], color='g');


# In[19]:


# Verificando os dados no boxplot valor total verificando possíveis outliers
plt.figure(figsize=(18.2, 8))

sns.boxplot(x="Income", y="Education", data = data)
plt.title("Educação dos consumidores")
plt.xlabel("Total")
plt.ylabel("Educação")


# In[20]:


plt.figure(figsize=(14, 10))
sns.heatmap(data.corr(), cmap="Blues")
plt.title("Correlations", size=16)
plt.show()


# In[21]:


# Gráfico da matriz de correlação

plt.figure(figsize=(35.5,30))
ax = sns.heatmap(corr, annot=True, cmap='YlGnBu');
plt.title("Matriz de correlação")


# In[22]:


plt.figure(figsize=(30.5,20))

mask = np.triu(np.ones_like(corr, dtype = bool))
sns.heatmap(corr, mask = mask, annot = True, fmt = '.2g', linewidths = 1)
plt.show()


# # Análise dados

# In[23]:


plt.figure(figsize=(20, 10))

plt.title("Educação dos consumidores")
sns.countplot(data["Education"])
plt.xlabel("Educação")
plt.ylabel("Total")


# In[24]:


plt.figure(figsize=(20, 10))

sns.histplot(data["Year_Birth"])
plt.title("Ano de nascimento dos consumidores")
plt.xlabel("Ano")
plt.ylabel("Idades")


# In[25]:


plt.figure(figsize=(20, 10))

sns.histplot(data["Income"])
plt.title("Renda dos consumidores")
plt.xlabel("Total da renda")
plt.ylabel("Total")


# In[26]:


# Gráfico condições de vida por valor do imóvel
plt.figure(figsize=(35.5, 15))

plt.title("Formações dos consumidores")
ax = sns.barplot(x="Education", y="Income", data = data, hue="Marital_Status")
plt.ylabel("Valor")
plt.xlabel("Formações")


# In[27]:


# Região das vendas dos imóveis pela área
plt.figure(figsize=(18.2, 8))

plt.title("Renda dos consumidores")
ax = sns.scatterplot(x="MntWines", y="Income", data = data, hue = "Marital_Status")
plt.xlabel("Renda ")
plt.ylabel("Valor")


# In[28]:


plt.figure(figsize=(20, 10))

# Graduation, PHD, master, basic, 2n cycle

plt.pie(data.groupby("Education")['Education'].count(), labels=["Graduation", "PHD", "master", "basic", "2n_cycle"], autopct = "%1.1f%%");
plt.title("Perfil de escolariedade dos consumidores")
plt.xlabel("Total consumidores")
plt.ylabel("Total")


# # Análise de dados = Univariada

# In[29]:


data.hist(bins = 40, figsize=(20.2, 20))
plt.show()


# In[30]:


fig1 , axes = plt.subplots(nrows=3,ncols=3 , figsize = (20,20))

sns.distplot(data["NumWebVisitsMonth"] , ax=axes[0, 0] )
sns.distplot(data["AcceptedCmp1"] ,  ax=axes[0, 1])
sns.distplot(data["AcceptedCmp2"] , ax=axes[0, 2])
sns.distplot(data["AcceptedCmp3"], ax=axes[1, 0])
sns.distplot(data["AcceptedCmp4"] , ax=axes[1, 1])
sns.distplot(data["AcceptedCmp5"] , ax=axes[1, 2])
sns.distplot(data["Recency"] , ax=axes[2, 0])
sns.distplot(data["MntWines"], ax=axes[2, 1])
sns.countplot(data["Education"], ax=axes[2, 2])

plt.show()


# In[31]:


# Nós imputamos os valores omissos na coluna "Renda" com a mediana dessa coluna em particular

data["Income"].fillna(data["Income"].median(), inplace=True)
data


# # Feature Engineering

# In[32]:


data["Age"] = 2022 - data["Year_Birth"]

data["Money_Spent"] = (data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"])
data["PurchaseNum"] = data["NumWebPurchases"] + data["NumCatalogPurchases"] + data["NumStorePurchases"]
data["Children"] = data["Kidhome"] + data["Teenhome"]
data["Marital_Status"] = data["Marital_Status"].replace({"Married": "Together", "Alone": "Single", "Absurd": "Single", "Divorced": "Single", "Widow": "Single", "Divorced": "Single", "YOLO": "Single"})

# exluindo coluna
data.drop(["Education", "Marital_Status", "Dt_Customer"], axis=1, inplace=True)
data


# # Treino e Teste
# 
# - Treino e teste da base de dados da coluna price e idade

# In[47]:


x = data.drop('Response',axis=1).values
y = data['Response'].values


# In[48]:


# Total de linhas e colunas dados variável x
x.shape


# In[49]:


# Total de linhas e colunas dados variável y
y.shape


# # Escalonamento
# 
# - Escalonamento uma forma de contornar os problemas relacionados à escala, mantendo a informação estatística dos dados. O procedimento consiste em realizar uma transformação sobre o conjunto original dos dados de modo que cada variável apresente média zero e variância unitária.

# In[50]:


# Escalonamento dos dados

# Importando biblioteca a biblioteca
from sklearn.preprocessing import StandardScaler

# Criando o escalonamento
model_scaler = StandardScaler()

# Treinamneto do escalonamento
model_scaler_fit = model_scaler.fit_transform(x)

# Visualizando linhas e colunas do escalonamento
model_scaler_fit.shape


# In[51]:


# Importação da biblioteca sklearn para treino e teste do modelo

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, # Variável x
                                                    y, # Variável y
                                                    test_size=0.2, # Divivindo os dados em 20% para treino e 80% para teste
                                                    random_state = 0) # Random state igual a zero


# In[42]:


# Total de linhas e colunas e linhas dos dados de treino x
x_train.shape


# In[43]:


# Total de linhas dos dados de treino y
y_train.shape


# In[44]:


# Total de linhas e colunas dos dados de treino x teste 
x_test.shape


# In[45]:


# Total de linhas e colunas dos dados de treino y teste 
y_test.shape


# # Model pipeline

# In[52]:


# Modelo pipeline

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

data_pipeline = Pipeline([
    ("scaler", StandardScaler()), # Scaler : Para pré-processamento de dados, ou seja, transforme os dados em média zero e variância de unidade usando o StandardScaler ().
    ("selector", VarianceThreshold()), # Seletor de recurso : Use VarianceThreshold () para descartar recursos cuja variação seja menor que um determinado limite definido.
    ("classifier", KNeighborsClassifier()) # Classificador : KNeighborsClassifier (), que implementa o classificador de k-vizinho mais próximo e seleciona a classe dos k pontos principais, que estão mais próximos do exemplo de teste.
])

data_pipeline_fit = data_pipeline.fit(x_train, y_train)
data_pipeline_score = data_pipeline.score(x_train, y_train)

print('Treinamento base treino - Pipeline: ' + str(data_pipeline.score(x_train,y_train)))
print('Treinamento base teste - Pipeline: ' + str(data_pipeline.score(x_test,y_test)))


# In[54]:


# Previsão do pipeline do modelo
data_pipeline_pred_1 = data_pipeline.predict(x_test)


# In[55]:


# Matrix confusion ou Matriz de Confusão
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

matrix_1 = confusion_matrix(y_test, data_pipeline_pred_1)
plot_confusion_matrix(matrix_1, show_normed=True, colorbar=False, class_names=['Acesso', 'Não acesso']) 


# In[56]:


# Accuracy do pipeline
from sklearn.metrics import accuracy_score

accuracy_pipeline_1 = accuracy_score(y_test, data_pipeline_pred_1)
print("Accuracy KNN - Pipeline: %.2f" % (accuracy_pipeline_1 * 100))


# In[57]:


# Curva ROC do modelo
from sklearn.metrics import roc_curve, roc_auc_score

roc = data_pipeline.predict_proba(x_test)[:,1]
tfp, tvp, limite = roc_curve(y_test, roc)
print('roc_auc', roc_auc_score(y_test, roc))

plt.subplots(1, figsize=(5,5))
plt.title('Curva ROC')
plt.plot(tfp,tvp)
plt.xlabel('Especifidade')
plt.ylabel('Sensibilidade')
plt.plot([0, 1], ls="--", c = 'red')
plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green')
plt.show()


# In[58]:


# Classification report
from sklearn.metrics import classification_report

classification = classification_report(y_test, data_pipeline_pred_1)
print("Modelo - Pipeline 1")
print()
print(classification)


# In[59]:


# Métricas do modelo 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

precision = precision_score(y_test, data_pipeline_pred_1)
Recall = recall_score(y_test, data_pipeline_pred_1)
Accuracy = accuracy_score(y_test, data_pipeline_pred_1)
F1_Score = f1_score(y_test, data_pipeline_pred_1)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# # Pipeline 2
# **Pipeline 2 - Decision Tree Classifier**

# In[60]:


# Pipeline decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

data_pipeline_2 = Pipeline([
    ("scaler", StandardScaler()), # Scaler : Para pré-processamento de dados, ou seja, transforme os dados em média zero e variância de unidade usando o StandardScaler ().
    ("selector", VarianceThreshold()), # Seletor de recurso : Use VarianceThreshold () para descartar recursos cuja variação seja menor que um determinado limite definido.
    ("classifier", DecisionTreeClassifier(max_depth=4, random_state=0)) # Classificador : DecisionTreeClassifier (), que implementa o classificador de árvore decisão Árvores de decisão são métodos de classificação que podem extrair regras simples sobre os recursos de dados que são inferidos do conjunto de dados de entrada
])

data_pipeline2_fit = data_pipeline_2.fit(x_train, y_train)
data_pipeline2_score = data_pipeline_2.score(x_train, y_train)

print('Treinamento base treino - Pipeline: ' + str(data_pipeline_2.score(x_train,y_train)))
print('Treinamento base teste - Pipeline: ' + str(data_pipeline_2.score(x_test,y_test)))


# In[61]:


# Previsão do pipeline 
data_pipeline_pred_2 = data_pipeline_2.predict(x_test)
data_pipeline_pred_2


# In[62]:


# Accuracy do pipeline
accuracy_pipeline_2 = accuracy_score(y_test, data_pipeline_pred_2)
print("Accuracy Pipeline 2: %.2f" % (accuracy_pipeline_2 * 100))


# In[63]:


# A matrix confusion do modelo
matrix_confusion_1 = confusion_matrix(y_test, data_pipeline_pred_2)
plot_confusion_matrix(matrix_confusion_1, show_normed=True, colorbar=False, class_names=['Acesso', 'Não acesso'])


# In[64]:


# Curva ROC do modelo
roc = data_pipeline_2.predict_proba(x_test)[:,1]
tfp, tvp, limite = roc_curve(y_test, roc)
print('roc_auc', roc_auc_score(y_test, roc))

plt.subplots(1, figsize=(5,5))
plt.title('Curva ROC')
plt.plot(tfp,tvp)
plt.xlabel('Especifidade')
plt.ylabel('Sensibilidade')
plt.plot([0, 1], ls="--", c = 'red')
plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green')
plt.show()


# In[65]:


# Classification_report
classification = classification_report(y_test, data_pipeline_pred_2)
print("Modelo Pipeline 2")
print()
print(classification)


# In[67]:


# Méricas do modelo
precision = precision_score(y_test, data_pipeline_pred_2)
Recall = recall_score(y_test, data_pipeline_pred_2)
Accuracy = accuracy_score(y_test, data_pipeline_pred_2)
F1_Score = f1_score(y_test, data_pipeline_pred_2)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# **Pipeline 3 - Naive bayes**

# In[68]:


# Modelo naive bayes

# Importando biblioteca ML naive bayes
from sklearn.naive_bayes import GaussianNB

# Pipeline Naive bayes
data_pipeline_3 = Pipeline([
    ("scaler", StandardScaler()), 
    ("selector", VarianceThreshold()), 
    ("classifier", GaussianNB())])

data_pipeline3_fit = data_pipeline_3.fit(x_train, y_train)
data_pipeline3_score = data_pipeline_3.score(x_train, y_train)

print('Treinamento base treino - Pipeline: ' + str(data_pipeline_3.score(x_train,y_train)))
print('Treinamento base teste - Pipeline: ' + str(data_pipeline_3.score(x_test,y_test)))


# In[69]:


# Previsão do modelo - Naive bayes

# Previsão do pipeline
data_pipeline_pred_3 = data_pipeline_3.predict(x_test)
data_pipeline_pred_3


# In[70]:


# Accuracy do pipeline
accuracy_pipeline_3 = accuracy_score(y_test, data_pipeline_pred_3)
print("Accuracy pipeline 3: %.2f" % (accuracy_pipeline_3 * 100))


# In[71]:


# A matrix confusion pipeline
matrix_confusion_4 = confusion_matrix(y_test, data_pipeline_pred_3)
plot_confusion_matrix(matrix_confusion_4, show_normed=True, colorbar=False, class_names=['Acesso', 'Não acesso'])


# In[72]:


# Curva ROC do pipeline
roc = data_pipeline_3.predict_proba(x_test)[:,1]
tfp, tvp, limite = roc_curve(y_test, roc)
print('roc_auc', roc_auc_score(y_test, roc))

plt.subplots(1, figsize=(5,5))
plt.title('Curva ROC')
plt.plot(tfp,tvp)
plt.xlabel('Especifidade')
plt.ylabel('Sensibilidade')
plt.plot([0, 1], ls="--", c = 'red')
plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green')
plt.show()


# In[73]:


# Classification report do modelo
class_report = classification_report(y_test, data_pipeline_pred_3)
print("Modelo 03 - Pipeline")
print("\n")
print(class_report)


# In[74]:


# Metricas do pipeline
precision = precision_score(y_test, data_pipeline_pred_3)
Recall = recall_score(y_test, data_pipeline_pred_3)
Accuracy = accuracy_score(y_test, data_pipeline_pred_3)
F1_Score = f1_score(y_test, data_pipeline_pred_3)

precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

precisao.sort_values(by = "Resultado", ascending = False)


# In[75]:


# Resultados - Modelos machine learning

modelos = pd.DataFrame({
    
    "Models" :["Pipeline 1: K-NN", 
               "Pipeline 2: Decision tree", 
               "Pipeline 3: Naive bayes"],

    "Acurácia" :[accuracy_pipeline_1, 
                      accuracy_pipeline_2, 
                      accuracy_pipeline_3]})

modelos_2 = modelos.sort_values(by = "Acurácia", ascending = False)
modelos_2.to_csv("modelos_2.csv")
modelos_2


# In[76]:


# Salvando pipeline Machine learning

import pickle    
    
with open('data_pipeline_pred_1.pkl', 'wb') as file:
    pickle.dump(data_pipeline_pred_1, file)
    
with open('data_pipeline_pred_2.pkl', 'wb') as file:
    pickle.dump(data_pipeline_pred_2, file)
    
with open('data_pipeline_pred_3.pkl', 'wb') as file:
    pickle.dump(data_pipeline_pred_3, file)

