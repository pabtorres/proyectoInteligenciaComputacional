#!/usr/bin/env python
# coding: utf-8

# # Código Proyecto de Inteligencia Compuacional

# Variables notables

# In[ ]:


anchoVentana = 400
pasoVentana = 200
kDeAnova = 48


# Se importan librerías

# In[ ]:


# Importar Librerías
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import math
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import accuracy_score
from time import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from numpy import array
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import kurtosis
from scipy.stats import skew


# In[ ]:


from sklearn import svm, datasets
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  # naive bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# Se suben los archivos para poder ser utilizados

# In[ ]:


# Upload data
get_ipython().system('mkdir /content/proyectoEMGdos')
get_ipython().run_line_magic('cd', '/content/proyectoEMGdos')
from google.colab import files
uploaded = files.upload()


# In[ ]:


get_ipython().system('unzip EMG_data_for_gestures-master.zip')


# In[8]:


get_ipython().system('ls')


# Se importa librería para poder moverse entre los directorios

# In[9]:


# Se importa OS para moverse entre los directorios
import os
curDir = os.getcwd()
print(curDir)
os.listdir() #ls


# In[ ]:


# Entrar a la carpeta
os.chdir('EMG_data_for_gestures-master')


# In[11]:


# Verificar que se esté dentro de la carpeta
get_ipython().system('ls')


# In[ ]:


## función que de acuerdo al valor de wid, step, uniformWindow, eliminarCeroSiete,
## fullData, procesa la información utilizando el algoritmo de la ventana deslizante
## extrayendo características al conjunto de datos obtenido
def windows(dfArray,wid,step,uniformWindow=False,eliminarCeroSiete = True, fullData = False):
  colsList = ['mu1','std1','krt1','vpp1','skew1','rms1','mu2','std2','krt2','vpp2','skew2','rms2','mu3','std3','krt3','vpp3','skew3','rms3',
                             'mu4','std4','krt4','vpp4','skew4','rms4','mu5','std5','krt5','vpp5','skew5','rms5','mu6','std6','krt6','vpp6','skew6','rms6',
                             'mu7','std7','krt7','vpp7','skew7','rms7','mu8','std8','krt8','vpp8','skew8','rms8','label'] # Etiquetas cols
  supahData = pd.DataFrame(columns = colsList) # Crear df para llenar
  paso = math.ceil(dfArray.shape[0]/step) # Paso cualsifuera función lineal
  for i in range(paso):
    init = step*i
    finit = init + wid
    try: # Si no me paso
      auxArray = dfArray[init:finit,:]
    except: # Si me paso
      auxArray = dfArray[dfArray.shape[0]-wid:dfArray.shape[0],:]
    fila = [] # Buffer para la fila
    for k in range(9):
      if (k == 0):
        continue
      auxSubArray = auxArray[:,k] # obtener columna k
      fila.append(np.mean(auxSubArray))
      fila.append(np.std(auxSubArray))
      fila.append(kurtosis(auxSubArray))
      fila.append(abs(np.max(auxSubArray)-np.min(auxSubArray)))
      fila.append(skew(auxSubArray))
      fila.append(np.sqrt(np.mean(auxSubArray**2))) # rms
    try: # Si no me paso
      label = dfArray[int((init+finit)/2),:][9]
      if (uniformWindow):
        c = dfArray[init,:][9]-dfArray[finit,:][9]
        if(c!=0):
          continue
    except: # Si me paso
      borderT = dfArray.shape[0]-wid
      label = dfArray[int((dfArray.shape[0] - borderT)/2),:][9]
      
    if (not fullData): # Elegir solo gestos o solo silencios
      if (eliminarCeroSiete):
        if (label == 0 or label == 7):
          continue
      else: # Caso para verificar silencios
        if (label == 0):
          label = 0
        else:
          label = 1
          
    fila.append(label)
    appendDf = pd.DataFrame([fila], columns=colsList) # fila es toda la fila
    new = supahData.append(appendDf, ignore_index=True) # Se realiza append
    supahData = new # Hay que cambiar pues no es insitu
  return supahData


# In[13]:


# Algoritmo para obtener datos del dataset de acuerdo al ancho y el paso
# Se eliminan las ventanas que tienen label distinta en los bordes
# Se eliminan labels 0 y 7, se elige solo gestos
wid = anchoVentana
step = pasoVentana
indexList = []
supahData = pd.DataFrame(columns = ['mu1','std1','krt1','vpp1','skew1','rms1','mu2','std2','krt2','vpp2','skew2','rms2','mu3','std3','krt3','vpp3','skew3','rms3',
                             'mu4','std4','krt4','vpp4','skew4','rms4','mu5','std5','krt5','vpp5','skew5','rms5','mu6','std6','krt6','vpp6','skew6','rms6',
                             'mu7','std7','krt7','vpp7','skew7','rms7','mu8','std8','krt8','vpp8','skew8','rms8','label']) 
listDirSubjects = os.listdir() # Se obtiene la lista de directorios
listDirSubjects.sort() # Se ordena la lista de directorios
for folder in listDirSubjects: # Para cada carpeta en la lista de directorios
  if (folder=='README.txt'): # Si es README.txt continuar iteración o finalizar
    continue
  if(folder=='29'):
    indexList.append(supahData.values.shape[0])
  os.chdir(folder) # Cambiar al directorio
  listTests = os.listdir() # Generar una lista con los archivos dentro del dir.
  for test in listTests: 
    dataSet = pd.read_csv(test, sep='\t') # Almacenar un un dataFrame de Pandas
    dfArray = dataSet.values
    dfToAppend = windows(dfArray,wid,step,True)
    supahData = supahData.append(dfToAppend,ignore_index=True)
  os.chdir("..") # Volver al directorio del que se partió
supahData


# In[ ]:


indexList

supahDataCopy = supahData.copy() # Se realiza una copia de los datos
dataForClassifcation = supahDataCopy.values[0:indexList[0],:] # Data de clasificación se extrae de la copia
np.random.shuffle(dataForClassifcation) # Se shufflean datos de la copia

dataForTesting = supahData.values[indexList[0]:supahData.shape[0],:] # Los datos de testing siempre del original


# In[ ]:


# Se divide la data
# Se genera un escaler
X = dataForClassifcation[:,0:dataForClassifcation.shape[1]-1]
Y = dataForClassifcation[:,dataForClassifcation.shape[1]-1]
scaler = StandardScaler()


# In[ ]:


# Se divide conjunto de entrenamiento y validación
from sklearn.model_selection import train_test_split
xTrain, xValid, yTrain, yValid = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)


# In[17]:


# Selector de Características, se utiliza ANOVA por defecto
selector = SelectKBest(f_classif, k=kDeAnova) # k de ANOVA
selector.fit(xTrain,yTrain)


# In[ ]:


# Proyectando de acuerdo a las características escogidas
xTrain = selector.transform(xTrain)
xValid = selector.transform(xValid)


# In[ ]:


scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xValid = scaler.transform(xValid)
yTrain.astype('int')

# DATOS DE TESTING
xTest = dataForTesting[:,:dataForTesting.shape[1]-1]
yTest = dataForTesting[:,dataForTesting.shape[1]-1]
# Se proyectan las columnas y se escala el conjunto de testing
xTest = selector.transform(xTest)
xTest = scaler.transform(xTest)


# In[ ]:


# Grilla utilizada en la experimentación
parameter_space = {
    'hidden_layer_sizes': [(5,5,5),(10,10,10,10)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05, 1e-5],
    'learning_rate': ['constant','adaptive'],
}
cList = [1,17,119,534,1041, 10245]


# In[ ]:


# Grilla con resultados obtenidos
parameter_space = {
    'hidden_layer_sizes': [(10,10,10,10)],
    'activation': ['relu'],
    'solver': ['lbfgs'],
    'alpha': [1e-5],
    'learning_rate': ['constant'],
}
cList = [1]


# In[22]:


c0 = ("LinearSVM", svm.LinearSVC())
c1 = ("Decision Tree", DecisionTreeClassifier())
c2 = ("Gaussian Naive Bayes", GaussianNB())
c3 = ("KNN", KNeighborsClassifier(n_neighbors=5))
c4 = ('MLP', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10, 10, 10), random_state=1))

parameters = {'C':cList}

classifiers = [c0, c1, c2, c3, c4]
fittedClfs = []

for nombre, clf in classifiers:
  if (nombre == 'LinearSVM'):
    crossedValidatedClf = GridSearchCV(clf, parameters, cv=7)
    fittedClf = crossedValidatedClf.fit(xTrain, yTrain.astype('int'))
    fittedClfs.append(fittedClf)
  elif (nombre == 'MLP'):
    crossedValidatedClf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=7)
    fittedClf = crossedValidatedClf.fit(xTrain, yTrain.astype('int'))
    fittedClfs.append(fittedClf)
  else:
    fittedClf = clf.fit(xTrain, yTrain.astype('int'))
    fittedClfs.append(fittedClf)

clasificadoresDeGesto = fittedClfs


# In[23]:


# <--- CLIQUEA EL PLAY Y MANDAME FOTOS
listaNombreClf = ["LinearSVM", "Decision Tree", "Gaussian Naive Bayes",
                 "KNN", "MLP"]
listaDeAcc = []
i = 0
for fc in clasificadoresDeGesto:#-.-----
  yPred = fc.predict(xValid)
  print("----------------")
  print(listaNombreClf[i])
  print("----------------")
  print("Accuracy:")
  listaDeAcc.append(accuracy_score(yValid.astype('int'), yPred)) # Added
  print(accuracy_score(yValid.astype('int'), yPred))
  print("Confusion Matrix:")
  print(confusion_matrix(yValid.astype('int'), yPred))
  i+=1
  print("----------------\n\n")


# In[24]:


# Elegir ganador
indiceGanador = np.argmax(array(listaDeAcc))


## Data test
xTest = dataForTesting[:,:dataForTesting.shape[1]-1] # Data for testing nunca se modifica y referencia dataset original que no cambia nunca, porque shuffle se hace sobre una copia
## Se agrega
xTest = selector.transform(xTest)
## Se agrega
xTest = scaler.transform(xTest)
yPred = clasificadoresDeGesto[indiceGanador].predict(xTest)
print("----------------")
print("CLASIFICADOR GANADOR")
print(listaNombreClf[indiceGanador])
print("----------------")
print("Accuracy:")
print(accuracy_score(yTest.astype('int'), yPred))
print("Confusion Matrix:")
print(confusion_matrix(yTest.astype('int'), yPred))
print("----------------\n\n")

confusionMatrixClasificador = confusion_matrix(yTest.astype('int'), yPred)


# In[25]:


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
listClasses = ["G1","G2","G3","G4","G5","G6"]
df_cm = pd.DataFrame(confusionMatrixClasificador, index = [i for i in listClasses],
                  columns = [i for i in listClasses])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.5)
sn.heatmap(df_cm, annot=True, cmap="Greens")


# In[26]:


# Algoritmo para obtener datos del dataset de acuerdo al ancho y el paso
# Se eliminan las ventanas que tienen label distinta en los bordes
# No se eliminan labels 0, 7.
# Internamente se transforman los labels distintos de cero a uno
wid = anchoVentana
step = pasoVentana
indexList = []
supahData = pd.DataFrame(columns = ['mu1','std1','krt1','vpp1','skew1','rms1','mu2','std2','krt2','vpp2','skew2','rms2','mu3','std3','krt3','vpp3','skew3','rms3',
                             'mu4','std4','krt4','vpp4','skew4','rms4','mu5','std5','krt5','vpp5','skew5','rms5','mu6','std6','krt6','vpp6','skew6','rms6',
                             'mu7','std7','krt7','vpp7','skew7','rms7','mu8','std8','krt8','vpp8','skew8','rms8','label']) 
listDirSubjects = os.listdir() # Se obtiene la lista de directorios
listDirSubjects.sort() # Se ordena la lista de directorios
for folder in listDirSubjects: # Para cada carpeta en la lista de directorios
  if (folder=='README.txt'): # Si es README.txt continuar iteración o finalizar
    continue
  if(folder=='29'):
    indexList.append(supahData.values.shape[0])
  os.chdir(folder) # Cambiar al directorio
  listTests = os.listdir() # Generar una lista con los archivos dentro del dir.
  for test in listTests: 
    dataSet = pd.read_csv(test, sep='\t') # Almacenar un un dataFrame de Pandas
    dfArray = dataSet.values
    dfToAppend = windows(dfArray,wid,step,True,False)
    supahData = supahData.append(dfToAppend,ignore_index=True)
  os.chdir("..") # Volver al directorio del que se partió
supahData


# In[ ]:


indexList
supahDataCopy = supahData.copy()
dataForClassifcation = supahDataCopy.values[0:indexList[0],:]
np.random.shuffle(dataForClassifcation)
dataForTesting = supahData.values[indexList[0]:supahData.shape[0],:]


# In[ ]:


X = dataForClassifcation[:,0:dataForClassifcation.shape[1]-1]
Y = dataForClassifcation[:,dataForClassifcation.shape[1]-1]
scaler = StandardScaler()


# In[ ]:


from sklearn.model_selection import train_test_split
xTrain, xValid, yTrain, yValid = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)


# In[30]:


# Selector de Características
selector = SelectKBest(f_classif, k=kDeAnova) # k de ANOVA
selector.fit(xTrain,yTrain)


# In[ ]:


# Proyectando
xTrain = selector.transform(xTrain)
xValid = selector.transform(xValid)


# In[32]:


selector


# In[ ]:


scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xValid = scaler.transform(xValid)
yTrain.astype('int')
# DATOS DE TESTING
xTest = dataForTesting[:,:dataForTesting.shape[1]-1]
yTest = dataForTesting[:,dataForTesting.shape[1]-1]

# Se agrega esto
xTest = selector.transform(xTest)
# Se agregó eso

xTest = scaler.transform(xTest)


# In[ ]:


# Grilla con resultados obtenidos
parameter_space = {
    'hidden_layer_sizes': [(10,10,10,10)],
    'activation': ['relu'],
    'solver': ['lbfgs'],
    'alpha': [1e-5],
    'learning_rate': ['constant'],
}
cList = [1]


# In[35]:


c0 = ("LinearSVM", svm.LinearSVC())
c1 = ("Decision Tree", DecisionTreeClassifier())
c2 = ("Gaussian Naive Bayes", GaussianNB())
c3 = ("KNN", KNeighborsClassifier(n_neighbors=5))
c4 = ('MLP', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10, 10, 10), random_state=1))

parameters = {'C':cList}

classifiers = [c0, c1, c2, c3, c4]
fittedClfs = []

for nombre, clf in classifiers:
  if (nombre == 'LinearSVM'):
    crossedValidatedClf = GridSearchCV(clf, parameters, cv=7)
    fittedClf = crossedValidatedClf.fit(xTrain, yTrain.astype('int'))
    fittedClfs.append(fittedClf)
  elif (nombre == 'MLP'):
    crossedValidatedClf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=7)
    fittedClf = crossedValidatedClf.fit(xTrain, yTrain.astype('int'))
    fittedClfs.append(fittedClf)
  else:
    fittedClf = clf.fit(xTrain, yTrain.astype('int'))
    fittedClfs.append(fittedClf)
clasificadoresDeSilencio = fittedClfs


# In[36]:


listaNombreClf = ["LinearSVM", "Decision Tree", "Gaussian Naive Bayes",
                 "KNN", "MLP"]
listaDeAcc = []
i = 0
for fc in clasificadoresDeSilencio:
  yPred = fc.predict(xValid)
  print("----------------")
  print(listaNombreClf[i])
  print("----------------")
  print("Accuracy:")
  listaDeAcc.append(accuracy_score(yValid.astype('int'), yPred)) # Added
  print(accuracy_score(yValid.astype('int'), yPred))
  print("Confusion Matrix:")
  print(confusion_matrix(yValid.astype('int'), yPred))
  i+=1
  print("----------------\n\n")
print(yValid)
print(yPred)


# In[37]:


# Elegir ganador
indiceGanador = np.argmax(array(listaDeAcc))
xTest = dataForTesting[:,:dataForTesting.shape[1]-1]

## Se agrega
xTest = selector.transform(xTest)
## Se agrega

xTest = scaler.transform(xTest)
yPred = clasificadoresDeSilencio[indiceGanador].predict(xTest)
print("----------------")
print("CLASIFICADOR GANADOR")
print(listaNombreClf[indiceGanador])
print("----------------")
print("Accuracy:")
print(accuracy_score(yTest.astype('int'), yPred))
print("Confusion Matrix:")
print(confusion_matrix(yTest.astype('int'), yPred))
i+=1
print("----------------\n\n")

confusionMatrixDetector = confusion_matrix(yTest.astype('int'), yPred)


# In[38]:


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
listClasses = ["Silencio","Gesto"]
df_cm = pd.DataFrame(confusionMatrixDetector, index = [i for i in listClasses],
                  columns = [i for i in listClasses])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.5)
sn.heatmap(df_cm, annot=True, cmap="Greens")


# In[39]:


# Algoritmo para obtener datos del dataset de acuerdo al ancho y el paso
# Se obtienen todas las etiquetas sin cambios
wid = anchoVentana
step = pasoVentana
indexList = []
supahData = pd.DataFrame(columns = ['mu1','std1','krt1','vpp1','skew1','rms1','mu2','std2','krt2','vpp2','skew2','rms2','mu3','std3','krt3','vpp3','skew3','rms3',
                             'mu4','std4','krt4','vpp4','skew4','rms4','mu5','std5','krt5','vpp5','skew5','rms5','mu6','std6','krt6','vpp6','skew6','rms6',
                             'mu7','std7','krt7','vpp7','skew7','rms7','mu8','std8','krt8','vpp8','skew8','rms8','label']) 
listDirSubjects = os.listdir() # Se obtiene la lista de directorios
listDirSubjects.sort() # Se ordena la lista de directorios
for folder in listDirSubjects: # Para cada carpeta en la lista de directorios
  if (folder=='README.txt'): # Si es README.txt continuar iteración o finalizar
    continue
  if(folder=='29'):
    indexList.append(supahData.values.shape[0])
  os.chdir(folder) # Cambiar al directorio
  listTests = os.listdir() # Generar una lista con los archivos dentro del dir.
  for test in listTests: 
    dataSet = pd.read_csv(test, sep='\t') # Almacenar un un dataFrame de Pandas
    dfArray = dataSet.values
    dfToAppend = windows(dfArray,wid,step,True,fullData = True)
    supahData = supahData.append(dfToAppend,ignore_index=True)
  os.chdir("..") # Volver al directorio del que se partió
supahData


# In[ ]:


indexList
supahDataCopy = supahData.copy()
dataForClassifcation = supahDataCopy.values[0:indexList[0],:]
np.random.shuffle(dataForClassifcation)
dataForTesting = supahData.values[indexList[0]:supahData.shape[0],:] 


# In[ ]:


X = dataForClassifcation[:,0:dataForClassifcation.shape[1]-1]
Y = dataForClassifcation[:,dataForClassifcation.shape[1]-1]
scaler = StandardScaler()


# In[ ]:


from sklearn.model_selection import train_test_split
xTrain, xValid, yTrain, yValid = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)


# In[43]:


# Selector de Características
selector = SelectKBest(f_classif, k=kDeAnova) # k de ANOVA
selector.fit(xTrain,yTrain)


# In[ ]:


# Proyectando
xTrain = selector.transform(xTrain)
xValid = selector.transform(xValid)


# In[ ]:


scaler.fit(xTrain)
xTrain = scaler.transform(xTrain)
xValid = scaler.transform(xValid)
yTrain.astype('int')
# DATOS DE TESTING
xTest = dataForTesting[:,:dataForTesting.shape[1]-1]
yTest = dataForTesting[:,dataForTesting.shape[1]-1]

# Se agrega esto
xTest = selector.transform(xTest)
# Se agregó eso

xTest = scaler.transform(xTest)


# In[46]:


listaNombreClf = ["LinearSVM", "Decision Tree", "Gaussian Naive Bayes",
                 "KNN", "MLP"]
listaDeAcc = []

for i in range(5):
  yPredS = clasificadoresDeSilencio[i].predict(xValid) # Silencio
  yPredG = clasificadoresDeGesto[i].predict(xValid)# Gesto
  yPred = []
  for j in range(yPredS.shape[0]):
    yPred.append(yPredS[j]*yPredG[j])
  yPred = array(yPred)
  print("----------------")
  print(listaNombreClf[i])
  print("----------------")
  print("Accuracy:")
  listaDeAcc.append(accuracy_score(yValid.astype('int'), yPred))
  print(accuracy_score(yValid.astype('int'), yPred))
  print("Confusion Matrix:")
  print(confusion_matrix(yValid.astype('int'), yPred))
  i+=1
  print("----------------\n\n")


# In[47]:


# Elegir ganador
indiceGanador = np.argmax(array(listaDeAcc))
xTest = dataForTesting[:,:dataForTesting.shape[1]-1]

## Se agrega ASEGURAR DIMENSIONES CON EL SCALER
xTest = selector.transform(xTest)
## Se agrega

xTest = scaler.transform(xTest)

yPredS = clasificadoresDeSilencio[indiceGanador].predict(xTest)
yPredG = clasificadoresDeGesto[indiceGanador].predict(xTest)
yPred = []
for j in range(yPredS.shape[0]):
  yPred.append(yPredS[j]*yPredG[j])
yPred = array(yPred)

print("----------------")
print("CLASIFICADOR GANADOR EN CONJUNTO DE VALIDACIÓN")
print(listaNombreClf[indiceGanador])
print("----------------")
print("Accuracy:")
print(accuracy_score(yTest.astype('int'), yPred))
print("Confusion Matrix:")
print(confusion_matrix(yTest.astype('int'), yPred))
i+=1
print("----------------\n\n")

combinedConfusionMatrix = confusion_matrix(yTest.astype('int'), yPred)


# In[48]:


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
listClasses = ["S","G1","G2","G3","G4","G5","G6","G7"]
df_cm = pd.DataFrame(combinedConfusionMatrix , index = [i for i in listClasses],
                  columns = [i for i in listClasses])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, cmap="Greens")

