#Descomentar está  sección en caso de correr en Colab
'''
#from google.colab import drive
#drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
#put your own path in google drive
# %cd "/content/drive/MyDrive/7 semestre/Uresti/1/dataset_mod2"
#!ls
'''

"""## Librerías**"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

"""## Lectura de datos**

Se hace la lectura de datos "dataset.csv". Fuente: https://www.kaggle.com/datasets/naveenkumar20bps1137/predict-students-dropout-and-academic-success
"""

df = pd.read_csv('dataset.csv')
print(df.head())

"""# Preprocesamiento de datos**

## Forma del dataset**
"""

#df.shape

# Tipo de datos
print(df.info())

df.describe().round(3)

# Para hacer una división de las columnas que son numéricas y categóricas, se hace desde el principio el select de las columnas

categoricas = df.select_dtypes(include='object').columns
numericas = df.select_dtypes(include='number').columns

print("Variables categóricas: ", categoricas)
print("Variables numéricas: ", numericas)

"""## Tratamiento de datos faltantes"""

# Conteo de las variables faltantes
df.isnull().sum()/df.shape[0]*100 #porcentaje

# Se verifica si hay datos duplicados
print('Total Duplicates: ', df.duplicated().sum())

# se puede observar que no hay datos duplicados por lo que no hay que tratar los datos en cuestión

"""## Exploración de datos**"""

# Observamos la variable a predecir para observar su distribución
df['Target'].value_counts()

# Graficamos su distribución para observar si se trata de una distribucón sesgada o simétrica
sns.countplot(df, x='Target')

# Debido a que solo nos interesa si el estudiante se gradúa o no eliminamos "Enrolled"
# Eesto debido a que el estudiante todavía puede seguir o dejar la escuela
df=df[df.Target!='Enrolled']

# Verificamos que se haya reducido su dimensión
print(df.shape)

# Graficamos nuevamente la distribución de "Target" con los nuevos valores
sns.countplot(df, x='Target')

"""## Tratamiento de datos**"""

# Cambiamos los valores categóricos a numéricos
# Se hace el cambio únicamente de "target"

# One-hot encode
df = pd.get_dummies(df, columns=categoricas)
print(df.head())

df.drop("Target_Graduate", axis=1, inplace=True)
print(df.head())

"""## Correlaciones**"""

# Se observan las correlaciones existentes
matriz_correlacion = df.corr()

"""### Eliminación de columnas irrelevantes para el modelo"""

umbral = 0.7

alto_umbral = []

for col in matriz_correlacion.columns:
    correlated_cols = matriz_correlacion.index[matriz_correlacion[col] >= umbral].tolist()
    correlated_cols.remove(col)
    for correlated_col in correlated_cols:
        pair = (col, correlated_col)
        alto_umbral.append(pair)

print(alto_umbral)

# Se decide eliminar las columnas que están altamente correlacionadas en este caso se hace un estudio del par de variables correlacionadas para tomar la decisión

columnas_a_eliminar = ['Nacionality', 'Curricular units 1st sem (enrolled)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 1st sem (grade)','Curricular units 2nd sem (enrolled)',  'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']

# Eliminar las columnas especificadas
df = df.drop(columnas_a_eliminar, axis=1)

# Se elimina Nacionalidad debido a que se prefiere quedar con una menor granularidad para tener un mayor nivel de explicatividad del modelo (se queda con Internacionalidad 1 en internacional, 0 portugal)
# Se elimina Curricular units 1st sem (enrolled) debido a que no nos interesa lo que se metió sino lo que se cumplió
# Se elimina Curricular units 2nd sem (credited) debido a que ya hicimos la relación anterior y por lo mismo se decidió elegir
# Al estar relacionaod con el de primer semestre se elimina el 'Curricular units 2nd sem (evaluations)'

print(df.shape) #se verifica que se hayan eliminado correctamente las columnas, por lo tanto ya no hay correlaciones existentes

# MODELO: Análisis discriminante lineal**

target = df['Target_Dropout']

# En caso de querer siempre la misma corrida
# x_train,x_test,y_train,y_test = train_test_split(df.drop('Target_Dropout', axis =1), target, test_size=0.3, random_state=0)
# Se va modificando la semilla de manera automática
x_train,x_test,y_train,y_test = train_test_split(df.drop('Target_Dropout', axis =1), target, test_size=0.3)

#x_train.shape

#x_test.shape

#y_train.shape

#y_test.shape

"""## Calculo modelo LDA**

Se hace uso de la fórmula antes vista para poder aplicar el análisis discriminante lineal.
"""

# Se comienza por generar una partición de muestra del entrenamiento de ambas clases (todo a partir del conjunto train)
df_dropout = x_train.copy()
df_dropout['Target_Dropout'] = y_train
df_grad = df_dropout[df_dropout['Target_Dropout']==0]
df_dropout = df_dropout[df_dropout['Target_Dropout']==1]

# Verificación de partición correcta
df_grad.Target_Dropout.value_counts() #se pbserva que grad se clasififca correctamente en "1"

# Se realiza esto para no tener la variable target en x (es decir, no se calcula la target en la media de x)
df_dropout.drop('Target_Dropout', axis =1, inplace=True)
df_grad.drop('Target_Dropout', axis =1, inplace=True)

# Media por clase
media_grad = df_grad.mean()
media_dropout = df_dropout.mean()

print("La media de grad es: ", str(media_grad))

print("La media de dropout es: ", str(media_dropout))

# Se transponen los datos para obtener la matriz de covarianza a partir de las columnas y no de las filas
data_transposed = np.transpose(x_train)

# Matriz de varianza y covarianga (sigma) (de todo)
var_covar = np.cov(data_transposed)
var_covar.shape

# Posteriormente se calcula la matriz inversa
inverse_varcovar = np.linalg.inv(var_covar)
print(inverse_varcovar.shape)

# Ya se tiene todo lo necesartio para realizar el modelo

# Se calcula la alpha
alpha = np.dot(np.transpose(media_dropout-media_grad), inverse_varcovar)
print(alpha.shape)

# Se obtiene la constante (promedio de ambas medias, punto medio)
constante = (-1/2)*(media_grad+media_dropout)
print(constante.shape)
# Regrese el punto medio por variable

# El constante final será alpha * punto medio (beta)
constante_final = np.dot(alpha,constante)
print(constante_final)

# Se multiplica cada fila o elemento x por lo obtenido en alpha y la matriz inversa, sumando la constante
# si es mayor a 0 se trata de clasificador dropout, de lo contrario (menor a 0) se trata de grad.
# se agrega un contador para observar el porcentaje de error

y_pred = []

for i in range(0,x_train.shape[0]):
  ej = x_train.iloc[i]
  y_pred.append(np.dot(alpha,ej) + constante_final)

# Aquí se encuentran todas las predicciones ubicadas en la muestra de entrenamiento

# Agregar predicciones a y_train

# Se comienza por convertir la list a un DF
y_pred = pd.DataFrame(y_pred)
y_train = pd.DataFrame(y_train)
y_train.reset_index(drop=True, inplace=True) # se resetea la variable para que respete el id

y_train['pred'] = y_pred[0]
y_train['pred_clase'] = np.where(y_train['pred'] >= 0,1,0)

#display(y_train)

# Impresión del reporte de clasificación para observar el desempeño del modelo (train)
print(confusion_matrix(y_train['Target_Dropout'], y_train['pred_clase']))
print(classification_report(y_train['Target_Dropout'], y_train['pred_clase']))

"""### Resultado de diferentes corridas

Accuracy
Corrida 1: 0.88
Corrida 2: 0.87
Corrida 3: 0.88
Corrida 4: 0.88
Corrida 5: 0.87

## Aplicación del modelo en la base de prueba
"""

# Aplicación del modelo para la base de datos de prueba (test), permite verificar el desempeño de modleo con infromación con la qu eno fue entrenada
y_pred_list = []

for i in range(0,x_test.shape[0]):
  ej = x_test.iloc[i]
  y_pred_list.append(np.dot(alpha,ej) + constante_final)

# Agregar predicciones a y_train

# Se comienza por convertir la list a un DF
y_pred = pd.DataFrame(y_pred_list)
y_test = pd.DataFrame(y_test)
y_test.reset_index(drop=True, inplace=True) # se resetea la variable para que respete el id

y_test['pred'] = y_pred[0]
y_test['pred_clase'] = np.where(y_test['pred'] >= 0,1,0)

# Impresión del reporte de clasificación para observar el desempeño del modelo (test)
print(confusion_matrix(y_test['Target_Dropout'], y_test['pred_clase']))
print(classification_report(y_test['Target_Dropout'], y_test['pred_clase']))

"""### Resultado de diferentes corridas

Accuracy
Corrida 1: 0.86
Corrida 2: 0.88
Corrida 3: 0.87
Corrida 4: 0.89
Corrida 5: 0.87
"""