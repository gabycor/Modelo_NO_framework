# Modelo_NO_framework


Implementación del modelo "Análisis discriminante lineal" para la clasificación de variables objetivo en la predicción del abandono escolar.

**DESCRIPCIÓN Y OBJETIVO**


Este proyecto se centra en la implementación y aplicación del modelo de "Análisis Discriminante Lineal" con el objetivo de clasificar la variable objetivo, que indica si un individuo completará sus estudios o abandonará antes de la graduación. Los esfuerzos se concentran en predecir de manera precisa y efectiva las dos posibles categorías de resultados: aquellos que culminan exitosamente y aquellos que abandonan el proceso educativo.

La fuente de datos proviene de un conjunto de datos proveniente de kaggle (link: https://www.kaggle.com/code/malik9/student-dropout-prediction-with-91-7-accuracy). El conjunto de datos abarca información disponible en el momento de la inscripción del estudiante (trayectoria académica, datos demográficos y factores socioeconómicos) y el rendimiento académico de los estudiantes al final del primer y segundo semestre.

Este conjunto de datos proporciona el fundamento necesario para desarrollar y entrenar el modelo de análisis discriminante lineal, aprovechando sus características y atributos para realizar predicciones informadas sobre el abandono escolar. En las secciones siguientes, se presentarán detalladamente las instrucciones esenciales para ejecutar el código asociado a este proyecto. Además, se ofrecerán descripciones exhaustivas sobre los aspectos clave del modelo de análisis discriminante lineal, subrayando su funcionamiento y relevancia en el contexto de la predicción de resultados educativos.

El objetivo se centra en la implementación del modelo, para comprender el mecanismo interno y su capacidad para aportar información valiosa en la toma de decisiones relacionadas con la retención estudiantil. 

**DESCRIPCIÓN GENERAL DEL MODELO**

El supuesto que se utiliza para realizar el análisis discriminante es:

> $S^{\sim }N(\overrightarrow{\mu _{0}}, \sum ) :\Omega _{0}$

> $S^{\sim }N(\overrightarrow{\mu _{1}}, \sum ) :\Omega _{01}$

Esto para:

> $f_{1}(x) = ke^{\frac{-1}{2}(\overrightarrow{x}-\overrightarrow{\mu _{i}})^{-t}\sum ^{-1}(\overrightarrow{x}-\overrightarrow{\mu _{i}})}$


De lo anterior, a través del coeficiente de verosimilitudes de Neyman Pearson, se tiene la fórmula y el supuesto en el cual se trabajará:


> $S = ln\frac{f_{1}(x)}{f_{0}(x)} = (\overrightarrow{\mu _{1}-\mu _{0}})^{t}\sum^{-1}[x-\frac{1}{2}(\overrightarrow{\mu _{1}}+\overrightarrow{\mu _{0}})]$

Clasifica a **s** en: 

> \Omega _{1} \quad si \quad \boldsymbol{s}> 0
> 
> \Omega _{0} \quad si \quad \boldsymbol{s}\leq  0\end{matrix}




**INSTRUCCIONES**


  1. Descargue el modelo en python (archivo.py) con el nombre "mod2_modeloA01751655".
  2. Descargue la base de datos con el nombre "dataset.csv"
  3. Abra en el entorno de desarrollo integrado de su preferencia el nombre del archivo.
  4. Modifique la ruta de lectura por la ruta de lectura que se encuentra localmente en su computador.
  5. Corra el modelo y verifique el desempeño del modelo en las últimas líneas de código donde se muestra el reporte de clasificación.
  *Opcional*: Descomente las líneas de código de su preferencia para obtener diferentes formas de visualización de los resultados. 
