## Archivos Generados

##### model_training\models\model_transformer.py 
El orginial, para entrenar el modelo de transformadores y predecir actividad, rol, timestamp. NO sirve para anomalias.

#### model_training\models\model_transformer_anomaly.py
Entrena el modelo para predecir las siguientes actividades y guarda datos para próximo uso en anomalias.

#### model_prediction\anomaly_detection.py
Detecta anomalias en el siguiente evento utilizando el modelo anteriormente entrenado.

#### READMES
Uno para el merge y el otro para ejecución


##Archivos Modificados

#### models_spec.ini
Últimas lineas modificadas para llamar a los dos modelos de entrenamiento

#### lstm.py
* Línea 10: importar la clase anomaly_detection.py
* Línea 43: Comentario
* Línea 64: Comentario
* Línea 71: Comentario
* Líneas 77-78: Opción anomaly_detection
* Líneas 124-127: Llama a la clase anomaly_detection.py

#### model_training\model_loader.py
* Líneas 17-18: importa los dos modelos de transformadores
* Líneas 31-32: agregué las opciones de los dos modelos

