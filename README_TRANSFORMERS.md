## Next Event Prediction (Activity, Rol, Timestamp)

De la misma forma que siempre lo han hecho

### Entrenamiento
1. activity: training; model_type: transformer
2. Los resultados y el modelo quedan de la misma forma: output_files\training_times & output_files\20201218....
Lo que si va a cambiar es la forma en como se guarda el modelo, ahora queda dentro de una carpeta llamada 'transf_model' dentro de la carpeta "20201218...", la cual se va a poner en vez de el archivo .h5 en model_file.

### Predicción
1. activity: predict_next; folder: 20201218... ; model_file: transf_model.
2. Los resultados quedan de la misma forma (en los csv).


## Anomaly Detection

### Entrenamiento
1. activity: training; model_type: transformer_anomaly
2. Los resultados van a quedar igual que al otro modelo de transformador. Lo único que va a camibar es que en la carpeta de "parameters" van a ver varios archivos que se van a utilizar en la detección de anomalias.

### Detección de anomalias
1. activity: anomaly_detection; folder: 20201218... 
2. Se genera el archivo "anomaly_detection.csv", donde se muestran todos los resultados (Accuracy, f1, etc).