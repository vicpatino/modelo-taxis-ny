# -*- coding: utf-8 -*-
"""01_import_dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rcUc9o7V1UvqBzkUXA4SI7HMNr3PopaV

# Clasificador de Propinas para Viajes en Taxi en NYC (2020)

Inspirado en la charla ["Keeping up with Machine Learning in Production"](https://github.com/shreyashankar/debugging-ml-talk) de [Shreya Shankar](https://twitter.com/sh_reya)

Este notebook muestra la construcción de un modelo de machine learning de juguete, usando datos de viajes de los taxis amarillos de Nueva York para el año 2020, [proporcionados por la NYC Taxi and Limousine Commission (TLC)](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

La idea es encontrar aquellos viajes donde la propina dejada por el pasajero fue alta, es decir, mayor al 20% del costo del viaje.

Para ello ajustaremos un modelo de classificación binaria RandomForest usando los datos de los viajes de enero de 2020. Probaremos el modelo resultante sobre los datos de los viajes de febrero de 2020. Compararemos el desempeño del modelo en ambos casos usando la métrica de [f1-score](https://en.wikipedia.org/wiki/F-score).

**Este notebook está construido para ser ejecutado en [Google Colab](https://colab.research.google.com/), al que podemos acceder de manera gratuita solo teniendo un usuario de Google (Gmail) y un navegador web. No es necesario instalar nada en el computador local.**

## Cargando las librerías necesarias
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

"""## Leemos los datos de enero 2020 (entrenamiento)

Dado que los datos están en la web creamos una función que, recibiendo como input el mes y año a descargar la data, retorne un dataframe con la información de los viajes en taxi en NY para dicho periodo
"""

def load_data(year: int, month: int) -> pd.DataFrame:
    """Read the data for a given year and month."""
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet"
    return pd.read_parquet(filename)



taxi = load_data(2020, 1)

taxi.head()

taxi.shape



"""## Clasificación binaria de datos tabulares usando Random Forest

### Creando la función predict_taxi_trip

Vamos a crear el método `predict_taxi_trip` que toma como entradas un **array con los valores de las características del viaje** y un **umbral de confianza**. La función determinará si el viaje es de clase propina alta o propina baja, devolviendo un 1 o un 0 dependiendo del caso.

La salida del modelo es un vector de probabilidades de pertenencia del viaje a alguna de las dos clases posibles. El último argumento de entrada a nuestra función (el nivel de confianza) será el umbral que dichas probabilidades deben superar para determinar que el viaje en cuestión si representa uno de propina alta. Por defecto `predict_taxi_trip` usa el valor 0.5 para esto.
"""



from google.colab import drive
drive.mount('/content/drive')

import joblib
import numpy as np

rfc = joblib.load("/content/drive/MyDrive/random_forest.joblib")

def predict_taxi_trip(features_trip, confidence=0.5):
    """Recibe un vector de características de un viaje en taxi en NYC y predice
       si el pasajero dejará o no una propina alta.

    Argumentos:
        features_trip (array): Características del viaje, vector de tamaño 11.
        confidence (float, opcional): Nivel de confianza. Por defecto es 0.5.
    """

    pred_value = rfc.predict_proba(features_trip.reshape(1, -1))[0][1]
    if pred_value >= confidence:
      return 1
    else:
      return 0

"""Probemos sobre un viaje de ejemplo:"""

features_trip = np.array([5.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.7000000e+01,
       1.0000000e+00, 2.5999999e+00, 7.7700000e+02, 3.3462034e-03,
       1.4500000e+02, 7.0000000e+00, 1.0000000e+00])

predict_taxi_trip(features_trip, 0.5)

predict_taxi_trip(features_trip) #si no especifico confidence, por default es 0.5

"""#### Cambiando el nivel de confianza"""

predict_taxi_trip(features_trip, 0.4)

predict_taxi_trip(features_trip, 0.6)

"""## Implementando el modelo usando fastAPI


### Poniendo nuestro modelo de clasificación de viajes en un servidor

### Conceptos importantes

#### Modelo Cliente-Servidor

Cuando hablamos de **implementar**, lo que usualmente se quiere decir es que vamos a poner todo el software necesario para realizar predicciones en un `server` (servidor). De esta forma un `client` (cliente) puede interactuar con el modelo enviando `requests` (solicitudes) al servidor.

Lo importante entonces es que el modelo de Machine Learning vive en un servidor esperando por clientes que le envíen solicitudes de predicciones. El cliente tiene que entregarle toda la información necesaria que el modelo necesita para poder hacer una predicción. Tengamos en mente que es común acumular más de una predicción en una misma solicitud. El servidor usará la información que le proporcionen para devolver predicciones al cliente, el cual puede usarlas a su antojo.

Empecemos creando una instancia de la clase `FastAPI`:

```python
app = FastAPI()
```

El siguiente paso es usar esa instancia para crear endpoints que manejarán la lógica para hacer predicciones. Una vez que todo el código está listo para correr el servidor solo hay que usar el siguiente comando:

```python
uvicorn.run(app)
```

La API está construida usando código de fastAPI pero "servirla" se hace mediante [`uvicorn`](https://www.uvicorn.org/), que es una Asynchronous Server Gateway Interface (ASGI) de muy rápida implementación. Ambas tecnologías están super conectadas pero no necesitamos entender los detalles técnicos. Sólo hay que tener en cuenta que es uvicorn el que se encarga de servir el código.

#### Endpoints

Podemos hospedar varios modelos de Machine Learning en el mismo servidor. Para esto podemos asignarles un `endpoint` diferente a cada modelo para que sepamos siempre cuál de los modelos estamos usando. Un endpoint se representa como un patrón en la `URL`. Por ejemplo si tenemos un sitio que se llama `misupermodelo.com` también podríamos tener tres diferentes modelos en los siguientes endpoints:

- `misupermodelo.com/contador-autos/`
- `misupermodelo.com/predictor-serie-de-tiempo/`
- `misupermodelo.com/recomendador-de-autos/`

Cada modelo llevaría a cabo la tarea que el patrón de la URL indica.

En fastAPI podemos definir un endpoint creando una función que se encargue de manejar la lógica que corresponde. Además se incluye un [decorador](https://www.python.org/dev/peps/pep-0318/) con una función que contiene la información de que método HTTP está permitido y cuál es el patrón de la URL que se usará para el endpoint en cuestión.

El siguiente ejemplo muestra como generar una solicitud HTTP GET en el endpoint endpoint "/mi-endpoint":

```python
@app.get("/mi-endpoint")
def handle_endpoint():
    ...
    ...
```


#### Solicitudes HTTP

El cliente y el servidor se comunican entre sí a través de un protocolo llamado `HTTP`. El concepto clave es que la comunicación entre cliente y servidor usa ciertos verbos que denotan acciones. Dos verbos comunes son:

- `GET` -> Obtiene información del servidor.
- `POST` -> Entrega información al servidor, la cual se usa para responder.

Si el cliente hace un `GET request` a un endpoint el servidor entregará información del endpoint sin necesidad de que le proporcionemos información adicional. En el caso de un `POST request` le estamos diciendo de manera explícita al servidor que le entregaremos información para que la procese de alguna forma.

Para interactuar con modelos de Machine Learning que estén viviendo en endpoints usualmente hacemos un `POST request` ya que siempre necesitaremos entregarle información para que realice predicciones.

Así luce un POST request:

```python
@app.post("/mi-otro-endpoint")
def handle_other_endpoint(param1: int, param2: str):
    ...
    ...

```

Para POST requests, la función debe contener parámetros. En contraste con un GET, las solicitudes POST esperan que el cliente les entregue alguna información. En este ejemplo proveerá un entero y un string.


### ¿Por qué fastAPI?

Con fastAPI podemos crear servidores web par hospedar modelos de manera muy sencilla. Adicionalmente la plataforma es extremadamente rápida.
"""

import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel #facilita validación de datos (define modelos de datos que se pueden validar)

# Asignamos una instancia de la clase FastAPI a la variable "app".
# Interacturaremos con la API usando este elemento.
app = FastAPI(title='Implementando un modelo de Machine Learning usando FastAPI')

# Creamos una clase para el vector de features de entrada
class Item(BaseModel):
    pickup_weekday: float
    pickup_hour: float
    work_hours: float
    pickup_minute: float
    passenger_count: float
    trip_distance: float
    trip_time: float
    trip_speed: float
    PULocationID: float
    DOLocationID: float
    RatecodeID: float

# Usando @app.get("/") definimos un método GET para el endpoint / (que sería como el "home").
@app.get("/")
def home():
    return "¡Felicitaciones! Tu API está funcionando según lo esperado. Anda ahora a http://localhost:8000/docs."


# Este endpoint maneja la lógica necesaria para clasificar.
# Requiere como entrada el vector de características del viaje y el umbral de confianza para la clasificación.
@app.post("/predict")
def prediction(item: Item, confidence: float):


    # 1. Correr el modelo de clasificación
    features_trip = np.array([item.pickup_weekday, item.pickup_hour, item.work_hours, item.pickup_minute, item.passenger_count, item.trip_distance,
                    item.trip_time, item.trip_speed, item.PULocationID, item.DOLocationID, item.RatecodeID])
    pred = predict_taxi_trip(features_trip, confidence)

    # 2. Transmitir la respuesta de vuelta al cliente

    # Retornar el resultado de la predicción
    return {'predicted_class': pred}

"""¡Corriendo la celda que viene echaremos a andar el servidor!

Esto causará que el notebook se bloquee (no podremos correr más celdas) hasta que interrumpamos de forma manual el kernel. Podemos hacer eso haciendo click en la pestaña `Kernel` y luego `Interrupt`.
"""

# Esto deja correr al servidor en un ambiente interactivo como un Jupyter notebook
nest_asyncio.apply()

# Donde se hospedará el servidor
host = "127.0.0.1"

# ¡Iniciemos el servidor!
uvicorn.run(app, host=host, port=8000)

"""¡El servidor está corriendo! Vamos a [http://localhost:8000/](http://localhost:8000/) para verlo en acción.

**Probemos enviando un viaje de ejemplo** y veamos como nuestra API es capaz de clasificarlo y retornar la etiquetas del tipo de viaje. **Podemos hacer eso visitando [http://localhost:8000/docs](http://localhost:8000/docs) para abrir un cliente que viene dentro de fastAPI.**

Si hacemos click en el endpoint `/predict` se verán más opciones. Para probar el servidor hay que usar el botón **Try it out**.

Podemos elegir un nivel de confianza usando el campo **confidence** y representar un **viaje** completando el diccionario del **Request body**.

## Consumiendo el modelo desde otro cliente

Es genial que fastAPI permita interactuar con la API por medio del cliente que tiene incorporado. Pero debemos aprender como usar la API desde cualquier tipo de código, no necesariamente con una interfaz.

Para eso iremos al siguiente notebook donde implementaremos un cliente básico en Python. Para esto **debemos dejar corriendo el servidor (no paremos el kernel ni cerremos esta ventana)** y abramos el notebook `02_client.ipynb` notebook.
"""