

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

import sys
sys.path.append("..")

# cargamos función load_data creada en script import_dataset.py
from data.import_dataset import load_data
from data.features import preprocess

#importar modelo:
loaded_rfc = joblib.load("random_forest.joblib")

#importamos data a predecir, por ejemplo, la de marzo 2020
taxi = load_data(2020,3)

# Preprocesamos
target_col = "high_tip"
taxi_test = preprocess(df=taxi, target_col=target_col)

#Ejecutamos predicciones:
preds_test = loaded_rfc.predict_proba(taxi_test[features])

#calculamos f1-score:
preds_test_labels = [p[1] for p in preds_test.round()]
print(f'F1: {f1_score(taxi_test[target_col], preds_test_labels)}')
