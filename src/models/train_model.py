import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

taxi_train = pd.read_csv("./data/processed/taxi_train_1000000_filas.csv")

numeric_feat = [
    "pickup_weekday",
    "pickup_hour",
    'work_hours',
    "pickup_minute",
    "passenger_count",
    'trip_distance',
    'trip_time',
    'trip_speed'
]
categorical_feat = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]
features = numeric_feat + categorical_feat
target_col = "high_tip"

rfc = RandomForestClassifier(n_estimators=100, max_depth=10)

rfc.fit(taxi_train[features], taxi_train[target_col])



# Calculamos F1:

preds = rfc.predict_proba(taxi_train[features])
preds_labels = [p[1] for p in preds.round()]
print(f'F1: {f1_score(taxi_train[target_col], preds_labels)}')

# Exportamos modelo:
import joblib

joblib.dump(rfc, "random_forest.joblib")
