import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from pandas import DataFrame


def format_data(data: DataFrame):
    data_ = data.iloc[::-1]
    transformer = create_pipeline()
    transformer.fit(data_)
    data_ = transformer.transform(data_)
    return transformer, data_

def create_sequences(Xnd: np.ndarray, steps=60):
    rows = Xnd.shape[0]
    X = []
    y = []
    for i in range(steps, rows):
        X.append(Xnd[i - steps:i, :])
        y.append(Xnd[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y


def create_pipeline() -> ColumnTransformer:
    transformer = ColumnTransformer(transformers=[
        ('drop_date', 'drop', [0]),
        ('open_scaler', StandardScaler(), [1])
    ], remainder=StandardScaler())
    return transformer
