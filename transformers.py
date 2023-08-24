import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from pandas import DataFrame


class Transformer:
    transformer = ColumnTransformer(transformers=[
        ('drop_date', 'drop', [0]),
        ('open_scaler', StandardScaler(), [1])
    ], remainder=StandardScaler())

    def __init__(self, steps=60):
        self.steps = steps

    def fit(self, X: DataFrame):
        X_ = reverse_data(X)
        self.transformer.fit(X_)

    def transform(self, new_data):

        new_data_ = reverse_data(new_data)
        return self.transformer.transform(new_data_)

    def sequence(self, data):

        rows = data.shape[0]
        X = []
        y = []
        for i in range(self.steps, rows):
            X.append(data[i - self.steps:i, :])
            y.append(data[i, 0])

        X = np.array(X)
        y = np.array(y)
        return X, y

    def get_y_scaler(self):
        return self.transformer.transformers_[1][1]


def reverse_data(data):
    if type(data) == DataFrame:
        data_ = data.iloc[::-1]
    else:
        data_ = data[::-1]
    return data_
